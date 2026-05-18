"""
Debug evaluation callback for Gemma4 finetuning.

At each eval step, prints for the first example in the batch:
  1. Full decoded prompt (user turn)
  2. Full decoded target (model turn, thinking + answer)
  3. Per-token label sequence (MASKED / token_text)
  4. Per-segment loss: thinking loss, answer loss, total loss
"""

from __future__ import annotations

import torch
import numpy as np
from transformers import TrainerCallback, TrainerState, TrainerControl


class DebugEvalCallback(TrainerCallback):
    """
    Attach to SFTTrainer to get per-segment loss diagnostics at eval steps.

    Usage:
        trainer.add_callback(DebugEvalCallback(tokenizer, ignore_index=-100))
    """

    def __init__(self, tokenizer, ignore_index: int = -100, max_examples: int = 1):
        self.tokenizer     = tokenizer
        self.ignore_index  = ignore_index
        self.max_examples  = max_examples

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        eval_dataloader=None,
        **kwargs,
    ):
        if eval_dataloader is None or model is None:
            return

        model.eval()
        device = next(model.parameters()).device

        batch = next(iter(eval_dataloader))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        n = min(self.max_examples, batch["input_ids"].shape[0])

        with torch.no_grad():
            outputs = model(
                input_ids      = batch["input_ids"][:n],
                attention_mask = batch["attention_mask"][:n],
                labels         = batch["labels"][:n],
            )

        logits = outputs.logits  # (n, L, V)

        for i in range(n):
            self._print_example(
                idx         = i,
                step        = state.global_step,
                input_ids   = batch["input_ids"][i],
                labels      = batch["labels"][i],
                logits      = logits[i],
            )

    # ------------------------------------------------------------------

    def _print_example(
        self,
        idx:       int,
        step:      int,
        input_ids: torch.Tensor,
        labels:    torch.Tensor,
        logits:    torch.Tensor,
    ):
        tok    = self.tokenizer
        ids    = input_ids.tolist()
        lbls   = labels.tolist()
        L      = len(ids)

        # Per-token cross-entropy (shifted: logits[t] predicts ids[t+1])
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (L, V)

        print(f"\n{'='*70}")
        print(f"[DebugEval] step={step} | example={idx}")
        print(f"{'='*70}")

        # ---- Decoded prompt (everything before first supervised token) ----
        first_supervised = next((k for k, l in enumerate(lbls) if l != self.ignore_index), L)
        prompt_ids  = ids[:first_supervised]
        print(f"\n[PROMPT decoded]:\n{tok.decode(prompt_ids, skip_special_tokens=False)}\n")

        # ---- Decoded target ----
        target_ids = [i for i, l in zip(ids, lbls) if l != self.ignore_index]
        print(f"[TARGET decoded]:\n{tok.decode(target_ids, skip_special_tokens=False)}\n")

        # ---- Per-token label sequence ----
        print("[LABEL sequence] (pos: token | label):")
        for pos in range(L):
            tok_text = tok.decode([ids[pos]], skip_special_tokens=False)
            lbl_text = "MASKED" if lbls[pos] == self.ignore_index else repr(tok_text)
            # Only print supervised tokens + a small context window
            if lbls[pos] != self.ignore_index:
                print(f"  [{pos:4d}] {repr(tok_text):30s} -> {lbl_text}")

        # ---- Per-segment loss ----
        # Detect segments by looking for <think> / </think> in supervised range
        think_open_str  = "<think>"
        think_close_str = "</think>"

        think_losses  = []
        answer_losses = []
        in_thinking   = False

        for pos in range(L - 1):   # logits[pos] predicts ids[pos+1]
            if lbls[pos + 1] == self.ignore_index:
                continue
            token_text = tok.decode([ids[pos + 1]], skip_special_tokens=False)
            if think_open_str in token_text:
                in_thinking = True
                continue
            if think_close_str in token_text:
                in_thinking = False
                continue
            loss_val = -log_probs[pos, ids[pos + 1]].item()
            if in_thinking:
                think_losses.append(loss_val)
            else:
                answer_losses.append(loss_val)

        def _fmt(losses):
            if not losses:
                return "n/a (0 tokens)"
            return f"{np.mean(losses):.4f}  (n={len(losses)})"

        print(f"\n[PER-SEGMENT LOSS]")
        print(f"  thinking : {_fmt(think_losses)}")
        print(f"  answer   : {_fmt(answer_losses)}")
        all_losses = think_losses + answer_losses
        print(f"  total    : {_fmt(all_losses)}")
        print(f"{'='*70}\n")

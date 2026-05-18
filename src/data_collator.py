"""
Gemma4 DataCollator with per-token loss masking.

Gemma4 chat template with thinking:
  <bos><start_of_turn>user
  {prompt}<end_of_turn>
  <start_of_turn>model
  <think>
  {thinking}
  </think>

  {answer}<end_of_turn>

Loss targets:
  - Prompt (user turn): always masked (ignore_index)
  - Thinking block: supervised only when USETHINKING=True
  - Answer: always supervised
  - All scaffold tokens (<start_of_turn>, <end_of_turn>, <think>, </think>): masked unless
    supervise_stop_tokens=True
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizerBase


# ---------- Gemma4 thinking scaffold strings ----------
THINK_OPEN  = "<think>"
THINK_CLOSE = "</think>"


def _find_subsequence(ids: List[int], pattern: List[int]) -> List[int]:
    """Return all start indices where pattern appears in ids."""
    positions = []
    n, m = len(ids), len(pattern)
    for i in range(n - m + 1):
        if ids[i : i + m] == pattern:
            positions.append(i)
    return positions


@dataclass
class Gemma4ThinkingCollator:
    """
    Custom DataCollator for Gemma4 with thinking-channel loss masking.

    Args:
        tokenizer:            HF tokenizer for the Gemma4 model.
        max_length:           Maximum sequence length (pad/truncate to this).
        use_thinking:         If True, thinking block tokens are supervised.
        supervise_stop_tokens: If True, <think>/<end_of_turn> tokens at
                              segment boundaries are included in loss.
        ignore_index:         Label value for masked (non-supervised) positions.
    """
    tokenizer:             PreTrainedTokenizerBase
    max_length:            int = 2048
    use_thinking:          bool = False
    supervise_stop_tokens: bool = False
    ignore_index:          int = -100

    # filled in __post_init__
    _think_open_ids:  List[int] = field(default_factory=list, init=False, repr=False)
    _think_close_ids: List[int] = field(default_factory=list, init=False, repr=False)
    _model_turn_ids:  List[int] = field(default_factory=list, init=False, repr=False)
    _eot_ids:         List[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        tok = self.tokenizer
        def _enc(s: str) -> List[int]:
            return tok(s, add_special_tokens=False)["input_ids"]

        self._think_open_ids  = _enc(THINK_OPEN)
        self._think_close_ids = _enc(THINK_CLOSE)
        self._model_turn_ids  = _enc("<start_of_turn>model\n")
        self._eot_ids         = _enc("<end_of_turn>")

        print(f"[Collator] think_open_ids   = {self._think_open_ids}")
        print(f"[Collator] think_close_ids  = {self._think_close_ids}")
        print(f"[Collator] model_turn_ids   = {self._model_turn_ids}")
        print(f"[Collator] eot_ids          = {self._eot_ids}")
        print(f"[Collator] use_thinking     = {self.use_thinking}")
        print(f"[Collator] supervise_stop   = {self.supervise_stop_tokens}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        features: list of dicts with keys: question, meta, thinking, answer
        Returns: input_ids, attention_mask, labels — all (B, max_length).
        """
        encoded = [self._encode_example(f) for f in features]
        return self._pad_and_batch(encoded)

    # ------------------------------------------------------------------
    # Per-example encoding
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str, meta: str) -> str:
        parts = [question.strip()]
        if meta and meta.strip():
            parts.append(meta.strip())
        return "\n\n".join(parts)

    def _encode_example(self, ex: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer
        prompt   = self._build_prompt(ex.get("question", ""), ex.get("meta", ""))
        thinking = ex.get("thinking", "").strip()
        answer   = ex.get("answer", "").strip()

        if self.use_thinking and thinking:
            full_text = (
                tok.apply_chat_template(
                    [
                        {"role": "user",  "content": prompt},
                        {"role": "model", "content": f"{THINK_OPEN}\n{thinking}\n{THINK_CLOSE}\n\n{answer}"},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        else:
            full_text = (
                tok.apply_chat_template(
                    [
                        {"role": "user",  "content": prompt},
                        {"role": "model", "content": answer},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        enc = tok(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids   = enc["input_ids"][0]        # (L,)
        attn_mask   = enc["attention_mask"][0]   # (L,)
        labels      = self._build_labels(input_ids, attn_mask)

        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

    # ------------------------------------------------------------------
    # Label masking
    # ------------------------------------------------------------------

    def _build_labels(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        ids   = input_ids.tolist()
        L     = len(ids)
        labels = torch.full((L,), self.ignore_index, dtype=torch.long)

        # Locate model turn start (<start_of_turn>model\n)
        model_starts = _find_subsequence(ids, self._model_turn_ids)
        if not model_starts:
            # Fallback: supervise everything after halfway (shouldn't happen)
            half = L // 2
            for k in range(half, L):
                if attn_mask[k].item() == 1:
                    labels[k] = ids[k]
            return labels

        # The first model turn is the response we want to supervise.
        model_start_idx = model_starts[0]
        response_start  = model_start_idx + len(self._model_turn_ids)

        if self.use_thinking:
            # Find <think> open inside response
            think_opens  = _find_subsequence(ids[response_start:], self._think_open_ids)
            think_closes = _find_subsequence(ids[response_start:], self._think_close_ids)

            if think_opens and think_closes:
                t_open_rel  = think_opens[0]
                t_close_rel = think_closes[0]
                think_start = response_start + t_open_rel + len(self._think_open_ids)
                think_end   = response_start + t_close_rel  # exclusive of </think>

                # Supervise thinking content
                for k in range(think_start, min(think_end, L)):
                    if attn_mask[k].item() == 1:
                        labels[k] = ids[k]

                # Optionally supervise the </think> token itself
                if self.supervise_stop_tokens:
                    for k in range(think_end, min(think_end + len(self._think_close_ids), L)):
                        if attn_mask[k].item() == 1:
                            labels[k] = ids[k]

                # Answer starts after </think>\n\n
                answer_start = response_start + t_close_rel + len(self._think_close_ids)
                # Skip any whitespace/newline tokens after </think>
                while answer_start < L and ids[answer_start] in self._whitespace_ids():
                    answer_start += 1
            else:
                # No thinking block found even though use_thinking=True; supervise all response
                answer_start = response_start
        else:
            answer_start = response_start

        # Supervise answer tokens up to (but not including) <end_of_turn>
        eot_positions = _find_subsequence(ids[answer_start:], self._eot_ids)
        if eot_positions:
            answer_end = answer_start + eot_positions[0]
            if self.supervise_stop_tokens:
                answer_end += len(self._eot_ids)
        else:
            answer_end = L

        for k in range(answer_start, min(answer_end, L)):
            if attn_mask[k].item() == 1:
                labels[k] = ids[k]

        return labels

    def _whitespace_ids(self) -> set:
        # IDs that decode to pure whitespace - skip them after </think>
        vocab = self.tokenizer.get_vocab()
        ws_ids = set()
        for tok_str, tok_id in vocab.items():
            decoded = self.tokenizer.convert_tokens_to_string([tok_str])
            if decoded.strip() == "":
                ws_ids.add(tok_id)
        return ws_ids

    # ------------------------------------------------------------------
    # Padding and batching
    # ------------------------------------------------------------------

    def _pad_and_batch(self, encoded: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        max_len = min(
            max(e["input_ids"].shape[0] for e in encoded),
            self.max_length,
        )

        input_ids_batch  = []
        attn_mask_batch  = []
        labels_batch     = []

        for e in encoded:
            L = e["input_ids"].shape[0]
            pad = max_len - L
            input_ids_batch.append(
                torch.cat([e["input_ids"],  torch.full((pad,), pad_id,               dtype=torch.long)])
            )
            attn_mask_batch.append(
                torch.cat([e["attention_mask"], torch.zeros(pad,                     dtype=torch.long)])
            )
            labels_batch.append(
                torch.cat([e["labels"],     torch.full((pad,), self.ignore_index,    dtype=torch.long)])
            )

        return {
            "input_ids":      torch.stack(input_ids_batch),
            "attention_mask": torch.stack(attn_mask_batch),
            "labels":         torch.stack(labels_batch),
        }

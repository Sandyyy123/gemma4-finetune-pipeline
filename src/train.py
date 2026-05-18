"""
Gemma4 finetuning pipeline with thinking-channel loss masking.

Supports:
  - Standard Gemma4 (bfloat16)
  - NVFP4-quantized Gemma4 (auto-detected via model config)
  - QLoRA / LoRA adapters
  - USETHINKING flag: supervise thinking traces or answer-only
  - Debug eval hook via DebugEvalCallback

Usage:
    python src/train.py --config config.yaml

    # Answer-only training (default)
    python src/train.py --config config.yaml --use_thinking false

    # Train on thinking steps too
    python src/train.py --config config.yaml --use_thinking true
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from data_collator import Gemma4ThinkingCollator
from debug_eval import DebugEvalCallback


# ------------------------------------------------------------------ #
#  Argument parsing                                                     #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Gemma4 finetuning with thinking-channel loss masking")
    p.add_argument("--config",        type=str, default="config.yaml")
    p.add_argument("--use_thinking",  type=str, default=None,
                   help="Override USETHINKING from config (true/false)")
    p.add_argument("--train_file",    type=str, default=None)
    p.add_argument("--output_dir",    type=str, default=None)
    p.add_argument("--model_name",    type=str, default=None)
    p.add_argument("--debug_eval",    action="store_true",
                   help="Enable DebugEvalCallback (prints per-segment loss at each eval step)")
    return p.parse_args()


# ------------------------------------------------------------------ #
#  Config loading                                                       #
# ------------------------------------------------------------------ #

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(cfg: dict, args) -> dict:
    """Merge CLI overrides into config dict."""
    if args.use_thinking is not None:
        cfg["training"]["use_thinking"] = args.use_thinking.lower() == "true"
    if args.train_file:
        cfg["data"]["train_file"] = args.train_file
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.model_name:
        cfg["model"]["name"] = args.model_name
    return cfg


# ------------------------------------------------------------------ #
#  Dataset loading                                                      #
# ------------------------------------------------------------------ #

def load_dataset(train_file: str) -> Dataset:
    records = []
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[Data] Loaded {len(records)} examples from {train_file}")
    return Dataset.from_list(records)


# ------------------------------------------------------------------ #
#  Model loading — auto-detect standard vs NVFP4                       #
# ------------------------------------------------------------------ #

def _is_nvfp4(model_name: str) -> bool:
    """
    Heuristic: NVFP4-patched variants typically have 'nvfp4', 'nf4', or
    'quantized' in the model name or repo path.
    """
    name_lower = model_name.lower()
    return any(tag in name_lower for tag in ["nvfp4", "nf4", "quantized", "bnb"])


def load_model_and_tokenizer(cfg: dict):
    model_name   = cfg["model"]["name"]
    lora_cfg     = cfg.get("lora", {})
    quant_cfg    = cfg.get("quantization", {})
    is_nvfp4     = _is_nvfp4(model_name)

    print(f"[Model] Loading: {model_name}")
    print(f"[Model] Variant: {'NVFP4' if is_nvfp4 else 'standard bfloat16'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    if is_nvfp4 or quant_cfg.get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit               = True,
            bnb_4bit_use_double_quant  = quant_cfg.get("double_quant", True),
            bnb_4bit_quant_type        = "nf4",
            bnb_4bit_compute_dtype     = torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            device_map          = "auto",
            trust_remote_code   = True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype    = torch.bfloat16,
            device_map     = "auto",
            trust_remote_code = True,
        )

    # Apply LoRA
    if lora_cfg.get("enabled", True):
        lora_config = LoraConfig(
            task_type    = TaskType.CAUSAL_LM,
            r            = lora_cfg.get("r", 16),
            lora_alpha   = lora_cfg.get("alpha", 32),
            lora_dropout = lora_cfg.get("dropout", 0.05),
            target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            bias         = "none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


# ------------------------------------------------------------------ #
#  Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    cfg    = resolve(cfg, args)

    train_cfg  = cfg["training"]
    data_cfg   = cfg["data"]

    use_thinking = bool(train_cfg.get("use_thinking", False))
    print(f"[Config] use_thinking = {use_thinking}")

    # Load data
    train_dataset = load_dataset(data_cfg["train_file"])
    eval_dataset  = None
    if data_cfg.get("eval_file"):
        eval_dataset = load_dataset(data_cfg["eval_file"])

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Build collator
    collator = Gemma4ThinkingCollator(
        tokenizer             = tokenizer,
        max_length            = train_cfg.get("max_length", 2048),
        use_thinking          = use_thinking,
        supervise_stop_tokens = train_cfg.get("supervise_stop_tokens", False),
    )

    # SFT config
    sft_config = SFTConfig(
        output_dir                  = train_cfg.get("output_dir", "./output"),
        num_train_epochs            = train_cfg.get("epochs", 3),
        per_device_train_batch_size = train_cfg.get("batch_size", 2),
        gradient_accumulation_steps = train_cfg.get("grad_accum", 4),
        learning_rate               = float(train_cfg.get("lr", 2e-4)),
        warmup_ratio                = float(train_cfg.get("warmup_ratio", 0.05)),
        lr_scheduler_type           = train_cfg.get("lr_scheduler", "cosine"),
        logging_steps               = train_cfg.get("logging_steps", 10),
        save_steps                  = train_cfg.get("save_steps", 100),
        eval_steps                  = train_cfg.get("eval_steps", 50) if eval_dataset else None,
        evaluation_strategy         = "steps" if eval_dataset else "no",
        fp16                        = False,
        bf16                        = True,
        dataloader_num_workers      = 0,
        report_to                   = train_cfg.get("report_to", "none"),
        max_seq_length              = train_cfg.get("max_length", 2048),
        dataset_text_field          = None,   # we use a custom collator
        remove_unused_columns       = False,
    )

    # Build trainer
    trainer = SFTTrainer(
        model           = model,
        args            = sft_config,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        data_collator   = collator,
        tokenizer       = tokenizer,
    )

    # Debug eval callback
    if args.debug_eval and eval_dataset is not None:
        trainer.add_callback(DebugEvalCallback(tokenizer))
        print("[Config] DebugEvalCallback attached")

    print("[Train] Starting training...")
    trainer.train()
    trainer.save_model(train_cfg.get("output_dir", "./output"))
    print("[Train] Done. Model saved.")


if __name__ == "__main__":
    main()

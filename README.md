> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# Gemma4 Finetuning Pipeline

Supervised finetuning for Gemma4 with optional thinking-channel loss masking.
Supports standard bfloat16 and NVFP4-quantized model variants via a single flag.

## Features

- Per-token loss masking: prompt always masked, thinking optional, answer always supervised
- `USETHINKING` flag - toggle thinking-step supervision without changing any other code
- Auto-detects standard vs NVFP4-patched Gemma4 based on model name
- QLoRA via bitsandbytes + PEFT (4-bit NF4 by default)
- `DebugEvalCallback` - prints full decoded prompt, label sequence, and per-segment loss at each eval step so you can verify gradient flow from step 1
- Cloud deployment scripts for RunPod, Vast.ai, and UpCloud

## Data Format

Each line of the `.jsonl` training file must have these fields:

```json
{
  "question": "User question text",
  "meta":     "FAKTEN:\n- ...\nHANDLUNGSANWEISUNGEN:\n- ...",
  "thinking": "Step-by-step reasoning trace",
  "answer":   "Final response to the user"
}
```

The collator wraps these into the correct Gemma4 chat template:

```
<bos><start_of_turn>user
{question}

{meta}<end_of_turn>
<start_of_turn>model
<think>
{thinking}
</think>

{answer}<end_of_turn>
```

When `use_thinking: false`, the `<think>...</think>` block is excluded entirely.

## Loss Masking Logic

| Segment | `use_thinking: false` | `use_thinking: true` |
|---|---|---|
| User prompt (question + meta) | masked | masked |
| `<think>...</think>` block | excluded | **supervised** |
| Answer | **supervised** | **supervised** |
| Scaffold tokens (`<start_of_turn>` etc.) | masked | masked |

Set `supervise_stop_tokens: true` in config to also supervise `<think>` / `</think>` boundary tokens.

## Quick Start

```bash
pip install -r requirements.txt

# Answer-only training
python src/train.py --config config.yaml

# Train on thinking steps too
python src/train.py --config config.yaml --use_thinking true

# With debug eval output (requires eval_file in config)
python src/train.py --config config.yaml --debug_eval
```

## Debug Eval Output

With `--debug_eval`, each eval step prints:

```
[PROMPT decoded]:
<bos><start_of_turn>user
Mein Account wurde gesperrt...

[TARGET decoded]:
<think>
1. Erkenne die Besorgnis...
</think>

Sehr geehrte Damen und Herren...

[LABEL sequence] (pos: token | label):
  [ 47] 'Sehr'     -> 'Sehr'
  [ 48] ' geehrte' -> ' geehrte'
  ...

[PER-SEGMENT LOSS]
  thinking : 1.2341  (n=23)
  answer   : 0.8821  (n=61)
  total    : 0.9712  (n=84)
```

## Model Variants

| Variant | Config setting |
|---|---|
| Standard bfloat16 | `model.name: google/gemma-4-9b-it` + `quantization.load_in_4bit: false` |
| 4-bit NF4 (QLoRA) | `quantization.load_in_4bit: true` (default) |
| NVFP4-patched | Set `model.name` to your NVFP4 variant path; pipeline auto-detects |

## Cloud Deployment

```bash
# RunPod
# Use deploy/runpod.yaml as pod template

# Vast.ai
bash deploy/vastai.sh

# UpCloud ($250 credit: signup.upcloud.com/?promo=affiliate-converter250)
bash deploy/upcloud.sh
```

## File Structure

```
src/
  data_collator.py  - Gemma4ThinkingCollator (per-token loss masking)
  train.py          - Main training script (SFTTrainer + QLoRA)
  debug_eval.py     - DebugEvalCallback (per-segment loss diagnostics)
data/
  dummy_train.jsonl - 5 sample examples (drop in your train.jsonl)
deploy/
  runpod.yaml       - RunPod pod template
  vastai.sh         - Vast.ai setup script
  upcloud.sh        - UpCloud setup script
config.yaml         - All training hyperparameters
requirements.txt
```

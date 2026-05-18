#!/usr/bin/env bash
# Vast.ai launch script for Gemma4 finetuning
# Recommended instance: ~80GB VRAM (A100/H100), CUDA 12.1+

set -euo pipefail

REPO_URL="https://github.com/Sandyyy123/gemma4-finetune-pipeline"
HF_TOKEN="${HF_TOKEN:-}"  # set in environment or .env

echo "[setup] Installing dependencies..."
pip install --quiet -r requirements.txt

echo "[setup] Logging into HuggingFace..."
if [ -n "$HF_TOKEN" ]; then
  huggingface-cli login --token "$HF_TOKEN"
fi

echo "[train] Starting training (answer-only)..."
python src/train.py \
  --config config.yaml \
  --debug_eval

echo "[done] Training complete. Adapter weights in ./output/"

#!/usr/bin/env bash
# UpCloud GPU instance setup for Gemma4 finetuning
# Promo: signup.upcloud.com/?promo=affiliate-converter250 ($250 credit)
# Recommended: GPU-optimized instance, 80GB+ VRAM, Ubuntu 22.04

set -euo pipefail

HF_TOKEN="${HF_TOKEN:-}"

echo "[upcloud] Installing CUDA + PyTorch dependencies..."
apt-get update -qq && apt-get install -y -qq python3-pip git unzip

echo "[upcloud] Cloning repo..."
git clone https://github.com/Sandyyy123/gemma4-finetune-pipeline /workspace/gemma4
cd /workspace/gemma4

echo "[upcloud] Installing Python deps..."
pip install -r requirements.txt

if [ -n "$HF_TOKEN" ]; then
  huggingface-cli login --token "$HF_TOKEN"
fi

echo "[upcloud] Running training (thinking mode)..."
python src/train.py \
  --config config.yaml \
  --use_thinking true \
  --debug_eval

echo "[upcloud] Done."

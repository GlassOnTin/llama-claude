#!/bin/bash
# ==============================================================================
# Download Qwen3.8-27B Vision Model + MTP Speculative Draft Head (Verified Weights)
# ==============================================================================

set -e
TARGET_DIR="${1:-$HOME/models}"
mkdir -p "$TARGET_DIR"

REPO="ggml-org/Qwen3.8-27B-GGUF"
MODEL_FILE="Qwen3.8-27B-Q8_0.gguf"
MMPROJ_FILE="mmproj-Qwen3.8-27B-BF16.gguf"
MTP_FILE="mtp-Qwen3.8-27B-Q8_0.gguf"

echo "=== Downloading Qwen3.8-27B (256k Vision + MTP) to $TARGET_DIR ==="

echo "1. Downloading Vision Projector ($MMPROJ_FILE - ~870MB)..."
hf download "$REPO" "$MMPROJ_FILE" --local-dir "$TARGET_DIR"

echo "2. Downloading MTP Draft Model ($MTP_FILE - ~2.95GB for 2x Speculative Decoding)..."
hf download "$REPO" "$MTP_FILE" --local-dir "$TARGET_DIR"

echo "3. Downloading Base Model ($MODEL_FILE - ~26.6GB Q8_0)..."
hf download "$REPO" "$MODEL_FILE" --local-dir "$TARGET_DIR"

echo "=== All files downloaded successfully! ==="
ls -lh "$TARGET_DIR/$MODEL_FILE" "$TARGET_DIR/$MMPROJ_FILE" "$TARGET_DIR/$MTP_FILE"

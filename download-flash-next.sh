#!/bin/bash
# ==============================================================================
# Download Qwen3.8-Flash-Next (Q6_K Near-Lossless 96GB Setup + Multimodal mmproj)
# ==============================================================================

set -e
TARGET_DIR="${1:-$HOME/models/Qwen3.8-Flash-Next}"
mkdir -p "$TARGET_DIR"

REPO="DevQuasar/Qwen.Qwen3.8-Flash-Next-GGUF"

echo "================================================================="
echo " Downloading Qwen3.8-Flash-Next (Q6_K ~75GB + mmproj Vision)"
echo " Target Directory: $TARGET_DIR"
echo "================================================================="

hf download "$REPO" \
    --include "Q6_K/*" \
    --include "mmproj-Qwen.Qwen3.8-Flash-Next.f16.gguf" \
    --local-dir "$TARGET_DIR"

echo "=== Download complete! ==="
ls -lh "$TARGET_DIR"/Q6_K/ "$TARGET_DIR"/mmproj* 2>/dev/null || true

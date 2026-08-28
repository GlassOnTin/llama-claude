#!/bin/bash
# ==============================================================================
# Qwen3.8-27B Dual RTX 5090 (64GB VRAM) Server Script
# 256k Full Context + Vision + MTP Speculative Decoding + Asymmetric Layer Split
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/models/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build-cuda/bin/llama-server}"

# Detect available NVIDIA GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 2)
if [ "$NUM_GPUS" -ge 3 ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2"
    DEFAULT_SPLIT_27B="20,22,22"
    DEFAULT_SPLIT_FLASH="12,18,18"
    GPU_LABEL="Triple RTX 5090 (96GB Total VRAM)"
else
    export CUDA_VISIBLE_DEVICES="0,1"
    DEFAULT_SPLIT_27B="30,34"
    DEFAULT_SPLIT_FLASH="20,28"
    GPU_LABEL="Dual RTX 5090 (64GB Total VRAM)"
fi

# Detect Model Stack: Flash-Next (Base vs Embedded MTP) vs Qwen3.8-27B
ENABLE_MTP="${ENABLE_MTP:-0}"

if [ "$ENABLE_MTP" = "1" ] && [ -f "$MODEL_DIR/Qwen3.8-Flash-Next/UD-Q4_K_XL-MTP/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00005.gguf" ]; then
    MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/UD-Q4_K_XL-MTP/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00005.gguf}"
    MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/mmproj-BF16.gguf}"
    MTP_MODE="embedded"
    TENSOR_SPLIT="${TENSOR_SPLIT:-15,17,17}" # 49 layers total (15 on GPU0, 17 on GPU1, 17 on GPU2)
    MODEL_NAME="Qwen3.8-Flash-Next (125B MoE + Embedded MTP Draft Head)"
elif [ -f "$MODEL_DIR/Qwen3.8-Flash-Next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" ]; then
    MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
    MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/mmproj-BF16.gguf}"
    MTP_MODE="none"
    TENSOR_SPLIT="${TENSOR_SPLIT:-14,17,17}" # 48 layers total (14 on GPU0, 17 on GPU1, 17 on GPU2)
    MODEL_NAME="Qwen3.8-Flash-Next (125B MoE / 512 Experts / QSA)"
elif [ -f "$MODEL_DIR/Qwen3.8-Flash-Next/UD-Q4_K_XL-MTP/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00005.gguf" ]; then
    MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/UD-Q4_K_XL-MTP/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00005.gguf}"
    MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/mmproj-BF16.gguf}"
    MTP_MODE="none"
    TENSOR_SPLIT="${TENSOR_SPLIT:-16,16,17}"
    MODEL_NAME="Qwen3.8-Flash-Next (125B MoE / 512 Experts / QSA - MTP Disabled)"
else
    MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.8-27B-Q8_0.gguf}"
    MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/mmproj-Qwen3.8-27B-BF16.gguf}"
    MTP_FILE="${MTP_FILE:-$MODEL_DIR/mtp-Qwen3.8-27B-Q8_0.gguf}"
    MTP_MODE="standalone"
    TENSOR_SPLIT="${TENSOR_SPLIT:-$DEFAULT_SPLIT_27B}"
    MODEL_NAME="Qwen3.8-27B (Dense Q8_0)"
fi

TEMPLATE_FILE="${TEMPLATE_FILE:-$SCRIPT_DIR/templates/qwen3.8-claude.jinja}"

# Server Configuration
PORT="${PORT:-8090}"
HOST="${HOST:-0.0.0.0}"
CTX_SIZE="${CTX_SIZE:-220000}" # 220k context (High-precision long-horizon reasoning)
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"

echo "========================================================"
echo " Starting $MODEL_NAME on $GPU_LABEL"
echo "========================================================"
echo " - Base Model:    $MODEL_FILE"
echo " - Vision mmproj: $MMPROJ_FILE"
echo " - Template:      $TEMPLATE_FILE"
echo " - Context Size:  $CTX_SIZE tokens"
echo " - KV Cache:      Q8_0 (Keys) / Q5_0 (Values) — High-Precision Asymmetric Cache"
echo " - Layer Split:   Pipelined ($TENSOR_SPLIT layers)"
echo " - Endpoint:      http://$HOST:$PORT"
echo "========================================================"

EXTRA_ARGS=()

# Multi-Token Prediction (MTP) Speculative Decoding
if [ "$MTP_MODE" = "embedded" ]; then
    echo " [+] Embedded MTP Draft Head detected: Enabling speculative decoding (3 draft tokens)"
    EXTRA_ARGS+=(
        --spec-type draft-mtp
        --spec-draft-n-max 3
        --spec-draft-p-min 0.75
    )
elif [ "$MTP_MODE" = "standalone" ] && [ -n "${MTP_FILE:-}" ] && [ -f "$MTP_FILE" ]; then
    echo " [+] MTP Draft Model detected: Enabling speculative decoding (3 draft tokens)"
    EXTRA_ARGS+=(
        --spec-draft-model "$MTP_FILE"
        --spec-type draft-mtp
        --spec-draft-n-max 3
        --spec-draft-ngl 99
    )
else
    echo " [-] Running standard full-precision QSA decoding (MTP draft disabled for maximum throughput)."
fi

# Vision Projector
if [ -f "$MMPROJ_FILE" ]; then
    echo " [+] Vision Projector detected: Enabling multimodal support"
    EXTRA_ARGS+=(--mmproj "$MMPROJ_FILE")
fi

# Custom Chat Template
if [ -f "$TEMPLATE_FILE" ]; then
    echo " [+] Custom Chat Template loaded: $TEMPLATE_FILE"
    EXTRA_ARGS+=(--chat-template-file "$TEMPLATE_FILE")
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found."
    echo "Please run: $SCRIPT_DIR/download.sh"
    exit 1
fi

exec "$LLAMA_SERVER" \
    --model "$MODEL_FILE" \
    --host "$HOST" \
    --port "$PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --split-mode layer \
    --tensor-split "$TENSOR_SPLIT" \
    --ctx-size "$CTX_SIZE" \
    --parallel 1 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q5_0 \
    --alias "qwen3.8-flash-next,qwen3.8-27b,claude-3-5-sonnet-20241022,claude-3-5-haiku-20241022,claude-3-opus-20240229,claude-sonnet-4-20250514,default" \
    --reasoning on \
    --reasoning-format deepseek \
    --reasoning-budget 8192 \
    --context-shift \
    --metrics \
    "${EXTRA_ARGS[@]}" \
    "$@"

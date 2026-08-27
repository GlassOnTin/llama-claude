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
    DEFAULT_SPLIT_FLASH="16,16,16"
    GPU_LABEL="Triple RTX 5090 (96GB Total VRAM)"
else
    export CUDA_VISIBLE_DEVICES="0,1"
    DEFAULT_SPLIT_27B="30,34"
    DEFAULT_SPLIT_FLASH="24,24"
    GPU_LABEL="Dual RTX 5090 (64GB Total VRAM)"
fi

# Detect Model Stack: Flash-Next vs Qwen3.8-27B
if [ -f "$MODEL_DIR/Qwen3.8-Flash-Next/Q6_K/Qwen.Qwen3.8-Flash-Next.f16.gguf.Q6_K.gguf-00001-of-00011.gguf" ]; then
    MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/Q6_K/Qwen.Qwen3.8-Flash-Next.f16.gguf.Q6_K.gguf-00001-of-00011.gguf}"
    MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/mmproj-Qwen.Qwen3.8-Flash-Next.f16.gguf}"
    MTP_FILE="${MTP_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
    TENSOR_SPLIT="${TENSOR_SPLIT:-$DEFAULT_SPLIT_FLASH}"
    MODEL_NAME="Qwen3.8-Flash-Next (125B MoE Q6_K)"
else
    MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.8-27B-Q8_0.gguf}"
    MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/mmproj-Qwen3.8-27B-BF16.gguf}"
    MTP_FILE="${MTP_FILE:-$MODEL_DIR/mtp-Qwen3.8-27B-Q8_0.gguf}"
    TENSOR_SPLIT="${TENSOR_SPLIT:-$DEFAULT_SPLIT_27B}"
    MODEL_NAME="Qwen3.8-27B (Dense Q8_0)"
fi

TEMPLATE_FILE="${TEMPLATE_FILE:-$SCRIPT_DIR/templates/qwen3.8-claude.jinja}"

# Server Configuration
PORT="${PORT:-8090}"
HOST="${HOST:-0.0.0.0}"
CTX_SIZE="${CTX_SIZE:-262144}" # Full 256k context
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"

echo "========================================================"
echo " Starting $MODEL_NAME on $GPU_LABEL"
echo "========================================================"
echo " - Base Model:    $MODEL_FILE"
echo " - Vision mmproj: $MMPROJ_FILE"
echo " - Template:      $TEMPLATE_FILE"
echo " - Context Size:  $CTX_SIZE tokens (256k full)"
echo " - KV Cache:      Q8_0 (~4.7GB total across 256k ctx)"
echo " - Layer Split:   Pipelined ($TENSOR_SPLIT layers, TB4 optimized)"
echo " - Endpoint:      http://$HOST:$PORT"
echo "========================================================"

EXTRA_ARGS=()

# Multi-Token Prediction (MTP) Speculative Decoding (1.6x - 2.4x speedup)
if [ -f "$MTP_FILE" ]; then
    echo " [+] MTP Draft Model detected: Enabling speculative decoding (3 draft tokens)"
    EXTRA_ARGS+=(
        --spec-draft-model "$MTP_FILE"
        --spec-type draft-mtp
        --spec-draft-n-max 3
        --spec-draft-ngl 99
    )
else
    echo " [-] MTP Draft Model ($MTP_FILE) not found; running standard single-token decoding."
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
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --alias "qwen3.8-27b,claude-3-5-sonnet-20241022,claude-3-5-haiku-20241022,claude-3-opus-20240229,claude-sonnet-4-20250514,default" \
    --reasoning on \
    --reasoning-format deepseek \
    --metrics \
    "${EXTRA_ARGS[@]}" \
    "$@"

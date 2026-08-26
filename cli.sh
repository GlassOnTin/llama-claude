#!/bin/bash
# Interactive CLI chat with Qwen3.8-27B on Dual RTX 5090 (256k Context + MTP)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/models/llama.cpp}"
LLAMA_CLI="${LLAMA_CLI:-$LLAMA_DIR/build-cuda/bin/llama-cli}"

MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.8-27B-Q8_0.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/mmproj-Qwen3.8-27B-BF16.gguf}"
MTP_FILE="${MTP_FILE:-$MODEL_DIR/mtp-Qwen3.8-27B-Q8_0.gguf}"
TEMPLATE_FILE="${TEMPLATE_FILE:-$SCRIPT_DIR/templates/qwen3.8-claude.jinja}"

CTX_SIZE="${CTX_SIZE:-262144}"
TENSOR_SPLIT="${TENSOR_SPLIT:-30,34}"

export CUDA_VISIBLE_DEVICES="0,1"

EXTRA_ARGS=()
if [ -f "$MTP_FILE" ]; then
    EXTRA_ARGS+=(
        --spec-draft-model "$MTP_FILE"
        --spec-type draft-mtp
        --spec-draft-n-max 3
        --spec-draft-ngl 99
    )
fi

if [ -f "$MMPROJ_FILE" ]; then
    EXTRA_ARGS+=(--mmproj "$MMPROJ_FILE")
fi

if [ -f "$TEMPLATE_FILE" ]; then
    EXTRA_ARGS+=(--chat-template-file "$TEMPLATE_FILE")
fi

exec "$LLAMA_CLI" \
    --model "$MODEL_FILE" \
    --n-gpu-layers 99 \
    --split-mode layer \
    --tensor-split "$TENSOR_SPLIT" \
    --ctx-size "$CTX_SIZE" \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --reasoning on \
    --reasoning-format deepseek \
    --conversation \
    "${EXTRA_ARGS[@]}" \
    "$@"

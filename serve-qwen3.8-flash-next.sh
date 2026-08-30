#!/bin/bash
# ==============================================================================
# Qwen3.8-Flash-Next 125B MoE — Triple RTX 5090 (96GB VRAM) + RAM expert offload
# ==============================================================================
# The model is ~101GB of weights (59.6GB dense + 41.2GB MoE experts) and does
# NOT fit in 96GB VRAM. The fix is not more quantisation: --n-cpu-moe spills
# the first N layers' expert weights into system RAM (62GB free). Only 10 of
# 512 experts fire per token, so the per-token RAM traffic is small and the
# GPU still does the compute.
#
# Budget (3x 32GB, ~93GB usable, GPU0 carries the desktop):
#   dense weights + KV + mmproj  ~65GB  -> must be on GPU
#   remaining VRAM               ~28GB  -> holds ~32 layers of experts
#   --n-cpu-moe 16               ~14GB  -> 16 layers of experts in RAM
#
# Tuning N_CPU_MOE:
#   load OOMs on GPU            -> raise it (18, 20)
#   loads with VRAM to spare    -> lower it (12, 14) for speed: the 10 active
#                                  experts per token hit VRAM instead of RAM
#
# MTP is OFF by default so the base benchmark is clean. Set MTP=1 to add the
# Q8_0 draft head (+4.1GB, and the draft has its own experts — it tightens
# the budget; raise N_CPU_MOE by ~2 if you enable it).
# ==============================================================================

set -e

MODEL_DIR="${MODEL_DIR:-/home/ian/models}"
LLAMA_DIR="${LLAMA_DIR:-/home/ian/models/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build-cuda/bin/llama-server}"

FLASH_DIR="$MODEL_DIR/Qwen3.8-Flash-Next"
MODEL_FILE="${MODEL_FILE:-$FLASH_DIR/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-$FLASH_DIR/mmproj-BF16.gguf}"
MTP_FILE="${MTP_FILE:-$FLASH_DIR/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
TEMPLATE_FILE="${TEMPLATE_FILE:-/home/ian/Code/llama-claude/templates/qwen3.8-claude.jinja}"

PORT="${PORT:-8090}"
HOST="${HOST:-0.0.0.0}"
CTX_SIZE="${CTX_SIZE:-262144}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-8}"
MTP="${MTP:-0}"

# Balanced split: GPU0 carries 18 layers (8 in RAM + 10 with experts in VRAM), GPUs 1 & 2 carry 15 layers each
TENSOR_SPLIT="${TENSOR_SPLIT:-18,15,15}"

export CUDA_VISIBLE_DEVICES="0,1,2"

echo "========================================================"
echo " Starting Qwen3.8-Flash-Next (125B MoE, 512 experts, 10 active)"
echo "========================================================"
echo " - Base Model:    $MODEL_FILE"
echo " - Vision mmproj: $MMPROJ_FILE"
echo " - Context Size:  $CTX_SIZE tokens (hybrid SSM: KV grows on 12/48 layers only)"
echo " - KV Cache:      Q8_0 (~3-4GB at 256k)"
echo " - Layer Split:   $TENSOR_SPLIT (even; experts offloaded separately)"
echo " - Expert Offload: first $N_CPU_MOE layers' experts in RAM (~$((N_CPU_MOE * 860 / 1000))GB)"
echo " - MTP:           $([ "$MTP" = "1" ] && echo on || echo off)"
echo " - Endpoint:      http://$HOST:$PORT"
echo "========================================================"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: model file not found: $MODEL_FILE"
    exit 1
fi

EXTRA_ARGS=()

if [ "$MTP" = "1" ] && [ -f "$MTP_FILE" ]; then
    echo " [+] MTP draft model detected: enabling speculative decoding (3 draft tokens)"
    EXTRA_ARGS+=(
        --spec-draft-model "$MTP_FILE"
        --spec-type draft-mtp
        --spec-draft-n-max 3
        --spec-draft-ngl 99
    )
else
    echo " [-] MTP disabled (set MTP=1 to enable)"
fi

exec "$LLAMA_SERVER" \
    --model "$MODEL_FILE" \
    --host "$HOST" \
    --port "$PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --split-mode layer \
    --tensor-split "$TENSOR_SPLIT" \
    --n-cpu-moe "$N_CPU_MOE" \
    --ctx-size "$CTX_SIZE" \
    --parallel 1 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --mmproj "$MMPROJ_FILE" \
    --alias "qwen3.8-flash-next,qwen3.8-27b,claude-3-5-sonnet-20241022,claude-3-5-haiku-20241022,claude-3-opus-20240229,claude-sonnet-4-20250514,default" \
    --chat-template-file "$TEMPLATE_FILE" \
    --reasoning on \
    --reasoning-format deepseek \
    --reasoning-budget 2048 \
    --temp 0.2 \
    --top-p 0.95 \
    --min-p 0.0 \
    --context-shift \
    --metrics \
    "${EXTRA_ARGS[@]}" \
    "$@"

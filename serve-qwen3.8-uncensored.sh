#!/bin/bash
# ==============================================================================
# Qwen3.8-Flash-Next-UNCENSORED — Triple RTX 5090 (96GB VRAM) + RAM expert offload
# ==============================================================================
# Serves the uncensored fine-tune (orcarouter/Qwen3.8-Flash-Next-Uncensored,
# converted to llama.cpp GGUF by mradermacher i1-GGUF, imatrix Q4_K_M).
# llama.cpp-native: qwen4exp arch, PLE table in-file, 1224 tensors — the
# Baekpica Mixed-Quant-SSD-PLE artifact is NOT llama.cpp-loadable (ds4-runtime
# naming: no qwen4exp.ple.layers KV, linear_attn.* / blk.N.ple.*.weight etc).
#
# Same serving recipe as serve-qwen3.8-flash-next.sh (identical geometry:
# 1224 tensors, 48 blocks, 512 experts / 10 active / 640-dim, ctx 262144,
# vocab 248320, embd 2560). Runs on port 8091.
#
# mradermacher ships no mmproj. MMPROJ_FILE reuses the production one from
# the same base model; llama-server validates it at load and errors loudly
# if the projector dims do not match. Set MMPROJ_FILE= to run text-only.
#
# mradermacher ships no MTP draft sidecar → MTP stays off.
# ==============================================================================

set -e

MODEL_DIR="${MODEL_DIR:-/home/ian/models}"
LLAMA_DIR="${LLAMA_DIR:-/home/ian/models/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build-cuda/bin/llama-server}"

UNCEN_DIR="$MODEL_DIR/Qwen3.8-Flash-Next-Uncensored"
MODEL_FILE="${MODEL_FILE:-$UNCEN_DIR/i1-Q4_K_M/Qwen3.8-Flash-Next-Uncensored.i1-Q4_K_M.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/mmproj-BF16.gguf}"
MTP_FILE="${MTP_FILE:-}"
TEMPLATE_FILE="${TEMPLATE_FILE:-/home/ian/Code/llama-claude/templates/qwen3.8-claude.jinja}"

PORT="${PORT:-8091}"
HOST="${HOST:-0.0.0.0}"
# Total KV budget across all slots; each --parallel slot gets CTX_SIZE/parallel.
# parallel=1 gives the single slot the full 640k — Claude Code's MCP-heavy
# requests overflowed a 131072 slot (5-way split).
CTX_SIZE="${CTX_SIZE:-655360}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-8}"
# ubatch 256, not 512: the 655k KV leaves ~2GB too little for the 4.3GB pp
# compute buffers on each GPU (alloc fails on device 1/2 with 8GB "free" per
# nvidia-smi). ubatch 256 halves the pp buffer to ~2.2GB and loads. Cost:
# prefill ~282 tok/s vs ~530 at 512 (generation speed unaffected).
UBATCH="${UBATCH:-256}"
MTP="${MTP:-0}"

# Balanced split: GPU0 carries 18 layers (8 in RAM + 10 with experts in VRAM), GPUs 1 & 2 carry 15 layers each
TENSOR_SPLIT="${TENSOR_SPLIT:-18,15,15}"

export CUDA_VISIBLE_DEVICES="0,1,2"

echo "========================================================"
echo " Starting Qwen3.8-Flash-Next-UNCENSORED (125B MoE, 512 experts, 10 active)"
echo "========================================================"
echo " - Base Model:    $MODEL_FILE"
echo " - Vision mmproj: $MMPROJ_FILE"
echo " - Context Size:  $CTX_SIZE tokens (hybrid SSM: KV grows on 12/48 layers only)"
echo " - KV Cache:      Q8_0 (~3-4GB at 256k)"
echo " - Layer Split:   $TENSOR_SPLIT (even; experts offloaded separately)"
echo " - Expert Offload: first $N_CPU_MOE layers' experts in RAM"
echo " - MTP:           $([ "$MTP" = "1" ] && echo on || echo off)"
echo " - Endpoint:      http://$HOST:$PORT"
echo "========================================================"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: model file not found: $MODEL_FILE"
    exit 1
fi

EXTRA_ARGS=()

if [ "$MTP" = "1" ] && [ -n "$MTP_FILE" ] && [ -f "$MTP_FILE" ]; then
    echo " [+] MTP draft model detected: enabling speculative decoding (3 draft tokens)"
    EXTRA_ARGS+=(
        --spec-draft-model "$MTP_FILE"
        --spec-type draft-mtp
        --spec-draft-n-max 3
        --spec-draft-ngl 99
    )
else
    echo " [-] MTP disabled"
fi

# GPU-resident LRU cache for the host-resident experts (upstream PR #27861,
# local commit 8603b4dc2). Production-tuned values from the qwen service.
MOE_CACHE="${MOE_CACHE:-128}"
MOE_CACHE_INSERTS="${MOE_CACHE_INSERTS:-4}"
if [ "$MOE_CACHE" != "0" ]; then
    echo " [+] MoE expert cache: $MOE_CACHE slots, $MOE_CACHE_INSERTS inserts/step"
    EXTRA_ARGS+=(
        --moe-expert-cache "$MOE_CACHE"
        --moe-expert-cache-inserts "$MOE_CACHE_INSERTS"
    )
else
    echo " [-] MoE expert cache disabled (MOE_CACHE=0)"
fi

# --load-mode none eager-loads the RAM-resident experts into anonymous memory
# instead of mmap'd file pages. Only viable with real RAM headroom (256GB kit).
LOAD_MODE="${LOAD_MODE:-none}"
if [ "$LOAD_MODE" != "mmap" ]; then
    echo " [+] Load mode: none (eager RAM-resident experts)"
    EXTRA_ARGS+=(--load-mode none)
else
    echo " [-] Load mode: mmap"
fi

MMGLUE=()
if [ -n "$MMPROJ_FILE" ] && [ -f "$MMPROJ_FILE" ]; then
    MMGLUE+=(--mmproj "$MMPROJ_FILE")
else
    echo " [-] No mmproj — text-only serving"
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
    --ubatch-size "$UBATCH" \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --alias "qwen3.8-flash-next-uncensored,qwen3.8-27b,claude-3-5-sonnet-20241022,claude-3-5-haiku-20241022,claude-3-opus-20240229,claude-sonnet-4-20250514,default" \
    --chat-template-file "$TEMPLATE_FILE" \
    --reasoning on \
    --reasoning-format deepseek \
    --reasoning-budget 2048 \
    --temp 0.2 \
    --top-p 0.95 \
    --min-p 0.0 \
    --context-shift \
    --metrics \
    "${MMGLUE[@]}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
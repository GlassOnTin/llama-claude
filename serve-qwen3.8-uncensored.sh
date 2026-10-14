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
# 48 blocks, 512 experts / 10 active / 640-dim, vocabulary/embd match),
# benchmarked on the base model 2026-10-12 — see that file for the full
# rationale and bench table. Runs on port 8091.
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
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build-cuda-master/bin/llama-server}"

UNCEN_DIR="$MODEL_DIR/Qwen3.8-Flash-Next-Uncensored"
MODEL_FILE="${MODEL_FILE:-$UNCEN_DIR/i1-Q4_K_M/Qwen3.8-Flash-Next-Uncensored.i1-Q4_K_M.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/Qwen3.8-Flash-Next/mmproj-BF16.gguf}"
MTP_FILE="${MTP_FILE:-}"
TEMPLATE_FILE="${TEMPLATE_FILE:-/home/ian/Code/llama-claude/templates/qwen3.8-claude.jinja}"

PORT="${PORT:-8091}"
HOST="${HOST:-0.0.0.0}"
# Train-time cap: 262144 (--ctx-size above is clamped by the server anyway).
CTX_SIZE="${CTX_SIZE:-262144}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
# In i1-Q4_K_M, blk.0..5 and blk.8,11,14,17 have q8_0 ffn_down_exps (1,750MiB),
# while blk.6,7,9,10,12,13,15,16 are uniform 1,450MiB (q5_0 ffn_down_exps + q4_K).
# Offloading those 8 uniform layers on GPU0 keeps all q8_0 layers in GPU0 VRAM
# and gives llama_moe_cache a single unified LRU layout group (1,988 slots @ 5632MiB).
N_CPU_MOE="${N_CPU_MOE:-0}"
MOE_OT_REGEX="${MOE_OT_REGEX:-blk\\.(6|7|9|10|12|13|15|16)\\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU}"
UBATCH="${UBATCH:-1024}"
MTP="${MTP:-0}"
KT="${KT:-f16}"
VT="${VT:-f16}"

# Desktop-aware split: GPU0 carries 18 layers, GPU1 carries 14 layers (+ desktop), GPU2 carries 16 layers
TENSOR_SPLIT="${TENSOR_SPLIT:-18,14,16}"

export CUDA_VISIBLE_DEVICES="0,1,2"

echo "========================================================"
echo " Starting Qwen3.8-Flash-Next-UNCENSORED (125B MoE, 512 experts, 10 active)"
echo "========================================================"
echo " - Base Model:    $MODEL_FILE"
echo " - Vision mmproj: $MMPROJ_FILE"
echo " - Context Size:  $CTX_SIZE tokens (hybrid SSM: KV grows on 12/48 layers only)"
echo " - KV Cache:      ${KT} / ${VT}"
echo " - Layer Split:   $TENSOR_SPLIT (GPU1 hosts desktop; GPU0 PCIe 5.0 x16)"
if [ "$N_CPU_MOE" != "0" ]; then
    echo " - Expert Offload: first $N_CPU_MOE layers' experts in RAM (--n-cpu-moe)"
else
    echo " - Expert Offload: -ot $MOE_OT_REGEX (8 uniform 1,450MiB layers in RAM)"
fi
echo " - MTP:           $([ "$MTP" = "1" ] && echo on || echo off)"
echo " - Endpoint:      http://$HOST:$PORT"
echo "========================================================"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: model file not found: $MODEL_FILE"
    exit 1
fi

EXTRA_ARGS=()

if [ "$N_CPU_MOE" != "0" ]; then
    EXTRA_ARGS+=(--n-cpu-moe "$N_CPU_MOE")
elif [ -n "$MOE_OT_REGEX" ]; then
    EXTRA_ARGS+=(-ot "$MOE_OT_REGEX")
fi

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

# Upstream MoE expert cache (PR #29887 + #30112): single unified LRU group on GPU0.
MOE_CACHE_MIB="${MOE_CACHE_MIB:-5632}"
if [ "$MOE_CACHE_MIB" != "0" ]; then
    echo " [+] MoE cache: --moe-cache-mib $MOE_CACHE_MIB (upstream; unified LRU on GPU0)"
    EXTRA_ARGS+=(--moe-cache-mib "$MOE_CACHE_MIB")
else
    echo " [-] MoE cache disabled (MOE_CACHE_MIB=0)"
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

# NOTE: --context-shift removed 2026-10-12: inert for qwen4exp
# (llama_memory_hybrid_idx does not implement KV shifting).

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
    --ctx-size "$CTX_SIZE" \
    --parallel 1 \
    --batch-size 2048 \
    --ubatch-size "$UBATCH" \
    --flash-attn on \
    --cache-type-k "$KT" \
    --cache-type-v "$VT" \
    --alias "qwen3.8-flash-next-uncensored" \
    --chat-template-file "$TEMPLATE_FILE" \
    --reasoning on \
    --reasoning-format deepseek \
    --reasoning-budget 2048 \
    --temp 0.2 \
    --top-p 0.95 \
    --min-p 0.0 \
    --metrics \
    "${MMGLUE[@]}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
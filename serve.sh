#!/bin/bash
# ==============================================================================
# Qwen3.8-Flash-Next 125B MoE — Triple RTX 5090 (96GB VRAM) + RAM expert offload
# ==============================================================================
# The model is ~103.7GiB of weights (48 layers) and does NOT fit in 96GB VRAM.
# Instead of --n-cpu-moe 8 (which spills blk.0..7 = 12,700MiB including the
# dynamic-quant outlier layers blk.2 [1,950MiB: q8_0+q5_K] and blk.4 [1,750MiB:
# q8_0+q4_K], fragmenting llama_moe_cache into 3 layout groups where blk.2/blk.4
# get only ~205 slots and bypass the cache on micro-batches >20 tokens), we use
# -ot to offload 8 uniform 1,500MiB layers on GPU0 (blk.5..12 = 12,000MiB:
# ffn_down_exps q5_1 + ffn_gate/up_exps q4_K).
#   - Keeps the heavy q8_0 outlier layers blk.2 & blk.4 permanently in GPU0 VRAM
#   - Consolidates GPU0's MoE cache into 1 unified LRU pool (5632MiB = 1,922 slots)
#   - Keeps all host-resident experts on GPU0 (PCIe 5.0 x16 @ 63 GB/s; GPUs 1 & 2
#     sit behind a chipset switch at PCIe 4.0 x4 @ 7.8 GB/s)
#   - Uses --tensor-split 18,14,16 because the desktop display lives on GPU1
#     (0000:19:00.0, ~1.3GiB), shifting 1 layer from GPU1 to headless GPU2.
#
# Config (2026-10-14, build-cuda-master f2cc7282c + 87a068c00 + 6388b92bf):
#   ctx     262144  (n_ctx_train cap; --ctx-size above this is ignored)
#   ubatch  1024    (saturated by 1280; see UBATCH block below)
#   KV      F16     (q8_0 flash-attn decode penalty measured: -6% @16k,
#                    -38% @240k depth vs F16. F16 costs ~+3.3GB total VRAM.)
#   offload -ot blk.5..12 (8 uniform 1,500MiB layers = 12,000MiB in pinned RAM)
#   cache   5632MiB (1 unified LRU group on GPU0; leaves ~1.2GiB free on GPU0,
#                    ~3.1GiB free on GPU1 with desktop, ~1.9GiB free on GPU2)
#
# Bench (code16k corpus a7705149, 5 runs):
#   2026-10-11 (fork cache 128/i4, ub256, q8 KV):           prefill  906, decode 77.5
#   2026-10-12 (ncmoe 8, mib5120, ub1024, f16, 18,15,15):   prefill 2233, decode 92.5
#   2026-10-14 (-ot blk.5..12, mib5632, ub1024, 18,14,16):  prefill 2379, decode 97.4
# ==============================================================================

set -e

MODEL_DIR="${MODEL_DIR:-/home/ian/models}"
LLAMA_DIR="${LLAMA_DIR:-/home/ian/models/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build-cuda-master/bin/llama-server}"

FLASH_DIR="$MODEL_DIR/Qwen3.8-Flash-Next"
MODEL_FILE="${MODEL_FILE:-$FLASH_DIR/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-$FLASH_DIR/mmproj-BF16.gguf}"
MTP_FILE="${MTP_FILE:-$FLASH_DIR/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
TEMPLATE_FILE="${TEMPLATE_FILE:-/home/ian/Code/llama-claude/templates/qwen3.8-claude.jinja}"

PORT="${PORT:-8090}"
HOST="${HOST:-0.0.0.0}"
# Train-time cap: 262144. The server clamps anything larger to this anyway.
CTX_SIZE="${CTX_SIZE:-262144}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
# Uniform 8-layer expert offload on GPU0 (blk.5..12: all 1,500MiB q5_1+q4_K).
# Set N_CPU_MOE > 0 to override with legacy --n-cpu-moe N instead.
N_CPU_MOE="${N_CPU_MOE:-0}"
MOE_OT_REGEX="${MOE_OT_REGEX:-blk\\.(5|6|7|8|9|10|11|12)\\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU}"
UBATCH="${UBATCH:-1024}"
MTP="${MTP:-0}"
KT="${KT:-f16}"
VT="${VT:-f16}"

# Desktop-aware split: GPU0 carries 18 layers (8 uniform in RAM + 10 in VRAM
# including heavy blk.2/blk.4), GPU1 carries 14 layers (+ ~1.3GiB desktop),
# GPU2 carries 16 layers.
TENSOR_SPLIT="${TENSOR_SPLIT:-18,14,16}"

export CUDA_VISIBLE_DEVICES="0,1,2"

echo "========================================================"
echo " Starting Qwen3.8-Flash-Next (125B MoE, 512 experts, 10 active)"
echo "========================================================"
echo " - Base Model:    $MODEL_FILE"
echo " - Vision mmproj: $MMPROJ_FILE"
echo " - Context Size:  $CTX_SIZE tokens (hybrid SSM: KV grows on 12/48 layers only)"
echo " - KV Cache:      ${KT} / ${VT} (decoded ~-6% @16k / -38% @240k vs F16)"
echo " - Layer Split:   $TENSOR_SPLIT (GPU1 hosts desktop; GPU0 PCIe 5.0 x16)"
if [ "$N_CPU_MOE" != "0" ]; then
    echo " - Expert Offload: first $N_CPU_MOE layers' experts in RAM (--n-cpu-moe)"
else
    echo " - Expert Offload: -ot $MOE_OT_REGEX (8 uniform 1,500MiB layers in RAM)"
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

if [ "$MTP" = "1" ] && [ -f "$MTP_FILE" ]; then
    echo " [+] MTP draft model detected: enabling speculative decoding (3 draft tokens)"
    EXTRA_ARGS+=(
        --spec-draft-model "$MTP_FILE"
        --spec-type draft-mtp
        --spec-draft-n-max 3
        --spec-draft-ngl 99
    )
else
    echo " [-] MTP disabled (set MTP=1 to enable; benched negative 2026-10-08/12)"
fi

# Upstream MoE expert cache (PR #29887 + #30112): GiB budget in MiB, single
# unified LRU bank on GPU0 (1,922 slots @ 5632MiB across blk.5..12).
MOE_CACHE_MIB="${MOE_CACHE_MIB:-5632}"
if [ "$MOE_CACHE_MIB" != "0" ]; then
    echo " [+] MoE cache: --moe-cache-mib $MOE_CACHE_MIB (upstream; unified LRU on GPU0)"
    EXTRA_ARGS+=(--moe-cache-mib "$MOE_CACHE_MIB")
else
    echo " [-] MoE cache disabled (MOE_CACHE_MIB=0)"
fi

# --load-mode none eager-loads the RAM-resident experts into anonymous memory
# instead of mmap'd file pages. Measured 2026-09-30 on 16k-code prompts,
# 5 runs: prefill 469 -> 820 tok/s (+75%), decode unchanged, cold run 0
# 472 vs 264-283 with mmap (first-prompt penalty mostly gone). LOAD_MODE=mmap
# reverts.
LOAD_MODE="${LOAD_MODE:-none}"
if [ "$LOAD_MODE" != "mmap" ]; then
    echo " [+] Load mode: none (eager RAM-resident experts)"
    EXTRA_ARGS+=(--load-mode none)
else
    echo " [-] Load mode: mmap"
fi

# NOTE: --context-shift removed 2026-10-12: it is inert for this model
# (llama_memory_hybrid_idx does not implement KV shifting; server warns
# "KV cache shifting is not supported for this context").

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
    --mmproj "$MMPROJ_FILE" \
    --alias "qwen3.8-flash-next,qwen3.8-125b,default" \
    --chat-template-file "$TEMPLATE_FILE" \
    --reasoning on \
    --reasoning-format deepseek \
    --reasoning-budget 2048 \
    --temp 0.2 \
    --top-p 0.95 \
    --min-p 0.0 \
    --metrics \
    "${EXTRA_ARGS[@]}" \
    "$@"
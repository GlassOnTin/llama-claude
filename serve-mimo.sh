#!/bin/bash
# ==============================================================================
# MiMo-V2.6-Flash (309B MoE / 15B active, arch mimo2) — 3x RTX 5090 + RAM spill
# ==============================================================================
# Native 4-bit (MXFP4) weights ~167GB (48 blocks; blk.0 dense, blk.1..47 MoE).
# Benched 2026-10-12 on 3x RTX 5090 (96GB VRAM) + 256GB DDR5:
#   - 22 of 47 MoE layers fit in VRAM via per-GPU -ot placement:
#       GPU0 (layers 0-17):  blk.1..6   in VRAM (6 MoE + blk.0 dense + 3.2GB vision)
#       GPU1 (layers 18-32): blk.18..25 in VRAM (8 MoE)
#       GPU2 (layers 33-47): blk.33..40 in VRAM (8 MoE + output head)
#     Remaining 25 MoE layers (blk.7..17, 26..32, 41..47) spill to pinned RAM.
#   - UBATCH=512 + F16 KV + --load-mode none:
#       prefill 390 -> 867 tok/s (+122%), decode 18.0 -> 21.8 tok/s (+21%).
#   - NOTE: --moe-cache-mib does NOT engage on MXFP4 expert weights (the cache
#     only supports Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K), so maximizing
#     direct VRAM-resident MoE layers via -ot is the primary lever.
# ==============================================================================

set -e

MODEL_DIR="${MODEL_DIR:-/home/ian/models/MiMo-V2.6-Flash}"
LLAMA_SERVER="${LLAMA_SERVER:-/home/ian/models/llama.cpp/build-cuda-master/bin/llama-server}"

MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/mmproj-MiMo-V2.6-Flash-RL-Q8_0.gguf}"
MTP_FILE="${MTP_FILE:-$MODEL_DIR/mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf}"
DFLASH_FILE="${DFLASH_FILE:-$MODEL_DIR/dflash-MiMo-V2.6-Flash-RL-Q8_0.gguf}"

PORT="${PORT:-8091}"
HOST="${HOST:-0.0.0.0}"
CTX_SIZE="${CTX_SIZE:-262144}"
UBATCH="${UBATCH:-512}"
KT="${KT:-f16}"
VT="${VT:-f16}"
# Balanced split over 3x5090 (layers 0..17 on GPU0, 18..32 on GPU1, 33..47 on GPU2)
TENSOR_SPLIT="${TENSOR_SPLIT:-18,15,15}"
# Spill 25 of 47 MoE layers to CPU (keeping 6 on GPU0, 8 on GPU1, 8 on GPU2 in VRAM)
MOE_OT_REGEX="${MOE_OT_REGEX:-blk\\.([7-9]|1[0-7]|2[6-9]|3[0-2]|4[1-7])\\.ffn_(up|down|gate)_exps\\.weight=CPU}"
LOAD_MODE="${LOAD_MODE:-none}"
SPEC_TYPE="${SPEC_TYPE:-none}"

export CUDA_VISIBLE_DEVICES="0,1,2"

echo "========================================================"
echo " MiMo-V2.6-Flash (309B MoE / 15B active, 256 experts, 8 active)"
echo "========================================================"
echo " - Base Model:     $MODEL_FILE"
echo " - Vision mmproj:  $MMPROJ_FILE"
echo " - Context Size:   $CTX_SIZE (39/48 layers SWA-128: KV is cheap, 9 global layers)"
echo " - KV Cache:       ${KT} / ${VT}"
echo " - Layer Split:    $TENSOR_SPLIT (6/8/8 MoE layers in VRAM, 25 in RAM)"
echo " - Expert Offload: -ot $MOE_OT_REGEX"
echo " - Spec decoding:  $SPEC_TYPE"
echo " - Endpoint:       http://$HOST:$PORT"
echo "========================================================"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: model file not found: $MODEL_FILE"
    exit 1
fi

EXTRA_ARGS=()

if [ "$LOAD_MODE" != "mmap" ]; then
    echo " [+] Load mode: none (eager RAM-resident experts)"
    EXTRA_ARGS+=(--load-mode none)
else
    echo " [-] Load mode: mmap"
fi

case "$SPEC_TYPE" in
    mtp)
        echo " [+] MTP draft sidecar enabled"
        EXTRA_ARGS+=(--spec-type draft-mtp --spec-draft-model "$MTP_FILE" --spec-draft-n-max 3)
        ;;
    dflash)
        echo " [+] DFlash drafter enabled"
        EXTRA_ARGS+=(--spec-type draft-dflash --spec-draft-model "$DFLASH_FILE")
        ;;
    *)
        echo " [-] Spec decoding off (SPEC_TYPE=mtp|dflash to enable)"
        ;;
esac

exec "$LLAMA_SERVER" \
    --model "$MODEL_FILE" \
    --host "$HOST" \
    --port "$PORT" \
    --n-gpu-layers 99 \
    --split-mode layer \
    --tensor-split "$TENSOR_SPLIT" \
    -ot "$MOE_OT_REGEX" \
    --ctx-size "$CTX_SIZE" \
    --parallel 1 \
    --batch-size 2048 \
    --ubatch-size "$UBATCH" \
    --flash-attn on \
    --cache-type-k "$KT" \
    --cache-type-v "$VT" \
    --mmproj "$MMPROJ_FILE" \
    --jinja \
    --alias "mimo-v2.6-flash,claude-3-5-sonnet-20241022,default" \
    --reasoning on \
    --reasoning-format deepseek \
    --reasoning-budget 2048 \
    --temp 0.2 \
    --top-p 0.95 \
    --min-p 0.0 \
    --metrics \
    "${EXTRA_ARGS[@]}" \
    "$@"
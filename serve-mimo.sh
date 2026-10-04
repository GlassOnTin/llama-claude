#!/bin/bash
# ==============================================================================
# MiMo-V2.6-Flash (309B MoE / 15B active, arch mimo2) — 3x RTX 5090 + RAM spill
# ==============================================================================
# Native 4-bit (MXFP4) weights ~167GB. Budget with 256GB system RAM:
#   dense + KV + buffers on GPU ~75GB  -> ~75GB experts in VRAM
#   remaining experts in RAM            -> ~92GB (~28 of 47 MoE layers)
# 256 experts, 8 fire per token -> per-token RAM traffic ~3.5GB.
#
# TUNE N_CPU_MOE on first load:
#   OOM on GPU      -> raise (30, 32)
#   VRAM to spare   -> lower for speed (hot experts hit VRAM; #27861 cache helps)
#
# Speculative decoding OFF by default (clean bench). Set SPEC_TYPE=dflash
# (dflash-Q8_0 drafter, 7-token predicts) or SPEC_TYPE=mtp (mtp-Q8_0 sidecar;
# the ggml-org repo reports a load bug on it — test both, trust the bench).
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
UBATCH="${UBATCH:-256}"
N_CPU_MOE="${N_CPU_MOE:-28}"
# Balanced split over 3x5090; adjust if load OOMs on a specific device
TENSOR_SPLIT="${TENSOR_SPLIT:-18,15,15}"
SPEC_TYPE="${SPEC_TYPE:-none}"

export CUDA_VISIBLE_DEVICES="0,1,2"

echo "========================================================"
echo " MiMo-V2.6-Flash (309B MoE / 15B active, 256 experts, 8 active)"
echo "========================================================"
echo " - Base Model:     $MODEL_FILE"
echo " - Vision mmproj:  $MMPROJ_FILE"
echo " - Context Size:   $CTX_SIZE (39/48 layers SWA-128: KV is cheap, 9 global layers)"
echo " - KV Cache:       Q8_0 K / Q8_0 V"
echo " - Layer Split:    $TENSOR_SPLIT"
echo " - Expert Spill:   first $N_CPU_MOE layers' experts in RAM"
echo " - Spec decoding:  $SPEC_TYPE"
echo " - Endpoint:       http://$HOST:$PORT"
echo "========================================================"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: model file not found: $MODEL_FILE"
    exit 1
fi

EXTRA_ARGS=()

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
    --n-cpu-moe "$N_CPU_MOE" \
    --ctx-size "$CTX_SIZE" \
    --parallel 1 \
    --batch-size 2048 \
    --ubatch-size "$UBATCH" \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --mmproj "$MMPROJ_FILE" \
    --jinja \
    --alias "mimo-v2.6-flash,claude-3-5-sonnet-20241022,default" \
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
#!/usr/bin/env bash
# ==============================================================================
# Upstream Watcher: Qwen3.8-Flash-Next MTP Speculative Decoding & Updates
# Checks ggml-org/llama.cpp PRs/commits and Hugging Face GGUF releases
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../upstream_watch.log"
DATE_STR="$(date '+%Y-%m-%d %H:%M:%S')"

echo "========================================================" | tee -a "$LOG_FILE"
echo " [Upstream Watch] Checking for Qwen3.8-Flash-Next MTP Updates ($DATE_STR)" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

FOUND_UPDATES=0

# 1. Check ggml-org/llama.cpp Pull Requests
echo "[+] Checking ggml-org/llama.cpp PRs..." | tee -a "$LOG_FILE"
PR_RESULTS=$(curl -s "https://api.github.com/repos/ggml-org/llama.cpp/pulls?state=all&sort=updated&per_page=30" | \
  jq -r '.[] | select((.title | test("qwen4exp|qwen3.8|mtp|speculative"; "i")) or (.body // "" | test("qwen4exp.*mtp|qwen3.8.*draft"; "i"))) | " - PR #" + (.number|tostring) + ": " + .title + " (" + .state + ") -> " + .html_url' 2>/dev/null || true)

if [ -n "$PR_RESULTS" ]; then
    echo "$PR_RESULTS" | tee -a "$LOG_FILE"
    FOUND_UPDATES=$((FOUND_UPDATES + 1))
else
    echo " - No new PR updates found." | tee -a "$LOG_FILE"
fi

# 2. Check Hugging Face for MTP / Flash-Next Draft GGUF Weights
echo "[+] Checking Hugging Face for MTP draft models..." | tee -a "$LOG_FILE"
HF_RESULTS=$(curl -s "https://huggingface.co/api/models?search=Qwen3.8-Flash-Next-MTP&sort=lastModified" | \
  jq -r '.[] | " - Model: " + .id + " (Updated: " + .lastModified + ")"' 2>/dev/null || true)

if [ -n "$HF_RESULTS" ]; then
    echo "$HF_RESULTS" | tee -a "$LOG_FILE"
    FOUND_UPDATES=$((FOUND_UPDATES + 1))
else
    echo " - No standalone MTP draft weights released yet." | tee -a "$LOG_FILE"
fi

# 3. Check llama.cpp git fetch status on master
if [ -d "/home/ian/models/llama.cpp/.git" ]; then
    echo "[+] Checking upstream git master for merged qwen4exp MTP commits..." | tee -a "$LOG_FILE"
    git -C /home/ian/models/llama.cpp fetch origin master --quiet 2>/dev/null || true
    GIT_COMMITS=$(git -C /home/ian/models/llama.cpp log origin/master --grep="qwen4exp" --grep="Qwen3.8" -n 5 --oneline 2>/dev/null || true)
    if [ -n "$GIT_COMMITS" ]; then
        echo "$GIT_COMMITS" | tee -a "$LOG_FILE"
    fi
fi

echo "========================================================" | tee -a "$LOG_FILE"
if [ "$FOUND_UPDATES" -gt 0 ]; then
    echo "Found relevant upstream activity. Check log: $LOG_FILE"
else
    echo "Watcher check complete. Everything up to date."
fi

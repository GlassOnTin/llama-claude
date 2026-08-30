# ==============================================================================
# Shell Module for Claude Code + Local Dual-GPU llama-server
# Add to ~/.bashrc: [ -f ~/Code/llama-claude/bash_module.sh ] && . ~/Code/llama-claude/bash_module.sh
# ==============================================================================

# Aliases to control the local server
alias llama-serve="ENABLE_VISION=1 $HOME/Code/llama-claude/serve.sh"
alias llama-serve-vision="ENABLE_VISION=1 $HOME/Code/llama-claude/serve.sh"
alias llama-serve-no-vision="ENABLE_VISION=0 $HOME/Code/llama-claude/serve.sh"
alias llama-serve-qwen="$HOME/Code/llama-claude/serve.sh"
alias llama-qwen="$HOME/Code/llama-claude/cli.sh"
alias llama-dl-qwen="$HOME/Code/llama-claude/download.sh"

# Claude Code launcher against local multi-GPU llama.cpp server (96GB VRAM)
unalias lclaude 2>/dev/null
lclaude() {
  ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8090}" \
  ANTHROPIC_API_KEY=llama-local \
  ANTHROPIC_AUTH_TOKEN=llama-local \
  ANTHROPIC_MODEL=qwen3.8-flash-next \
  ANTHROPIC_SMALL_FAST_MODEL=qwen3.8-flash-next \
  ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-flash-next \
  ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-flash-next \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-flash-next \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  CLAUDE_CODE_DISABLE_ARTIFACT=1 \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS="${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-220000}" \
  claude --disallowed-tools Artifact WebSearch Todo --model qwen3.8-flash-next "$@"
}

# Claude Code launcher for pure-text / fast compaction workflows
unalias lclaude-no-vision 2>/dev/null
lclaude-no-vision() {
  ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8090}" \
  ANTHROPIC_API_KEY=llama-local \
  ANTHROPIC_AUTH_TOKEN=llama-local \
  ANTHROPIC_MODEL=qwen3.8-flash-next \
  ANTHROPIC_SMALL_FAST_MODEL=qwen3.8-flash-next \
  ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-flash-next \
  ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-flash-next \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-flash-next \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  CLAUDE_CODE_DISABLE_ARTIFACT=1 \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS="${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-220000}" \
  claude --disallowed-tools Artifact WebSearch Todo --model qwen3.8-flash-next "$@"
}

# Haven Maintainer Bot launchers (isolated haven-bot user account)
alias lclaude-bot="sudo -u haven-bot /home/haven-bot/run-agent.sh"
alias lclaude-bot-no-vision="sudo -u haven-bot /home/haven-bot/run-agent.sh"
alias lclaude-no-vision-bot="sudo -u haven-bot /home/haven-bot/run-agent.sh"

# ==============================================================================
# Cross-User Session Handoff Tools (ian <-> haven-bot)
# ==============================================================================

# List recent sessions run by haven-bot for current or specified repo
claude-bot-sessions() {
  local target_dir="${1:-$PWD}"
  local repo_name="$(basename "$target_dir")"
  local bot_slug="-home-haven-bot-Code-${repo_name}"
  local bot_proj_dir="/home/haven-bot/.claude/projects/${bot_slug}"

  echo "=== Recent haven-bot sessions for ${repo_name} ==="
  sudo -u haven-bot python3 -c '
import glob, os, json, sys, datetime
proj_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(proj_dir, "*.jsonl")), key=os.path.getmtime, reverse=True)
if not files:
    print("No sessions found in " + proj_dir)
    sys.exit(0)
for f in files[:8]:
    sid = os.path.basename(f)[:-6]
    dt = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
    size_mb = os.path.getsize(f) / (1024 * 1024)
    last_prompt = ""
    try:
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                if "\"type\":\"last-prompt\"" in line:
                    data = json.loads(line)
                    last_prompt = data.get("lastPrompt", "")[:75]
    except Exception:
        pass
    print(f" • {sid}  [{dt}] ({size_mb:.2f} MB)")
    if last_prompt:
        print(f"     Last prompt: \"{last_prompt}\"")
' "$bot_proj_dir" 2>/dev/null
}

# Pull a session from haven-bot into your local session store and resume it
claude-takeover() {
  local target_dir="${1:-$PWD}"
  local repo_name="$(basename "$target_dir")"
  local bot_slug="-home-haven-bot-Code-${repo_name}"
  local ian_slug="$(printf '%s' "$target_dir" | sed 's#/#-#g')"
  local bot_proj_dir="/home/haven-bot/.claude/projects/${bot_slug}"
  local ian_proj_dir="$HOME/.claude/projects/${ian_slug}"

  mkdir -p "$ian_proj_dir"

  local session_id="$2"
  if [ -z "$session_id" ]; then
    if [[ "$1" =~ ^[0-9a-fA-F-]{36}$ ]]; then
      session_id="$1"
      target_dir="$PWD"
      repo_name="$(basename "$target_dir")"
      bot_slug="-home-haven-bot-Code-${repo_name}"
      bot_proj_dir="/home/haven-bot/.claude/projects/${bot_slug}"
      ian_slug="$(printf '%s' "$target_dir" | sed 's#/#-#g')"
      ian_proj_dir="$HOME/.claude/projects/${ian_slug}"
    else
      local latest_file="$(sudo -u haven-bot bash -c 'ls -t "$1"/*.jsonl 2>/dev/null | head -n 1' _ "$bot_proj_dir")"
      if [ -z "$latest_file" ]; then
        echo "No sessions found in ${bot_proj_dir}"
        return 1
      fi
      session_id="$(basename "$latest_file" .jsonl)"
    fi
  fi

  local src_file="${bot_proj_dir}/${session_id}.jsonl"
  local dst_file="${ian_proj_dir}/${session_id}.jsonl"

  echo "==> Importing haven-bot session: ${session_id}"
  if ! sudo -u haven-bot test -f "$src_file"; then
    echo "Error: Session file ${src_file} does not exist."
    return 1
  fi

  sudo -u haven-bot cat "$src_file" > "$dst_file"
  chmod 644 "$dst_file"
  echo "==> Resuming session in ${target_dir}..."
  cd "$target_dir" && lclaude --resume "$session_id"
}

# Push a local session to haven-bot and optionally launch it under the bot account
claude-pass-to-bot() {
  local target_dir="${1:-$PWD}"
  local repo_name="$(basename "$target_dir")"
  local bot_slug="-home-haven-bot-Code-${repo_name}"
  local ian_slug="$(printf '%s' "$target_dir" | sed 's#/#-#g')"
  local bot_proj_dir="/home/haven-bot/.claude/projects/${bot_slug}"
  local ian_proj_dir="$HOME/.claude/projects/${ian_slug}"

  sudo -u haven-bot mkdir -p "$bot_proj_dir"

  local session_id="$2"
  if [ -z "$session_id" ]; then
    if [[ "$1" =~ ^[0-9a-fA-F-]{36}$ ]]; then
      session_id="$1"
      target_dir="$PWD"
      repo_name="$(basename "$target_dir")"
      ian_slug="$(printf '%s' "$target_dir" | sed 's#/#-#g')"
      ian_proj_dir="$HOME/.claude/projects/${ian_slug}"
    else
      local latest_file="$(ls -t "${ian_proj_dir}"/*.jsonl 2>/dev/null | head -n 1)"
      if [ -z "$latest_file" ]; then
        echo "No sessions found in ${ian_proj_dir}"
        return 1
      fi
      session_id="$(basename "$latest_file" .jsonl)"
    fi
  fi

  local src_file="${ian_proj_dir}/${session_id}.jsonl"
  local dst_file="${bot_proj_dir}/${session_id}.jsonl"

  if [ ! -f "$src_file" ]; then
    echo "Error: Session file ${src_file} does not exist."
    return 1
  fi

  echo "==> Transferring session ${session_id} to haven-bot..."
  cat "$src_file" | sudo -u haven-bot tee "$dst_file" > /dev/null
  echo "==> Session ready for haven-bot."
  lclaude-bot --resume "$session_id"
}

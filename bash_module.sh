# ==============================================================================
# Shell Module for Claude Code + Local Dual-GPU llama-server
# Add to ~/.bashrc: [ -f ~/Code/llama-claude/bash_module.sh ] && . ~/Code/llama-claude/bash_module.sh
# ==============================================================================

# Aliases to control the local server
alias llama-serve="$HOME/Code/llama-claude/serve.sh"
alias llama-serve-qwen="$HOME/Code/llama-claude/serve.sh"
alias llama-qwen="$HOME/Code/llama-claude/cli.sh"
alias llama-dl-qwen="$HOME/Code/llama-claude/download.sh"

# Claude Code launcher against local multi-GPU llama.cpp server (96GB VRAM)
unalias lclaude 2>/dev/null
lclaude() {
  ANTHROPIC_BASE_URL=http://127.0.0.1:8090 \
  ANTHROPIC_API_KEY=llama-local \
  ANTHROPIC_AUTH_TOKEN=llama-local \
  ANTHROPIC_MODEL=qwen3.8-flash-next \
  ANTHROPIC_SMALL_FAST_MODEL=qwen3.8-flash-next \
  ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-flash-next \
  ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-flash-next \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-flash-next \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS="${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-50000}" \
  claude --model qwen3.8-flash-next "$@"
}

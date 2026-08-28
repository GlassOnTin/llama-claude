#!/bin/bash
# ==============================================================================
# Installer for Dual RTX 5090 llama.cpp Engine for Claude Code
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_USER="${SUDO_USER:-$USER}"
USER_HOME=$(eval echo "~$CURRENT_USER")

echo "========================================================"
echo " Installing llama-claude (Dual RTX 5090 Engine)"
echo " User: $CURRENT_USER | Directory: $SCRIPT_DIR"
echo "========================================================"

# --- 1. Shell Integration ---
echo ""
echo "[1/4] Installing Shell Integration..."
BASHRC="$USER_HOME/.bashrc"
MODULE_SOURCE="[ -f \"$SCRIPT_DIR/bash_module.sh\" ] && . \"$SCRIPT_DIR/bash_module.sh\""

if grep -Fxq "$MODULE_SOURCE" "$BASHRC" 2>/dev/null; then
    echo "  -> Shell module already present in $BASHRC"
else
    echo "  -> Adding module source line to $BASHRC"
    echo "" >> "$BASHRC"
    echo "# llama-claude (Dual 5090 local inference engine)" >> "$BASHRC"
    echo "$MODULE_SOURCE" >> "$BASHRC"
fi

# --- 2. Host Hardening (udev, modprobe, ASPM) ---
echo ""
echo "[2/4] Host Configuration (PCIe ASPM, udev hotplug, modprobe)..."
if [ "$EUID" -eq 0 ]; then
    bash "$SCRIPT_DIR/host-config/setup-host.sh"
else
    echo "  -> Applying host configuration via sudo..."
    sudo "$SCRIPT_DIR/host-config/setup-host.sh" || {
        echo "  -> Note: Run 'sudo $SCRIPT_DIR/host-config/setup-host.sh' if needed later."
    }
fi

# --- 3. Systemd Service ---
echo ""
echo "[3/4] Installing systemd service..."
USER_SYSTEMD_DIR="$USER_HOME/.config/systemd/user"
mkdir -p "$USER_SYSTEMD_DIR"

cat << EOF > "$USER_SYSTEMD_DIR/llama-qwen.service"
[Unit]
Description=llama.cpp Triple RTX 5090 Server (Qwen3.8-Flash-Next)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/serve.sh
Restart=on-failure
RestartSec=5s
KillSignal=SIGINT
TimeoutStopSec=15s
LimitNOFILE=65536
Environment="PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload 2>/dev/null || true
echo "  -> Installed user systemd service: $USER_SYSTEMD_DIR/llama-qwen.service"

# --- 4. Model Weights Check ---
echo ""
echo "[4/4] Verifying Model Weights..."
MODEL_PATH="$USER_HOME/models/Qwen3.8-27B-Q8_0.gguf"
if [ -f "$MODEL_PATH" ]; then
    echo "  -> Model weights found: $MODEL_PATH"
else
    echo "  -> Model weights not yet downloaded."
    echo "     Run: $SCRIPT_DIR/download.sh to download base model, vision projector, and MTP draft head."
fi

echo ""
echo "========================================================"
echo " Installation Complete!"
echo "========================================================"
echo " Service Management (No Sudo Needed):"
echo "   systemctl --user enable --now llama-qwen   # Start on boot & launch now"
echo "   systemctl --user status llama-qwen         # Check service status"
echo "   systemctl --user stop llama-qwen           # Stop service"
echo "   systemctl --user restart llama-qwen        # Restart service"
echo ""
echo " Usage:"
echo "   1. Launch Claude Code:          lclaude"
echo "   2. Launch Bot Agent:            lclaude-bot"
echo "   3. Interactive CLI chat:        llama-qwen"
echo "========================================================"

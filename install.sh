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
    echo "  -> To apply kernel/udev host settings, run:"
    echo "     sudo $SCRIPT_DIR/host-config/setup-host.sh"
fi

# --- 3. Systemd Service ---
echo ""
echo "[3/4] Configuring systemd service..."
SERVICE_SRC="$SCRIPT_DIR/systemd/llama-qwen.service"
SERVICE_DST="/etc/systemd/system/llama-qwen.service"

if [ "$EUID" -eq 0 ]; then
    # Adjust user if run under sudo
    sed -i "s|^User=.*|User=$CURRENT_USER|" "$SERVICE_SRC"
    sed -i "s|^Group=.*|Group=$(id -gn "$CURRENT_USER")|" "$SERVICE_SRC"
    sed -i "s|^WorkingDirectory=.*|WorkingDirectory=$SCRIPT_DIR|" "$SERVICE_SRC"
    sed -i "s|^ExecStart=.*|ExecStart=$SCRIPT_DIR/serve.sh|" "$SERVICE_SRC"
    
    cp "$SERVICE_SRC" "$SERVICE_DST"
    systemctl daemon-reload
    echo "  -> Installed $SERVICE_DST"
    echo "  -> Service commands:"
    echo "     sudo systemctl enable llama-qwen"
    echo "     sudo systemctl start llama-qwen"
    echo "     sudo systemctl status llama-qwen"
else
    echo "  -> To install the systemd service, run:"
    echo "     sudo cp $SERVICE_SRC $SERVICE_DST && sudo systemctl daemon-reload"
fi

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
echo " Usage:"
echo "   1. Start server:          llama-serve-qwen   (or sudo systemctl start llama-qwen)"
echo "   2. Launch Claude Code:    lclaude"
echo "   3. Interactive CLI chat:  llama-qwen"
echo "========================================================"

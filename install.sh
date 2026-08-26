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
SERVICE_SRC="$SCRIPT_DIR/systemd/llama-qwen.service"
SERVICE_DST="/etc/systemd/system/llama-qwen.service"

CONFIGURED_SERVICE=$(mktemp)
sed -e "s|^User=.*|User=$CURRENT_USER|" \
    -e "s|^Group=.*|Group=$(id -gn "$CURRENT_USER")|" \
    -e "s|^WorkingDirectory=.*|WorkingDirectory=$SCRIPT_DIR|" \
    -e "s|^ExecStart=.*|ExecStart=$SCRIPT_DIR/serve.sh|" \
    "$SERVICE_SRC" > "$CONFIGURED_SERVICE"

if [ "$EUID" -eq 0 ]; then
    cp "$CONFIGURED_SERVICE" "$SERVICE_DST"
    systemctl daemon-reload
    rm -f "$CONFIGURED_SERVICE"
    echo "  -> Successfully installed $SERVICE_DST"
else
    echo "  -> Copying service unit to $SERVICE_DST (via sudo)..."
    if sudo cp "$CONFIGURED_SERVICE" "$SERVICE_DST"; then
        sudo systemctl daemon-reload
        rm -f "$CONFIGURED_SERVICE"
        echo "  -> Successfully installed $SERVICE_DST"
    else
        rm -f "$CONFIGURED_SERVICE"
        echo "  -> Failed to install service. You can manually run:"
        echo "     sudo cp $SERVICE_SRC $SERVICE_DST && sudo systemctl daemon-reload"
    fi
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
echo " Service Management:"
echo "   sudo systemctl enable --now llama-qwen   # Start on boot & launch now"
echo "   sudo systemctl status llama-qwen         # Check service status"
echo "   sudo systemctl stop llama-qwen           # Stop service"
echo ""
echo " Usage:"
echo "   1. Start server interactively:  llama-serve-qwen"
echo "   2. Launch Claude Code:          lclaude"
echo "   3. Interactive CLI chat:        llama-qwen"
echo "========================================================"

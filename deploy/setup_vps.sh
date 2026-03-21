#!/bin/bash
# ═══════════════════════════════════════════════════
# Scalp2 VPS Setup Script — DigitalOcean Ubuntu 22.04+
# ═══════════════════════════════════════════════════
# Usage:
#   scp deploy/setup_vps.sh root@your-vps-ip:/tmp/
#   ssh root@your-vps-ip 'bash /tmp/setup_vps.sh'
#
# After setup:
#   1. Copy checkpoints to /opt/scalp2/checkpoints/
#   2. Edit /opt/scalp2/.env with API keys
#   3. systemctl start scalp2
# ═══════════════════════════════════════════════════

set -euo pipefail

echo "=== Scalp2 VPS Setup ==="

# 1. System packages
echo ">>> Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3-pip git

# 2. Create dedicated user (non-root)
echo ">>> Creating scalp2 user..."
if ! id -u scalp2 &>/dev/null; then
    useradd --system --home-dir /opt/scalp2 --shell /bin/bash scalp2
fi

# 3. Clone repository
echo ">>> Cloning Scalp2 repo..."
INSTALL_DIR="/opt/scalp2"
if [ -d "$INSTALL_DIR/.git" ]; then
    git -C "$INSTALL_DIR" pull --ff-only
else
    git clone https://github.com/sergul74/Scalp2.git "$INSTALL_DIR"
fi

# 4. Python virtual environment
echo ">>> Setting up Python venv..."
python3.12 -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"

# Install CPU-only PyTorch (no GPU needed for inference)
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet xgboost ccxt PyWavelets hmmlearn numba scikit-learn \
    pyyaml tqdm pyarrow pandas numpy requests

# 5. Create directories
mkdir -p "$INSTALL_DIR/checkpoints"
mkdir -p "$INSTALL_DIR/state"
mkdir -p "$INSTALL_DIR/logs"

# 6. Copy .env template if not exists
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    echo ">>> Created .env from template — EDIT IT with your API keys!"
fi

# 7. Set permissions
chown -R scalp2:scalp2 "$INSTALL_DIR"
chmod 600 "$INSTALL_DIR/.env"

# 8. Install systemd service
echo ">>> Installing systemd service..."
cp "$INSTALL_DIR/deploy/scalp2.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable scalp2

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy model checkpoints to $INSTALL_DIR/checkpoints/"
echo "     scp -r checkpoints/ root@this-vps:$INSTALL_DIR/checkpoints/"
echo ""
echo "  2. Edit .env with your Binance API keys:"
echo "     nano $INSTALL_DIR/.env"
echo ""
echo "  3. Start the bot:"
echo "     systemctl start scalp2"
echo ""
echo "  4. Check logs:"
echo "     journalctl -u scalp2 -f"
echo ""
echo "  5. Stop the bot:"
echo "     systemctl stop scalp2"
echo ""

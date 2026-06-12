# Server deployment (Ubuntu 24.04 VPS, 2 vCPU / 4 GB)

Testnet bot — still NEVER put real-money keys on any of this.

## 1. Base setup (as root, once)
    adduser kronos && usermod -aG sudo kronos
    apt update && apt install -y python3.12-venv git ufw
    ufw allow OpenSSH && ufw enable        # nothing else exposed — ever

## 2. Code + environment (as kronos)
    git clone https://github.com/Zentazuke/kronos-scalping-bot.git
    cd kronos-scalping-bot
    git clone https://github.com/NeoQuasar/Kronos.git   # model classes (gitignored)
    python3 -m venv .venv && . .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

## 3. Secrets
Copy `.env` from the PC (scp / paste over SSH). Never commit it.
Keep HEADLESS=true on servers (systemd owns the console; bot.log still written).
First boot downloads the Kronos weights from Hugging Face — remove
HF_HUB_OFFLINE=1 from `.env` for that boot, put it back after.

## 4. Services
    sudo cp deploy/kronos-*.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now kronos-bot kronos-dashboard
    journalctl -u kronos-bot -f          # live logs

## 5. Dashboard access — Tailscale (do NOT open port 8765 to the internet;
the dashboard has no authentication)
    curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
Install Tailscale on the phone/PC too, then open http://<tailscale-ip>:8765
from anywhere. Alternative: SSH tunnel `ssh -L 8765:localhost:8765 kronos@server`.

## 6. Day-2 operations
    git -C ~/kronos-scalping-bot pull && sudo systemctl restart kronos-bot   # deploy update
    sudo systemctl stop kronos-bot                                           # pause trading
    .venv/bin/python learner.py train                                        # retrain meta
    .venv/bin/python calibrate.py BTC/USDT --days 14 --stride 3              # calibration
    rm emergency_lock.lock   # ONLY after inspecting why the kill switch fired

## Notes
- journal.db/bot.log live on the server then; copy back with scp for analysis.
- Linux keeps time via NTP — the Windows clock-drift issue does not exist here.
- Binance Spot Testnet keys work from any IP. For real keys (DON'T yet):
  IP-whitelist them to the server only.

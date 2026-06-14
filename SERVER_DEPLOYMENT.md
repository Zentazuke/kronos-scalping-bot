# Kronos Bot — Server Deployment Runbook

Move the bot off your PC onto a small Linux VPS so it runs 24/7. Follow the
phases in order. Each ends with a **✓ Verify** step — don't move on until it passes.

> **Safety:** this is the **testnet** bot. Never put real-money exchange keys on the
> server. `USE_SANDBOX=true` must stay set. The dashboard has **no login**, so it must
> never be exposed to the public internet (Phase 6 uses Tailscale for that reason).

---

## Phase 0 — Pre-flight (gather these first)

You need:

- **A VPS**: Ubuntu 24.04, **2 vCPU / 4 GB RAM** minimum (Hetzner CX22 / Netcup VPS 500 — ~€4/mo). 4 GB is required: PyTorch + Kronos won't fit in 2 GB.
- **SSH access** to it (IP + password or key).
- **Your GitHub token** — it's the `GITHUB_TOKEN=` line in your PC's `.env`.
- **Two files from your PC**: `.env` (your keys/config) and `journal.db` (your ~300 trades of learning data).

**Decision — fresh box or resuming?**
- *Fresh box* → start at Phase 1.
- *Resuming the box you partly set up earlier* → the `kronos` user and Python are likely already there. **Skip to Phase 2** and use the `git pull` branch to grab today's code (4 symbols, strategy equity, mainnet macro, new dashboard).

---

## Phase 1 — Base OS setup  *(as `root`, once)*

```bash
ssh root@YOUR_SERVER_IP            # set a new password if it forces one

adduser --disabled-password --gecos "" kronos
echo "kronos:CHOOSE_A_PASSWORD" | chpasswd
usermod -aG sudo kronos
apt update && apt install -y python3.12-venv git ufw
ufw allow OpenSSH && echo "y" | ufw enable
su - kronos
```

**✓ Verify:** prompt now reads `kronos@...:~$`, and `python3 --version` shows **3.12.x**.
(3.12 matters — newer Python may lack a prebuilt PyTorch wheel.)

---

## Phase 2 — Code & dependencies  *(as `kronos`)*

**If fresh** (replace `YOUR_GITHUB_TOKEN` with the token from your `.env`):

```bash
cd ~
git clone https://YOUR_GITHUB_TOKEN@github.com/Zentazuke/kronos-scalping-bot.git
cd kronos-scalping-bot
git clone https://github.com/shiyu-coder/Kronos.git     # model classes (NOT NeoQuasar — that URL is wrong)
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only, avoids 2 GB of CUDA
pip install -r requirements.txt
```

**If resuming** (repo already cloned — just get today's code):

```bash
cd ~/kronos-scalping-bot
git pull                                                # pulls strategy equity, 4 symbols, mainnet macro, new dashboard
. .venv/bin/activate
pip install -r requirements.txt                         # no-op unless deps changed
[ -d Kronos ] || git clone https://github.com/shiyu-coder/Kronos.git
```

**✓ Verify:**
```bash
python -c "import torch, ccxt, transformers, pandas; print('deps OK')"
grep -c "BTC/USDT,ADA/USDT,ETH/USDT,BNB/USDT" feed.py   # should print 1 (the 4-symbol default)
```

---

## Phase 3 — Configuration (`.env`)

From a **second terminal on your PC**, in the Trading Bot folder:

```powershell
scp .env kronos@YOUR_SERVER_IP:~/kronos-scalping-bot/.env
```

Back on the server, edit it: `nano ~/kronos-scalping-bot/.env`

- Set **`HEADLESS=true`** (no terminal UI on a server; the bot still writes `bot.log`).
- **Delete** the `HF_HUB_OFFLINE=1` line **for the first boot only** (the first run downloads ~1 GB of Kronos weights from Hugging Face). Re-add it after the model is cached.
- Confirm **`USE_SANDBOX=true`** is present (safety — never trade real money here).
- Symbols default to BTC/ADA/ETH/BNB in code; to override, add `SYMBOLS=BTC/USDT,ETH/USDT` etc.

**✓ Verify:** `grep -E "HEADLESS|USE_SANDBOX|HF_HUB_OFFLINE" .env` shows `HEADLESS=true`, `USE_SANDBOX=true`, and **no** `HF_HUB_OFFLINE`.

---

## Phase 4 — Carry over your learning data

Your `journal.db` holds ~300 decided trades with their full feature vectors — the meta-learner's training data. Without it the server starts from zero.

> Do this in **Phase 5, right after stopping the PC bot**, so it's the freshest copy. Listed here so you don't forget it exists.

```powershell
# from the PC, Trading Bot folder (run during cutover):
scp journal.db kronos@YOUR_SERVER_IP:~/kronos-scalping-bot/journal.db
```

---

## Phase 5 — Cutover (stop PC → start server)  ⚠️ the one critical sequence

**Two bots on the same testnet account double-trade and fight over orders.** Order matters:

1. **On your PC:** stop the bot (Ctrl+C in its window) and close the dashboard window. The PC bot must be fully stopped.
2. **From the PC:** copy the now-final journal over (the `scp journal.db ...` from Phase 4).
3. **On the server:** install and start the services:

```bash
cd ~/kronos-scalping-bot
sudo cp deploy/kronos-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kronos-bot kronos-dashboard
journalctl -u kronos-bot -f          # live logs; Ctrl+C exits the log view, not the bot
```

First boot downloads the model (~1 GB) — give it a few minutes before bars start.

**✓ Verify (in the live log):**
- `supervisor boot: symbols=BTC/USDT, ADA/USDT, ETH/USDT, BNB/USDT`  ← 4 symbols
- `daily baseline equity snapshot: ~10000 USDT`  ← strategy equity (not 74k)
- `daily macro context refreshed (250 candles)`  ← mainnet macro (not 11)
- within ~5 min, trades journaling across the symbols

After this boot succeeds, re-add `HF_HUB_OFFLINE=1` to `.env` and `sudo systemctl restart kronos-bot`.

---

## Phase 6 — Dashboard access (Tailscale)

The dashboard has no auth, so **never open port 8765 to the internet.** Tailscale puts your server, PC and phone on a private mesh instead.

```bash
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
```

Follow the login link it prints, install the Tailscale app on your phone/PC (same account), then open `http://<server-tailscale-ip>:8765` from anywhere.

> The dashboard fetches candles over the internet (testnet 5-min + mainnet daily for macro). The server has outbound internet, so this just works.

**✓ Verify:** the dashboard loads on your phone, shows 4 symbols, the candlestick chart renders, and "Strategy equity" reads ~10k.

---

## Phase 7 — Final checklist

- [ ] `systemctl status kronos-bot` → **active (running)**
- [ ] `systemctl status kronos-dashboard` → **active (running)**
- [ ] Dashboard reachable via Tailscale, 4 symbols, candlestick chart + markers
- [ ] Strategy equity ~10k; drawdown bar moving
- [ ] `journalctl -u kronos-bot --since "10 min ago" | grep -c journaled` → trades accruing
- [ ] PC bot is **OFF** (only one bot per testnet account)

---

## Phase 8 — Day-2 operations

```bash
# Deploy a code update (your push-to-GitHub IS the deploy pipeline):
git -C ~/kronos-scalping-bot pull && sudo systemctl restart kronos-bot

# Restart the dashboard after a dashboard.html / server change:
sudo systemctl restart kronos-dashboard

# Pause trading:
sudo systemctl stop kronos-bot

# Live logs:
journalctl -u kronos-bot -f

# Retrain the meta-labeler / run a walk-forward (offline, safe while running):
~/kronos-scalping-bot/.venv/bin/python learner.py walkforward --db journal.db
~/kronos-scalping-bot/.venv/bin/python learner.py train --db journal.db
```

**Kill switch:** if the drawdown breaker fires, the bot exits non-zero and writes
`emergency_lock.lock`, which blocks every restart until **a human deletes it** —
deliberate, so systemd's auto-restart can't override a real stop. Investigate first,
then `rm emergency_lock.lock` to resume.

---

## Gotchas worth knowing

- **CPU on 4 symbols:** the bot runs Kronos inference per symbol per bar — 4 inferences every 5 min on 2 vCPUs. Fine with headroom, but if you ever see `inference failure — safe-state NEUTRAL` timeouts in the log, either raise `INFERENCE_TIMEOUT_S` in `.env`, drop to 2–3 symbols, or move up a VPS tier.
- **Kronos repo URL:** clone **`shiyu-coder/Kronos`**, not `NeoQuasar/Kronos` (the old DEPLOY.md typo). The HF *weights* `NeoQuasar/Kronos-small` download automatically — that's separate from the code repo.
- **Rollback:** if an update misbehaves, `git -C ~/kronos-scalping-bot log --oneline -5`, then `git checkout <good-commit> -- .` and restart. Your `.env` and `journal.db` are gitignored, so they're never touched by a checkout.
- **Phantom fills:** testnet's thin liquidity gives unrealistic exit fills — testnet PnL is not real performance. (Known; doesn't affect the bot's running, just how you read results.)

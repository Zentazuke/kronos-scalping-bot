"""intraday_tsm_live.py — trade the intraday-TSM edge LIVE on testnet.

This is the forward-validated strategy (vol-gated + momentum-regime + on-chain risk-off
gate), now placing REAL testnet orders instead of shadow-logging. It EXECUTES the trial's
committed decisions from tsm_forward.db verbatim — ONE BRAIN (the forward logger), ONE
executor (this) — so the live book is provably identical to the shadow forward test.
Sandbox-only, both directions, one position per coin per day.

  * ENTER  (run >= 08:05 UTC): refresh the TRIAL's committed decisions (run_daily), then
    place a market order ($NOTIONAL each) for every coin the trial committed LONG/SHORT
    today. We do NOT recompute the signal here — that independent recompute was the bug
    that let the live book take LONGs the trial gated out. Records to tsm_live.db.
  * EXIT   (run >= 00:05 UTC next day): flatten every open position (cancel orders +
    market close), record realized P&L. This is the strategy's day-close exit.
  * STATUS : show open positions with live unrealized P&L.

Safety: refuses to run unless USE_SANDBOX=true; idempotent (won't double-enter/exit a
day); order failures are logged and skipped, never crash; --dry-run computes the signal
and would-be orders WITHOUT placing anything.

    python intraday_tsm_live.py enter --dry-run     # safe: see today's signals, no orders
    python intraday_tsm_live.py enter               # place today's entries (08:05 UTC)
    python intraday_tsm_live.py exit                # flatten everything (00:05 UTC)
    python intraday_tsm_live.py status              # live P&L of open positions

Schedule (cron, UTC):  5 8 * * *  -> enter   ·   5 0 * * *  -> exit
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:  # load .env so a cron run has USE_SANDBOX + API keys without sourcing it
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:  # noqa: BLE001
    pass

import intraday_tsm_forward as fwd       # the trial brain: we execute ITS decisions

DB_PATH = "tsm_live.db"
FWD_DB = "tsm_forward.db"                # source of truth for today's committed decisions
NOTIONAL = float(os.getenv("TSM_NOTIONAL", "200"))   # $ per trade
# Count the realized track record only from the corrected strategy's start — pre-rewire
# trades were a bug, not the strategy. Override via .env (YYYY-MM-DD).
TSM_LIVE_START = os.getenv("TSM_LIVE_START", "2026-06-27")


def ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS positions (
            day TEXT, symbol TEXT, side TEXT, amount REAL, entry_price REAL, entry_ts TEXT,
            status TEXT, exit_price REAL, exit_ts TEXT, pnl REAL, order_id TEXT,
            PRIMARY KEY (day, symbol))""")
    # exit_by records WHO closed the trade: 'auto' (the 00:05 day-close cron) vs 'manual'
    # (you, via the dashboard) — so we can compare discretionary exits to mechanical ones.
    try:
        conn.execute("ALTER TABLE positions ADD COLUMN exit_by TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def exchange(dry: bool):
    """Sandbox ccxt client mirroring the bot's config (no defaultType). None if not
    configured. In --dry-run we still build it for price reads but never order."""
    if os.getenv("USE_SANDBOX", "").strip().lower() != "true":
        print("REFUSED: USE_SANDBOX is not 'true' — live trading disabled."); return None
    key, sec = os.getenv("EXCHANGE_API_KEY"), os.getenv("EXCHANGE_API_SECRET")
    if not (key and sec):
        print("REFUSED: no API keys in env."); return None
    import ccxt  # type: ignore[import-untyped]
    ex = getattr(ccxt, os.getenv("EXCHANGE_ID", "binance"))(
        {"enableRateLimit": True, "apiKey": key, "secret": sec,
         "options": {"adjustForTimeDifference": True}})
    sm = getattr(ex, "set_sandbox_mode", None)
    if callable(sm):
        sm(True)
    ex.load_markets()
    return ex


def _todays_decisions(today: str) -> List[tuple]:
    """The single source of truth: read the TRIAL's committed directional decisions for
    `today` from tsm_forward.db. Returns [(symbol, direction), ...] for LONG/SHORT only
    (FLAT / gated days are skipped). This is what makes the live book == trial by
    construction — we execute what the one brain committed, nothing else."""
    if not os.path.exists(FWD_DB):
        return []
    c = sqlite3.connect(FWD_DB)
    try:
        return c.execute(
            "SELECT symbol, direction FROM forward_trades "
            "WHERE decision_day=? AND direction IN ('LONG','SHORT')", (today,)).fetchall()
    finally:
        c.close()


def do_enter(dry: bool) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # ONE BRAIN: make sure the trial has committed today's decisions, then execute exactly
    # those. We no longer recompute the signal here — that independent recompute was the
    # source of the live/trial divergence (live took LONGs the trial gated out).
    try:
        fwd.run_daily()
    except Exception as exc:  # noqa: BLE001 — fall back to whatever is already committed
        print(f"  warning: couldn't refresh trial decisions ({str(exc)[:60]}); "
              f"using existing {FWD_DB}.")
    ex = exchange(dry)
    if ex is None:
        return 1
    decisions = _todays_decisions(today)
    if not decisions:
        print(f"[{_now_iso()}] enter: trial committed no directional trades for {today} "
              f"(all gated/FLAT){' (dry-run)' if dry else ''}.")
        return 0
    conn = sqlite3.connect(DB_PATH); ensure(conn)
    placed = 0
    for sym, direction in decisions:
        if conn.execute("SELECT 1 FROM positions WHERE day=? AND symbol=?", (today, sym)).fetchone():
            continue                                      # already executed this coin today
        try:
            price = float(ex.fetch_ticker(sym)["last"])
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: ticker failed ({str(exc)[:40]})"); continue
        raw_amt = NOTIONAL / price
        amount = float(ex.amount_to_precision(sym, raw_amt))
        side = "buy" if direction == "LONG" else "sell"
        if dry:
            print(f"  DRY {sym:<10} {direction:<5} (trial) would {side} {amount} @ ~{price}")
            continue
        try:
            order = ex.create_order(sym, "market", side, amount)
            fill = float(order.get("average") or order.get("price") or price)
            conn.execute(
                "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (today, sym, direction, amount, fill, _now_iso(), "OPEN",
                 None, None, None, str(order.get("id"))))
            conn.commit(); placed += 1
            print(f"  ENTER {sym:<10} {direction:<5} {side} {amount} @ {fill}  [trial decision]")
        except Exception as exc:  # noqa: BLE001 — never crash the run on one bad order
            print(f"  {sym}: ORDER FAILED ({str(exc)[:90]})")
    conn.close()
    print(f"[{_now_iso()}] enter: {placed} position(s) opened for {today} "
          f"(executing trial decisions){' (dry-run)' if dry else ''}.")
    return 0


def do_exit(dry: bool) -> int:
    ex = exchange(dry)
    if ex is None:
        return 1
    conn = sqlite3.connect(DB_PATH); ensure(conn)
    opens = conn.execute(
        "SELECT day,symbol,side,amount,entry_price FROM positions WHERE status='OPEN'").fetchall()
    closed = 0
    for day, sym, side, amount, entry in opens:
        exit_side = "sell" if side == "LONG" else "buy"
        if dry:
            print(f"  DRY would flatten {sym} ({exit_side} {amount})"); continue
        try:
            ca = getattr(ex, "cancel_all_orders", None)
            if callable(ca):
                ca(sym)
        except Exception:  # noqa: BLE001
            pass
        try:
            order = ex.create_order(sym, "market", exit_side, float(amount))
            fill = float(order.get("average") or order.get("price") or 0) or float(ex.fetch_ticker(sym)["last"])
            sign = 1.0 if side == "LONG" else -1.0
            pnl = sign * (fill - float(entry)) * float(amount)
            conn.execute(
                "UPDATE positions SET status='CLOSED', exit_price=?, exit_ts=?, pnl=?, exit_by='auto' "
                "WHERE day=? AND symbol=?", (fill, _now_iso(), pnl, day, sym))
            conn.commit(); closed += 1
            print(f"  EXIT  {sym:<10} {side:<5} @ {fill}  pnl {pnl:+.2f} $")
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: CLOSE FAILED ({str(exc)[:90]})")
    conn.close()
    print(f"[{_now_iso()}] exit: {closed} position(s) flattened{' (dry-run)' if dry else ''}.")
    return 0


def do_status() -> int:
    if not os.path.exists(DB_PATH):
        print("no tsm_live.db yet — run 'enter' first."); return 0
    conn = sqlite3.connect(DB_PATH); ensure(conn)
    opens = conn.execute(
        "SELECT day,symbol,side,amount,entry_price FROM positions WHERE status='OPEN'").fetchall()
    closed = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(pnl),0) FROM positions WHERE status='CLOSED' AND day>=?",
        (TSM_LIVE_START,)).fetchone()
    conn.close()
    print(f"\n=== INTRADAY-TSM LIVE (testnet) — ${NOTIONAL:g}/trade ===")
    print(f"realized (since {TSM_LIVE_START}): {closed[0]} closed trades, total P&L {closed[1]:+.2f} $\n")
    if not opens:
        print("no open positions."); return 0
    try:
        ex = exchange(False)
    except Exception:  # noqa: BLE001
        ex = None
    print(f"{'symbol':<10}{'side':>6}{'amount':>12}{'entry':>12}{'now':>12}{'P&L $':>10}{'P&L %':>9}")
    for day, sym, side, amount, entry in opens:
        mark = None
        if ex is not None:
            try:
                mark = float(ex.fetch_ticker(sym)["last"])
            except Exception:  # noqa: BLE001
                mark = None
        if mark and entry:
            sign = 1.0 if side == "LONG" else -1.0
            pnl = sign * (mark - entry) * amount
            pct = sign * (mark / entry - 1) * 100
            print(f"{sym:<10}{side:>6}{amount:>12.6f}{entry:>12.4f}{mark:>12.4f}{pnl:>+9.2f}{pct:>+8.2f}%")
        else:
            print(f"{sym:<10}{side:>6}{amount:>12.6f}{entry:>12.4f}{'?':>12}{'?':>10}{'?':>9}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Intraday-TSM live testnet trader")
    ap.add_argument("action", choices=["enter", "exit", "status"])
    ap.add_argument("--dry-run", action="store_true", help="compute signals/orders without placing them")
    args = ap.parse_args()
    if args.action == "enter":
        return do_enter(args.dry_run)
    if args.action == "exit":
        return do_exit(args.dry_run)
    return do_status()


if __name__ == "__main__":
    raise SystemExit(main())

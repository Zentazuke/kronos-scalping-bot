"""intraday_tsm_live.py — trade the intraday-TSM edge LIVE on testnet.

This is the forward-validated strategy (vol-gated + momentum-regime + on-chain risk-off
gate), now placing REAL testnet orders instead of shadow-logging. It reuses the EXACT
signal functions from intraday_tsm_forward, so what trades is provably what we backtested
and forward-tested. Sandbox-only, both directions, one position per coin per day.

  * ENTER  (run >= 08:05 UTC): for each coin in the basket, if the signal fires, place a
    market order ($NOTIONAL each), sized to the venue's precision. Mirrors the bot's own
    create_order(symbol, "market", buy/sell). Records to tsm_live.db.
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

from consensus_backtest import fetch_ohlcv
from intraday_tsm_forward import (FETCH_DAYS, SPLIT, _day, _hour, day_prices,
                                  onchain_risk_off, regime_on, trailing_threshold)

DB_PATH = "tsm_live.db"
BASKET = ["ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT"]
NOTIONAL = float(os.getenv("TSM_NOTIONAL", "200"))   # $ per trade


def ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS positions (
            day TEXT, symbol TEXT, side TEXT, amount REAL, entry_price REAL, entry_ts TEXT,
            status TEXT, exit_price REAL, exit_ts TEXT, pnl REAL, order_id TEXT,
            PRIMARY KEY (day, symbol))""")
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


def _signal(candles, today: str) -> Optional[str]:
    """Reuse the forward logger's exact gates -> 'LONG' / 'SHORT' / None."""
    today_bars = [b for b in candles if _day(int(b[0])) == today]
    if not any(_hour(int(b[0])) == SPLIT - 1 for b in today_bars):
        return None                                       # morning window not complete yet
    open_px, split_px, _c = day_prices(candles, today, SPLIT)
    if not open_px or not split_px:
        return None
    thr = trailing_threshold(candles, today)
    regime = regime_on(candles, today)
    if thr is None or regime is None:
        return None
    morning = split_px / open_px - 1.0
    risk_off = onchain_risk_off(today)
    long_blocked = morning > 0 and risk_off
    if abs(morning) >= thr and regime and not long_blocked:
        return "LONG" if morning > 0 else "SHORT"
    return None


def do_enter(dry: bool) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ex = exchange(dry)
    if ex is None:
        return 1
    conn = sqlite3.connect(DB_PATH); ensure(conn)
    placed = 0
    for sym in BASKET:
        if conn.execute("SELECT 1 FROM positions WHERE day=? AND symbol=?", (today, sym)).fetchone():
            continue                                      # already decided today
        try:
            candles = fetch_ohlcv(sym, "1h", FETCH_DAYS)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: fetch failed ({str(exc)[:40]})"); continue
        direction = _signal(candles, today)
        if direction is None:
            continue                                      # gated out (or too early) — try again next run
        price = float(ex.fetch_ticker(sym)["last"])
        raw_amt = NOTIONAL / price
        amount = float(ex.amount_to_precision(sym, raw_amt))
        side = "buy" if direction == "LONG" else "sell"
        if dry:
            print(f"  DRY {sym:<10} {direction:<5} would {side} {amount} @ ~{price}")
            continue
        try:
            order = ex.create_order(sym, "market", side, amount)
            fill = float(order.get("average") or order.get("price") or price)
            conn.execute(
                "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (today, sym, direction, amount, fill, _now_iso(), "OPEN",
                 None, None, None, str(order.get("id"))))
            conn.commit(); placed += 1
            print(f"  ENTER {sym:<10} {direction:<5} {side} {amount} @ {fill}")
        except Exception as exc:  # noqa: BLE001 — never crash the run on one bad order
            print(f"  {sym}: ORDER FAILED ({str(exc)[:90]})")
    conn.close()
    print(f"[{_now_iso()}] enter: {placed} position(s) opened for {today}"
          f"{' (dry-run)' if dry else ''}.")
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
                "UPDATE positions SET status='CLOSED', exit_price=?, exit_ts=?, pnl=? "
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
        "SELECT COUNT(*), COALESCE(SUM(pnl),0) FROM positions WHERE status='CLOSED'").fetchone()
    conn.close()
    print(f"\n=== INTRADAY-TSM LIVE (testnet) — ${NOTIONAL:g}/trade ===")
    print(f"realized: {closed[0]} closed trades, total P&L {closed[1]:+.2f} $\n")
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

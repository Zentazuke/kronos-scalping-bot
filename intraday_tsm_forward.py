"""intraday_tsm_forward.py — LIVE shadow forward-test of the intraday-TSM edge.

This is the gold-standard test: the edge that survived the backtest (trailing gate,
holdout, split sweep, 2x-fee stress) now has to prove itself on data that did not
exist when we found it. SHADOW mode = it logs a pre-committed decision and entry
price each day, then records the outcome at day-close. No orders, no risk — just an
honest, out-of-sample track record we can watch fill in.

It reuses the EXACT functions from intraday_tsm_backtest (day_samples) and the same
candle fetch (consensus_backtest.fetch_ohlcv), so the live signal is provably the
same rule we tested. Locked config (chosen for robustness, NOT best-OOS):

    split 08:00 UTC · both directions · trailing 60-day vol gate (top tertile) · 10bps

How it runs (idempotent — safe to run repeatedly):
  * SETTLE: any logged day that has now fully closed gets its exit price + net return.
  * DECIDE: once it's past 08:00 UTC and today isn't logged yet, compute each coin's
    morning return (00:00->08:00), gate on the trailing-60d threshold, and log the
    pre-committed direction + entry price (the 07:00 close) BEFORE the afternoon plays
    out. Gated-out coins are logged FLAT for transparency.

Schedule it once a day at ~08:05 UTC (Task Scheduler / cron). Each run settles
yesterday and commits today.

    .venv\\Scripts\\python.exe intraday_tsm_forward.py            # run the daily step
    .venv\\Scripts\\python.exe intraday_tsm_forward.py --report   # show the live track record
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from consensus_backtest import fetch_ohlcv
from intraday_tsm_backtest import day_samples

DB_PATH = "tsm_forward.db"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
           "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT"]
SPLIT = 8           # UTC hour: morning = 00:00->08:00, afternoon = 08:00->day close
VOL_WINDOW = 60     # trailing days for the vol-gate threshold (no lookahead)
VOL_Q = 0.667       # top-tertile |morning move|
FEE_BPS = 10.0
FETCH_DAYS = 80     # enough for the 60d trailing window + settle backlog


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS forward_trades (
            decision_day TEXT, symbol TEXT, direction TEXT,
            morning_ret REAL, vol_thr REAL,
            entry_price REAL, entry_ts TEXT,
            status TEXT, exit_price REAL, exit_ts TEXT,
            gross_ret REAL, net_ret REAL,
            created_at TEXT, settled_at TEXT,
            PRIMARY KEY (decision_day, symbol))"""
    )
    conn.commit()


def _hour(ts_ms: int) -> int:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour


def _day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def day_prices(candles: List[List[float]], day: str, split_hour: int
               ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(open, split_close, day_close) for `day`. split_close = close of last bar
    before split_hour; day_close = close of the last bar (only if hour-23 bar present)."""
    bars = sorted((b for b in candles if _day(int(b[0])) == day), key=lambda b: b[0])
    if not bars:
        return None, None, None
    open_px = bars[0][1]
    pre = [b for b in bars if _hour(int(b[0])) < split_hour]
    split_px = pre[-1][4] if pre else None
    has_close = any(_hour(int(b[0])) == 23 for b in bars)
    close_px = bars[-1][4] if has_close else None
    return open_px, split_px, close_px


def trailing_threshold(candles: List[List[float]], today: str) -> Optional[float]:
    """Top-tertile |morning return| over the prior VOL_WINDOW complete days (no lookahead)."""
    hist = [abs(m) for (d, m, _a) in day_samples(candles, SPLIT) if d < today]
    if len(hist) < VOL_WINDOW:
        return None
    return float(np.quantile(hist[-VOL_WINDOW:], VOL_Q))


def decide(conn: sqlite3.Connection, symbol: str, candles: List[List[float]],
           today: str, now_iso: str) -> Optional[str]:
    """Log today's pre-committed decision for one symbol (idempotent)."""
    row = conn.execute("SELECT 1 FROM forward_trades WHERE decision_day=? AND symbol=?",
                       (today, symbol)).fetchone()
    if row:
        return None  # already logged today
    # Only commit once the morning window is COMPLETE: the bar at hour (SPLIT-1) must
    # have closed, so morning return + entry price match the backtest exactly. Running
    # before ~08:00 UTC would commit on a partial morning — refuse and wait.
    today_bars = [b for b in candles if _day(int(b[0])) == today]
    if not any(_hour(int(b[0])) == SPLIT - 1 for b in today_bars):
        return None  # too early: the 07:00 UTC bar hasn't closed yet
    open_px, split_px, _close = day_prices(candles, today, SPLIT)
    if not open_px or not split_px:
        return None  # not enough of today's session yet
    thr = trailing_threshold(candles, today)
    if thr is None:
        return None  # not enough history to set the gate honestly
    morning = split_px / open_px - 1.0
    if abs(morning) >= thr:
        direction = "LONG" if morning > 0 else "SHORT"
        status, entry = "PENDING", split_px
    else:
        direction, status, entry = "FLAT", "SETTLED", split_px  # gated out
    conn.execute(
        "INSERT INTO forward_trades (decision_day,symbol,direction,morning_ret,vol_thr,"
        "entry_price,entry_ts,status,exit_price,exit_ts,gross_ret,net_ret,created_at,settled_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (today, symbol, direction, morning, thr, entry, now_iso, status,
         None, None, (0.0 if direction == "FLAT" else None),
         (0.0 if direction == "FLAT" else None), now_iso,
         now_iso if direction == "FLAT" else None))
    conn.commit()
    return direction


def settle(conn: sqlite3.Connection, symbol: str, candles: List[List[float]],
           today: str, now_iso: str, fee: float) -> int:
    """Fill exit price + net return for matured PENDING rows of this symbol."""
    pend = conn.execute(
        "SELECT decision_day,direction,entry_price FROM forward_trades "
        "WHERE symbol=? AND status='PENDING' AND decision_day < ?", (symbol, today)
    ).fetchall()
    n = 0
    for day, direction, entry in pend:
        _o, _s, close_px = day_prices(candles, day, SPLIT)
        if not close_px or not entry:
            continue  # day not fully closed in the fetched window yet
        sign = 1.0 if direction == "LONG" else -1.0
        gross = sign * (close_px / entry - 1.0)
        net = gross - fee
        conn.execute(
            "UPDATE forward_trades SET status='SETTLED', exit_price=?, exit_ts=?, "
            "gross_ret=?, net_ret=?, settled_at=? WHERE decision_day=? AND symbol=?",
            (close_px, f"{day}T23:59:59+00:00", gross, net, now_iso, day, symbol))
        n += 1
    conn.commit()
    return n


def run_daily(db_path: str = DB_PATH) -> int:
    fee = FEE_BPS / 10000.0
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    conn = sqlite3.connect(db_path)
    ensure_db(conn)
    settled = decided = 0
    decisions: List[str] = []
    for sym in SYMBOLS:
        try:
            candles = fetch_ohlcv(sym, "1h", FETCH_DAYS)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym:<10} fetch failed ({str(exc)[:40]})")
            continue
        settled += settle(conn, sym, candles, today, now_iso, fee)
        d = decide(conn, sym, candles, today, now_iso)
        if d:
            decided += 1
            if d != "FLAT":
                decisions.append(f"{sym.split('/')[0]} {d}")
    conn.close()
    print(f"[{now_iso}] settled {settled} matured day(s); logged {decided} new decision(s) for {today}.")
    if decisions:
        print("  today's committed trades: " + ", ".join(decisions))
    elif decided:
        print("  today: all coins gated out (no high-vol move) — FLAT.")
    return 0


def report(db_path: str = DB_PATH) -> int:
    conn = sqlite3.connect(db_path)
    ensure_db(conn)
    rows = conn.execute(
        "SELECT decision_day,symbol,direction,status,net_ret FROM forward_trades "
        "ORDER BY decision_day, symbol").fetchall()
    conn.close()
    if not rows:
        print("no forward-test records yet — run the daily step first (schedule it at ~08:05 UTC).")
        return 0
    settled = [r for r in rows if r[3] == "SETTLED" and r[2] != "FLAT" and r[4] is not None]
    pending = [r for r in rows if r[3] == "PENDING"]
    flat = [r for r in rows if r[2] == "FLAT"]
    days = sorted({r[0] for r in rows})
    print(f"\n=== INTRADAY-TSM LIVE FORWARD TEST — split 08:00 UTC, both dirs, "
          f"trailing vol-gate, {FEE_BPS:g}bps ===")
    print(f"span {days[0]} -> {days[-1]} ({len(days)} day(s)) · "
          f"{len(settled)} settled trades · {len(pending)} pending · {len(flat)} gated-out\n")
    if settled:
        nets = np.array([r[4] for r in settled])
        print(f"net/trade {nets.mean()*100:+.3f}%   win {100*(nets>0).mean():.0f}%   "
              f"total {nets.sum()*100:+.1f}%   (n={len(nets)})")
        by: Dict[str, List[float]] = {}
        for r in settled:
            by.setdefault(r[1].split("/")[0], []).append(r[4])
        print("\nby coin:")
        for c in sorted(by, key=lambda k: -float(np.mean(by[k]))):
            a = np.array(by[c])
            print(f"  {c:<6} n={len(a):<3} win {100*(a>0).mean():>3.0f}%  net/trade {a.mean()*100:>+7.3f}%")
        print("\nThis is the only test that can't be gamed — real, out-of-sample, pre-committed.")
        print("Let it run a few weeks; if net/trade stays positive and broad, the edge is live-confirmed.")
    else:
        print("no settled trades yet — decisions are committed and waiting to mature at day-close.")
    if pending:
        print("\npending (committed, awaiting day-close):")
        for r in pending:
            print(f"  {r[0]}  {r[1].split('/')[0]:<6} {r[2]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Live shadow forward-test of the intraday-TSM edge")
    ap.add_argument("--report", action="store_true", help="print the live track record")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()
    return report(args.db) if args.report else run_daily(args.db)


if __name__ == "__main__":
    raise SystemExit(main())

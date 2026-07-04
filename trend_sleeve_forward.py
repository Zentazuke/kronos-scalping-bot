"""trend_sleeve_forward.py — SHADOW forward test of the analyst-regime sleeves.

Earned on 2026-07-04: regime_allocator_backtest (analyst regime CSV, 4.5yr, 7 coins)
passed the pre-registered bar —
    trend-rider   beat buy&hold OOS on 6/7 coins (median Sharpe 0.58 vs 0.27, DD 34% vs 77%)
    default-long  beat buy&hold OOS on 7/7 coins (median Sharpe 0.82)
— and consistency held in-sample too. Per house rules that buys a shadow forward
test, nothing more. NO ORDERS — a pre-committed daily log, like tsm_forward.

One brain: the position comes from the ANALYST API's regime read (the same
classifier the backtest used), never recomputed here. If the API is down the day
is logged MISSED — fail loud, never guess.

  * COMMIT (run ~00:10 UTC): for each coin, GET {ANALYST_API_BASE}/regime/{sym}?timeframe=1D
    -> label+confidence -> positions: trend-rider long iff label in {trend_up, breakout};
    default-long long UNLESS label in {trend_down, flash, flash_risk}. Recorded for TODAY.
  * SETTLE: any past committed day whose daily candle has closed gets that day's
    close-to-close return applied to each sleeve's committed position, charging one
    taker+slip leg whenever a sleeve's position changed from the prior day.
    Buy&hold is settled alongside as the benchmark.

Timing note: at 00:10 UTC the 1D bar the analyst sees is minutes old, so its regime
read ≈ yesterday's completed bar — matching the backtest's +1-bar causal shift.

    python trend_sleeve_forward.py           # settle matured days + commit today
    python trend_sleeve_forward.py --report  # scorecard: sleeves vs buy&hold

Schedule (cron, UTC):  10 0 * * *  -> daily step
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import urllib.request
from datetime import datetime, timezone

from consensus_backtest import fetch_ohlcv
from costs import SLIPPAGE_BPS, taker_bps

DB_PATH = "sleeve_forward.db"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
           "DOGE/USDT", "LINK/USDT", "AVAX/USDT"]
ANALYST_API = os.getenv("ANALYST_API_BASE", "http://127.0.0.1:8000").rstrip("/")
CONFIRM = {"trend_up", "breakout"}
DAMAGE = {"trend_down", "flash", "flash_risk"}
FLIP_FEE = (taker_bps() + SLIPPAGE_BPS) / 1e4      # one leg per position change


def ensure(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS sleeve_days (
        day TEXT, symbol TEXT, regime TEXT, confidence REAL,
        pos_tr INTEGER, pos_dl INTEGER,
        status TEXT, ret REAL, ret_tr REAL, ret_dl REAL, ret_bh REAL,
        created_at TEXT, settled_at TEXT,
        PRIMARY KEY (day, symbol))""")
    conn.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def analyst_regime(symbol: str) -> tuple[str, float] | None:
    """(label, confidence) from the analyst API, or None if unreachable."""
    sym = symbol.replace("/", "_")
    try:
        with urllib.request.urlopen(f"{ANALYST_API}/regime/{sym}?timeframe=1D",
                                    timeout=10) as resp:  # noqa: S310 — local API
            data = json.loads(resp.read().decode("utf-8"))
        cur = data.get("current") or {}
        if cur.get("label"):
            return str(cur["label"]), float(cur.get("confidence") or 0.0)
    except Exception:  # noqa: BLE001 — down/missing -> MISSED, never guessed
        pass
    return None


def commit_today(conn: sqlite3.Connection) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = _now_iso()
    committed = missed = 0
    for sym in SYMBOLS:
        if conn.execute("SELECT 1 FROM sleeve_days WHERE day=? AND symbol=?",
                        (today, sym)).fetchone():
            continue  # already committed (idempotent)
        read = analyst_regime(sym)
        if read is None:
            conn.execute("INSERT INTO sleeve_days VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (today, sym, "MISSED", None, None, None, "MISSED",
                          None, None, None, None, now, now))
            missed += 1
            continue
        label, conf = read
        pos_tr = 1 if label in CONFIRM else 0
        pos_dl = 0 if label in DAMAGE else 1
        conn.execute("INSERT INTO sleeve_days VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (today, sym, label, conf, pos_tr, pos_dl, "PENDING",
                      None, None, None, None, now, None))
        committed += 1
    conn.commit()
    print(f"[{now}] commit {today}: {committed} committed, {missed} MISSED"
          + (" — IS THE ANALYST API RUNNING? (fail-loud by design)" if missed else ""))


def settle(conn: sqlite3.Connection) -> None:
    """Fill returns for matured PENDING days using daily candles."""
    now = _now_iso()
    pend = conn.execute("SELECT DISTINCT symbol FROM sleeve_days WHERE status='PENDING'").fetchall()
    total = 0
    for (sym,) in pend:
        try:
            candles = fetch_ohlcv(sym, "1d", 40)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: fetch failed ({str(exc)[:50]})"); continue
        close_by_day = {datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc)
                        .strftime("%Y-%m-%d"): float(c[4]) for c in candles}
        days_sorted = sorted(close_by_day)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT day, pos_tr, pos_dl FROM sleeve_days "
            "WHERE symbol=? AND status='PENDING' AND day<? ORDER BY day", (sym, today)).fetchall()
        for day, pos_tr, pos_dl in rows:
            if day not in close_by_day:
                continue
            i = days_sorted.index(day)
            if i == 0:
                continue
            ret = close_by_day[day] / close_by_day[days_sorted[i - 1]] - 1.0
            prev = conn.execute(
                "SELECT pos_tr, pos_dl FROM sleeve_days WHERE symbol=? AND day<? "
                "AND status='SETTLED' ORDER BY day DESC LIMIT 1", (sym, day)).fetchone()
            prev_tr, prev_dl = (prev if prev else (0, 1))   # sleeves start flat/long
            fee_tr = FLIP_FEE if pos_tr != prev_tr else 0.0
            fee_dl = FLIP_FEE if pos_dl != prev_dl else 0.0
            ret_tr = pos_tr * ret - fee_tr
            ret_dl = pos_dl * ret - fee_dl
            conn.execute(
                "UPDATE sleeve_days SET status='SETTLED', ret=?, ret_tr=?, ret_dl=?, "
                "ret_bh=?, settled_at=? WHERE day=? AND symbol=?",
                (ret, ret_tr, ret_dl, ret, now, day, sym))
            total += 1
    conn.commit()
    print(f"[{now}] settled {total} day-rows.")


def report(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT day, symbol, regime, ret_tr, ret_dl, ret_bh FROM sleeve_days "
        "WHERE status='SETTLED' ORDER BY day").fetchall()
    missed = conn.execute("SELECT COUNT(*) FROM sleeve_days WHERE status='MISSED'").fetchone()[0]
    if not rows:
        print("no settled days yet — run the daily step (00:10 UTC) and wait for maturity.")
        if missed:
            print(f"({missed} MISSED rows — the analyst API was down at commit time)")
        return
    days = sorted({r[0] for r in rows})
    print(f"\n=== SLEEVE SHADOW FORWARD TEST — {days[0]} -> {days[-1]} "
          f"({len(days)} settled day(s), {missed} missed) ===")
    print(f"{'sleeve':<14}{'cum ret':>10}{'mean/day':>10}{'pos days':>10}")
    for name, idx in (("trend-rider", 3), ("default-long", 4), ("buy&hold", 5)):
        per_day: dict[str, list[float]] = {}
        for r in rows:
            if r[idx] is not None:
                per_day.setdefault(r[0], []).append(float(r[idx]))
        daily = [sum(v) / len(v) for _, v in sorted(per_day.items())]   # equal-weight basket
        cum = 1.0
        for d in daily:
            cum *= 1 + d
        pos = sum(1 for d in daily if d > 0)
        print(f"{name:<14}{(cum-1)*100:>+9.2f}%{(sum(daily)/len(daily))*100:>+9.3f}%"
              f"{pos:>6}/{len(daily)}")
    print("\nJudge over weeks vs buy&hold — same discipline as the TSM forward test.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Shadow forward test: analyst-regime sleeves")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    ensure(conn)
    if args.report:
        report(conn)
    else:
        settle(conn)
        commit_today(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

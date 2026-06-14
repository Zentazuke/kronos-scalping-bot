"""label_observations.py — offline labeler for the observation journal.

Every row in ``observations.db`` is a hypothetical trade the bot evaluated but
may never have taken (blocked by the position cap, vetoed, etc.). This script
replays each one's TP/SL bracket against **real mainnet candles** to stamp a
clean WIN / LOSS — clean because mainnet price paths carry none of the testnet's
phantom-fill noise. That is the whole point of the observation journal: 10x the
training rows *and* honest labels.

Run it periodically (daily, or just before a walk-forward):

    python label_observations.py                          # labels observations.db
    python label_observations.py --db observations.db --window 48

Then train on the clean, plentiful data:

    python learner.py walkforward --db observations.db --model xgb

Idempotent: only OPEN observations old enough to have resolved are labeled;
the rest are left for the next run. Never touches journal.db.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

logger = logging.getLogger("bot.labeler")

TP_MULT: float = float(os.getenv("TP_ATR_MULT", "2.5"))
SL_MULT: float = float(os.getenv("SL_ATR_MULT", "2.5"))

STATUS_OPEN = "OPEN"
STATUS_WIN = "WIN"
STATUS_LOSS = "LOSS"
STATUS_SCRATCH = "SCRATCH"


def _to_ms(ts: str) -> Optional[int]:
    """Journal timestamps are UTC ISO strings; return epoch ms."""
    if not ts:
        return None
    s = ts.replace(" ", "T")
    tail = s[10:]
    if not (s.endswith("Z") or "+" in tail or "-" in tail):
        s += "+00:00"
    s = s.replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(s).timestamp() * 1000)
    except ValueError:
        return None


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def simulate(
    direction: str,
    entry: float,
    atr: float,
    candles: List[List[float]],
    start_ms: int,
    window: int,
) -> Optional[Tuple[str, float, int]]:
    """Replay the 2.5x-ATR bracket against future candles.

    ``candles`` is ``[[ms, open, high, low, close], ...]`` ascending. Returns
    ``(status, exit_price, ts_close_ms)`` once TP or SL is touched, SCRATCH if
    the full window elapses untouched, or ``None`` if there aren't yet enough
    future candles to decide (leave the observation OPEN, retry next run).

    When a single candle straddles both legs we take the STOP — the pessimistic
    reading, matching the live monitor.
    """
    is_long = direction.endswith("LONG")
    tp = entry + TP_MULT * atr if is_long else entry - TP_MULT * atr
    sl = entry - SL_MULT * atr if is_long else entry + SL_MULT * atr
    future = [c for c in candles if c[0] >= start_ms][:window]
    for c in future:
        high, low = c[2], c[3]
        hit_tp = (high >= tp) if is_long else (low <= tp)
        hit_sl = (low <= sl) if is_long else (high >= sl)
        if hit_sl and hit_tp:
            return (STATUS_LOSS, sl, int(c[0]))
        if hit_tp:
            return (STATUS_WIN, tp, int(c[0]))
        if hit_sl:
            return (STATUS_LOSS, sl, int(c[0]))
    if len(future) >= window:
        return (STATUS_SCRATCH, future[-1][4], int(future[-1][0]))
    return None


def _pnl(direction: str, entry: float, exit_price: float) -> float:
    return exit_price - entry if direction.endswith("LONG") else entry - exit_price


def _fetch_mainnet_candles(symbol: str, since_ms: int) -> List[List[float]]:
    """Mainnet 5m candles from ``since_ms`` to now, paginated. Public, keyless."""
    import ccxt  # type: ignore[import-untyped]

    ex = ccxt.binance({"enableRateLimit": True})
    now_ms = int(time.time() * 1000)
    out: List[List[float]] = []
    cursor = since_ms
    while cursor < now_ms:
        batch = ex.fetch_ohlcv(symbol, "5m", since=cursor, limit=1000)
        if not batch:
            break
        out.extend(
            [int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])]
            for b in batch
        )
        if len(batch) < 1000:
            break
        cursor = int(batch[-1][0]) + 1
    return out


def label(db_path: str, window: int) -> Tuple[int, int, int, int]:
    """Label every resolvable OPEN observation. Returns
    (wins, losses, scratches, still_open)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, ts_open, symbol, direction, entry_price, atr "
        "FROM trades WHERE status = ? ORDER BY symbol, id",
        (STATUS_OPEN,),
    ).fetchall()
    if not rows:
        conn.close()
        return (0, 0, 0, 0)

    by_symbol: dict[str, list] = {}
    for r in rows:
        by_symbol.setdefault(r["symbol"], []).append(r)

    wins = losses = scratches = still_open = 0
    for symbol, obs in by_symbol.items():
        starts = [m for r in obs if (m := _to_ms(r["ts_open"])) is not None]
        if not starts:
            still_open += len(obs)
            continue
        candles = _fetch_mainnet_candles(symbol, min(starts) - 5 * 60_000)
        logger.info("%s: %d observations, %d mainnet candles fetched",
                    symbol, len(obs), len(candles))
        for r in obs:
            start_ms = _to_ms(r["ts_open"])
            entry = r["entry_price"]
            atr = r["atr"]
            if start_ms is None or entry in (None, "") or atr in (None, ""):
                still_open += 1
                continue
            res = simulate(r["direction"], float(entry), float(atr),
                           candles, start_ms, window)
            if res is None:
                still_open += 1
                continue
            status, exit_price, ts_close_ms = res
            pnl = _pnl(r["direction"], float(entry), exit_price)
            conn.execute(
                "UPDATE trades SET status=?, exit_price=?, pnl=?, ts_close=? "
                "WHERE id=?",
                (status, f"{exit_price:.8f}", f"{pnl:.8f}", _iso(ts_close_ms), r["id"]),
            )
            wins += status == STATUS_WIN
            losses += status == STATUS_LOSS
            scratches += status == STATUS_SCRATCH
    conn.commit()
    conn.close()
    return (wins, losses, scratches, still_open)


def main() -> int:
    parser = argparse.ArgumentParser(description="Label the observation journal")
    parser.add_argument("--db", default="observations.db")
    parser.add_argument(
        "--window", type=int, default=48,
        help="max 5m bars to wait for the bracket to resolve (default 48 = 4h)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    wins, losses, scratches, still_open = label(args.db, args.window)
    logger.info(
        "labeled: %d WIN · %d LOSS · %d SCRATCH · %d still OPEN (not yet resolved)",
        wins, losses, scratches, still_open,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

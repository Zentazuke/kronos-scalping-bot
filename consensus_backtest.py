"""consensus_backtest.py — replay the TA net-bias over months of real candles.

The observation journal only carries ~2-3 days of consensus data, far too short
to tell a real edge from a good week. This sidesteps the wait: it recomputes the
SAME 7-indicator TA board (MACD, Supertrend, Stochastic, CCI, Bollinger,
Donchian, OBV) over historical mainnet candles and asks the reactive question
across many weeks and regimes — *when the board strongly agrees, and you trade
its direction, do you win net of fees?*

No Kronos / GPU needed. Over months of 5m bars Kronos is impractical (and that's
the point of the original backtest skipping it), so here the trade **direction is
the consensus's own sign** — long when the board points up, short when down,
strength = |consensus|. That isolates the net-bias signal and tests it directly.

Honest like the offline labeler: a 2.5x/2.5x ATR bracket replayed against forward
candles, a bar that straddles both legs counts as the STOP, timeout = scratch at
the last close. Trades overlap in time (one per qualifying bar), scored per-setup
like the observation journal — this measures signal quality, not a single-position
equity curve.

    python consensus_backtest.py --days 150 --min-consensus 5 --fee-bps 10
    python consensus_backtest.py --days 150 --min-consensus 9 --stride 1 --symbols "XRP/USDT,SOL/USDT"
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from ta_signals import compute_signals

logger = logging.getLogger("bot.consensus_bt")

DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
                   "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT")
LOOKBACK = 150            # bars of context handed to the TA board (>= MIN_BARS 60)
TP_MULT = 2.5             # bracket geometry the edge was measured at (R:R 1.0)
SL_MULT = 2.5
ATR_PERIOD = 14
_VOTE = {"long": 1, "short": -1, "neutral": 0}
_BOARD = {"MACD", "Supertrend", "Stochastic", "CCI", "Bollinger", "Donchian", "OBV"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_ohlcv(symbol: str, timeframe: str, days: int) -> List[List[float]]:
    """Mainnet OHLCV [[ts,o,h,l,c,v],...] for the last `days`, paginated. Keyless."""
    import ccxt  # type: ignore[import-untyped]
    import time

    ex = ccxt.binance({"enableRateLimit": True})
    now = int(time.time() * 1000)
    since = now - days * 86_400_000
    out: List[List[float]] = []
    cursor = since
    while cursor < now:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        out.extend([float(b[0]), float(b[1]), float(b[2]),
                    float(b[3]), float(b[4]), float(b[5])] for b in batch)
        if len(batch) < 1000:
            break
        cursor = int(batch[-1][0]) + 1
    return out


def atr_series(candles: Sequence[Sequence[float]], period: int = ATR_PERIOD) -> List[Optional[float]]:
    """Wilder ATR aligned to bars (None during warm-up)."""
    n = len(candles)
    out: List[Optional[float]] = [None] * n
    if n < period + 1:
        return out
    trs = []
    for i in range(1, n):
        h, l, pc = candles[i][2], candles[i][3], candles[i - 1][4]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs[:period]) / period
    out[period] = atr
    for i in range(period + 1, n):
        atr = (atr * (period - 1) + trs[i - 1]) / period
        out[i] = atr
    return out


def consensus_of(res: Dict) -> float:
    return float(sum(_VOTE.get(s["dir"], 0) * s["strength"]
                     for s in res["signals"] if s["name"] in _BOARD))


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def replay(future: Sequence[Sequence[float]], is_long: bool, entry: float,
           atr: float, window: int) -> Optional[float]:
    """Return the per-trade NET-of-nothing return (gross) for a 2.5/2.5 bracket,
    or None if there aren't `window` forward bars yet. Pessimistic straddle."""
    tp = entry + TP_MULT * atr if is_long else entry - TP_MULT * atr
    sl = entry - SL_MULT * atr if is_long else entry + SL_MULT * atr
    seen = 0
    exit_price: Optional[float] = None
    for c in future[:window]:
        seen += 1
        hi, lo = c[2], c[3]
        hit_tp = (hi >= tp) if is_long else (lo <= tp)
        hit_sl = (lo <= sl) if is_long else (hi >= sl)
        if hit_sl and hit_tp:
            exit_price = sl
            break
        if hit_tp:
            exit_price = tp
            break
        if hit_sl:
            exit_price = sl
            break
    if exit_price is None:
        if seen < window:
            return None  # not resolvable yet
        exit_price = future[window - 1][4]  # scratch at last close
    return (exit_price - entry) / entry if is_long else (entry - exit_price) / entry


# row = (ts_ms, symbol, abs_consensus, gross_ret)
Row = Tuple[int, str, float, float]


def backtest_symbol(symbol: str, candles: List[List[float]], *, min_consensus: float,
                    stride: int, window: int) -> List[Row]:
    atr = atr_series(candles)
    out: List[Row] = []
    last = len(candles) - window - 1
    for i in range(LOOKBACK, last + 1, stride):
        a = atr[i]
        if a is None or a <= 0:
            continue
        res = compute_signals(candles[i - LOOKBACK + 1: i + 1])
        if res is None:
            continue
        cons = consensus_of(res)
        if abs(cons) < min_consensus:
            continue
        is_long = cons > 0
        ret = replay(candles[i + 1:], is_long, candles[i][4], a, window)
        if ret is None:
            continue
        out.append((int(candles[i][0]), symbol, abs(cons), ret))
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _perf(nets: Sequence[float]) -> Dict[str, float]:
    n = len(nets)
    if n == 0:
        return {"n": 0, "win": 0.0, "exp": 0.0, "total": 0.0}
    return {"n": n, "win": sum(1 for x in nets if x > 0) / n,
            "exp": sum(nets) / n, "total": sum(nets)}


def _line(label: str, p: Dict[str, float]) -> str:
    if p["n"] == 0:
        return f"  {label:<20} (no trades)"
    return (f"  {label:<20} n={int(p['n']):<6} win {p['win']*100:>4.0f}%  "
            f"net/trade {p['exp']*100:>+7.3f}%  total {p['total']*100:>+8.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest the TA net-bias on historical candles")
    ap.add_argument("--days", type=int, default=150, help="history window in days (default 150)")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--min-consensus", type=float, default=5.0)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=2, help="evaluate every Nth bar (cuts CPU)")
    ap.add_argument("--resolve-window", type=int, default=48, help="max 5m bars to resolve a bracket")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    fee = args.fee_bps / 10000.0
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    rows: List[Row] = []
    for sym in symbols:
        try:
            candles = fetch_ohlcv(sym, args.timeframe, args.days)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: fetch failed (%s) — skipped", sym, exc)
            continue
        if len(candles) < LOOKBACK + args.resolve_window + 5:
            logger.warning("%s: only %d bars — skipped", sym, len(candles))
            continue
        sym_rows = backtest_symbol(sym, candles, min_consensus=args.min_consensus,
                                   stride=args.stride, window=args.resolve_window)
        logger.info("%s: %d bars -> %d gated trades", sym, len(candles), len(sym_rows))
        rows.extend(sym_rows)

    if not rows:
        logger.info("no gated trades produced — widen --days or lower --min-consensus")
        return 1

    nets = [r[3] - fee for r in rows]
    overall = _perf(nets)
    span_days = (max(r[0] for r in rows) - min(r[0] for r in rows)) / 86_400_000

    print(f"\n=== CONSENSUS BACKTEST — |consensus| >= {args.min_consensus:g}, "
          f"{args.fee_bps:.0f} bps fee, {args.timeframe} ===")
    print(f"{overall['n']} gated trades across {len(symbols)} symbols over ~{span_days:.0f} days "
          f"(2.5/2.5 ATR bracket, dir = board sign)\n")
    print("OVERALL (net of fees):")
    print(_line("gated", overall))

    # strength staircase
    print("\nBY |consensus| STRENGTH (net of fees):")
    edges = [(args.min_consensus, 7), (7, 10), (10, 13), (13, 1e9)]
    for lo, hi in edges:
        b = [r[3] - fee for r in rows if lo <= r[2] < hi]
        if b:
            hi_lbl = "+" if hi > 1e8 else f"{hi:g}"
            print(_line(f"[{lo:g}, {hi_lbl})", _perf(b)))

    # per-symbol
    print("\nBY SYMBOL (net of fees):")
    by_sym: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_sym[r[1]].append(r[3] - fee)
    for s in sorted(by_sym, key=lambda k: -_perf(by_sym[k])["total"]):
        print(_line(s, _perf(by_sym[s])))

    # per-month — the regime-persistence test
    print("\nBY MONTH (regime persistence — the whole point):")
    by_month: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        ym = datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc).strftime("%Y-%m")
        by_month[ym].append(r[3] - fee)
    pos_months = 0
    for ym in sorted(by_month):
        p = _perf(by_month[ym])
        pos_months += 1 if p["exp"] > 0 else 0
        print(_line(ym, p))

    # verdict
    n_months = len(by_month)
    print("\n=== read ===")
    pos_syms = sum(1 for s in by_sym if _perf(by_sym[s])["exp"] > 0)
    if overall["exp"] > 0 and pos_months >= max(2, n_months * 0.6) and pos_syms >= len(by_sym) * 0.6:
        print(f"net-positive overall (+{overall['exp']*100:.3f}%/trade), in {pos_months}/{n_months} "
              f"months and {pos_syms}/{len(by_sym)} symbols — BROAD and PERSISTENT. "
              f"This is a real signal; worth wiring as a soft size-tilt.")
    elif overall["exp"] > 0:
        print(f"net-positive overall (+{overall['exp']*100:.3f}%/trade) but NOT broad/persistent "
              f"({pos_months}/{n_months} months, {pos_syms}/{len(by_sym)} symbols positive) — "
              f"concentrated or regime-dependent. Promising but not deployable as a general rule.")
    else:
        print(f"NOT net-positive overall ({overall['exp']*100:+.3f}%/trade) — over months of real "
              f"candles the net-bias edge does not survive fees. The 2-3 day result was a mirage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

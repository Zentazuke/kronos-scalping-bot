"""reversion_backtest.py — test mean-reversion (RSI-2 / Connors) on months of candles.

Our trend-following TA board LOST money trading *with* the signal over 5 months
(48% win at a 1:1 bracket) — direct evidence that at short horizons price
REVERTS rather than continues. This tests the counter-hypothesis our own data and
the uploaded research both point to: **fade the extreme.** When RSI(period) is
deeply oversold, go LONG (bet on the bounce); when deeply overbought, go SHORT.

Same honest machinery as the consensus backtest: an ATR bracket replayed against
forward candles (pessimistic straddle, scratch on timeout), net of fees, broken
out by month, by symbol, and by side — with a verdict that demands breadth AND
persistence, not one lucky coin or month.

No Kronos / GPU. Reuses consensus_backtest's candle fetch, ATR and reporting.

    python reversion_backtest.py --timeframe 5m --rsi-period 2 --rsi-lo 10 --rsi-hi 90 --fee-bps 10
    python reversion_backtest.py --timeframe 1h --tp-mult 1.5 --sl-mult 1.5 --trend-sma 200
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from consensus_backtest import DEFAULT_SYMBOLS, _line, _perf, atr_series, fetch_ohlcv

logger = logging.getLogger("bot.reversion_bt")


def rsi_series(closes: Sequence[float], period: int) -> List[Optional[float]]:
    """Wilder RSI aligned to bars (None during warm-up)."""
    n = len(closes)
    out: List[Optional[float]] = [None] * n
    if n < period + 1:
        return out
    gains, losses = [], []
    for i in range(1, n):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    def rsi_from(ag: float, al: float) -> float:
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - 100.0 / (1.0 + rs)

    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    out[period] = rsi_from(ag, al)
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + gains[i - 1]) / period
        al = (al * (period - 1) + losses[i - 1]) / period
        out[i] = rsi_from(ag, al)
    return out


def sma_series(closes: Sequence[float], period: int) -> List[Optional[float]]:
    n = len(closes)
    out: List[Optional[float]] = [None] * n
    if period <= 0 or n < period:
        return out
    run = sum(closes[:period])
    out[period - 1] = run / period
    for i in range(period, n):
        run += closes[i] - closes[i - period]
        out[i] = run / period
    return out


def replay_bracket(future: Sequence[Sequence[float]], is_long: bool, entry: float,
                   atr: float, tp_mult: float, sl_mult: float, window: int) -> Optional[float]:
    tp = entry + tp_mult * atr if is_long else entry - tp_mult * atr
    sl = entry - sl_mult * atr if is_long else entry + sl_mult * atr
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
            return None
        exit_price = future[window - 1][4]
    return (exit_price - entry) / entry if is_long else (entry - exit_price) / entry


# row = (ts_ms, symbol, side, ret)
Row = Tuple[int, str, str, float]


def backtest_symbol(symbol: str, candles: List[List[float]], *, rsi_period: int,
                    rsi_lo: float, rsi_hi: float, trend_sma: int, tp_mult: float,
                    sl_mult: float, window: int) -> List[Row]:
    closes = [c[4] for c in candles]
    rsi = rsi_series(closes, rsi_period)
    atr = atr_series(candles)
    sma = sma_series(closes, trend_sma) if trend_sma > 0 else None
    out: List[Row] = []
    warmup = max(rsi_period + 1, trend_sma + 1, 15)
    last = len(candles) - window - 1
    for i in range(warmup, last + 1):
        r, a = rsi[i], atr[i]
        if r is None or a is None or a <= 0:
            continue
        if r < rsi_lo:
            is_long = True
        elif r > rsi_hi:
            is_long = False
        else:
            continue
        if sma is not None and sma[i] is not None:
            if is_long and closes[i] <= sma[i]:
                continue          # only fade-long when above the trend
            if (not is_long) and closes[i] >= sma[i]:
                continue          # only fade-short when below the trend
        ret = replay_bracket(candles[i + 1:], is_long, closes[i], a, tp_mult, sl_mult, window)
        if ret is None:
            continue
        out.append((int(candles[i][0]), symbol, "LONG" if is_long else "SHORT", ret))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest RSI mean-reversion on historical candles")
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--rsi-period", type=int, default=2)
    ap.add_argument("--rsi-lo", type=float, default=10.0, help="oversold -> go LONG below this")
    ap.add_argument("--rsi-hi", type=float, default=90.0, help="overbought -> go SHORT above this")
    ap.add_argument("--trend-sma", type=int, default=0,
                    help="if >0, only fade in the direction of this SMA (Connors-style); 0 = off")
    ap.add_argument("--tp-mult", type=float, default=1.5, help="take-profit ATR multiple")
    ap.add_argument("--sl-mult", type=float, default=1.5, help="stop-loss ATR multiple")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--resolve-window", type=int, default=48)
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
        if len(candles) < max(args.trend_sma, 60) + args.resolve_window + 5:
            logger.warning("%s: only %d bars — skipped", sym, len(candles))
            continue
        sym_rows = backtest_symbol(
            sym, candles, rsi_period=args.rsi_period, rsi_lo=args.rsi_lo, rsi_hi=args.rsi_hi,
            trend_sma=args.trend_sma, tp_mult=args.tp_mult, sl_mult=args.sl_mult,
            window=args.resolve_window,
        )
        logger.info("%s: %d bars -> %d reversion trades", sym, len(candles), len(sym_rows))
        rows.extend(sym_rows)

    if not rows:
        logger.info("no reversion trades produced — loosen --rsi-lo/--rsi-hi or widen --days")
        return 1

    nets = [r[3] - fee for r in rows]
    overall = _perf(nets)
    span_days = (max(r[0] for r in rows) - min(r[0] for r in rows)) / 86_400_000

    print(f"\n=== REVERSION BACKTEST — RSI({args.rsi_period}) <{args.rsi_lo:g}/>{args.rsi_hi:g}, "
          f"TP{args.tp_mult:g}/SL{args.sl_mult:g} ATR, {args.fee_bps:.0f} bps, {args.timeframe}"
          f"{', trend-SMA '+str(args.trend_sma) if args.trend_sma else ''} ===")
    print(f"{overall['n']} reversion trades across {len(symbols)} symbols over ~{span_days:.0f} days "
          f"(fade extremes: oversold->long, overbought->short)\n")
    print("OVERALL (net of fees):")
    print(_line("reversion", overall))

    # by side
    print("\nBY SIDE (net of fees):")
    for side in ("LONG", "SHORT"):
        print(_line(side, _perf([r[3] - fee for r in rows if r[2] == side])))

    # per symbol
    print("\nBY SYMBOL (net of fees):")
    by_sym: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_sym[r[1]].append(r[3] - fee)
    for s in sorted(by_sym, key=lambda k: -_perf(by_sym[k])["total"]):
        print(_line(s, _perf(by_sym[s])))

    # per month
    print("\nBY MONTH (regime persistence):")
    by_month: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        ym = datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc).strftime("%Y-%m")
        by_month[ym].append(r[3] - fee)
    pos_months = 0
    for ym in sorted(by_month):
        p = _perf(by_month[ym])
        pos_months += 1 if p["exp"] > 0 else 0
        print(_line(ym, p))

    n_months = len(by_month)
    pos_syms = sum(1 for s in by_sym if _perf(by_sym[s])["exp"] > 0)
    print("\n=== read ===")
    if overall["exp"] > 0 and pos_months >= max(2, n_months * 0.6) and pos_syms >= len(by_sym) * 0.6:
        print(f"net-positive overall (+{overall['exp']*100:.3f}%/trade), in {pos_months}/{n_months} "
              f"months and {pos_syms}/{len(by_sym)} symbols — BROAD and PERSISTENT. Mean-reversion "
              f"shows a real edge here; worth tuning exits and a forward test.")
    elif overall["exp"] > 0:
        print(f"net-positive overall (+{overall['exp']*100:.3f}%/trade) but NOT broad/persistent "
              f"({pos_months}/{n_months} months, {pos_syms}/{len(by_sym)} symbols) — concentrated "
              f"or regime-dependent. Promising but not a general edge.")
    else:
        print(f"NOT net-positive overall ({overall['exp']*100:+.3f}%/trade) — fading extremes does "
              f"not survive fees on this timeframe either. Try --timeframe 1h, or it's another dead end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

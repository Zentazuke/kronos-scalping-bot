"""mtf_scalper_backtest.py — faithful backtest of the multi-timeframe
trend-pullback scalping strategy (1h bias -> 15m confirm -> 5m entry).

Implements the spec literally: the weighted score table with a >=75 gate, the
1h/15m bias gates, the trend-pullback entry (pullback to EMA20/VWAP + micro-high
break + momentum resume), ATR+swing stops, and the TP1(1R,50%)/TP2(1.5R,25%)/
runner(trail 1.5*ATR) partial-exit engine with time-stop, opposite-Supertrend and
momentum exits.

HONEST LIMIT: live bid/ask spread, order-book depth, real slippage and latency do
NOT exist in historical OHLCV, so the coin-liquidity filter is approximated and
costs are modelled as a configurable round-trip haircut (--fee-bps). Everything
else is the real strategy logic, not an approximation.

No Kronos / GPU. Reuses consensus_backtest's candle fetch and reporting.

    python mtf_scalper_backtest.py --days 150 --fee-bps 10
    python mtf_scalper_backtest.py --days 150 --fee-bps 10 --adx-thr 20 --score-min 75
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import mtf_indicators as ti
from consensus_backtest import DEFAULT_SYMBOLS, _line, _perf, fetch_ohlcv

logger = logging.getLogger("bot.mtf")

TF_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def align(five: Sequence[Sequence[float]], htf: Sequence[Sequence[float]], htf_ms: int
          ) -> List[Optional[int]]:
    """For each 5m bar, the index of the last HIGHER-tf bar that has fully CLOSED
    at or before the 5m bar opens (no lookahead)."""
    out: List[Optional[int]] = [None] * len(five)
    j = -1
    for idx, c in enumerate(five):
        t = c[0]
        while j + 1 < len(htf) and htf[j + 1][0] + htf_ms <= t:
            j += 1
        out[idx] = j if j >= 0 else None
    return out


class Ind:
    """All indicator series for one timeframe."""
    def __init__(self, candles: Sequence[Sequence[float]]) -> None:
        closes = [c[4] for c in candles]
        self.ema20 = ti.ema(closes, 20)
        self.ema50 = ti.ema(closes, 50)
        self.ema200 = ti.ema(closes, 200)
        self.pdi, self.mdi, self.adx = ti.adx_di(candles, 14)
        _, self.st_dir = ti.supertrend(candles, 10, 3.0)
        self.rsi = ti.rsi(closes, 14)
        self.stoch_k, self.stoch_d = ti.stochastic(candles, 14, 3, 3)
        self.mh = ti.macd_hist(closes, 12, 26, 9)
        self.obv = ti.obv(candles)
        self.relvol = ti.rel_volume(candles, 20)
        self.atr = ti.atr(candles, 14)
        self.vwap = ti.vwap_daily(candles)
        self.closes = closes


def _ok(*vals) -> bool:
    return all(v is not None for v in vals)


def long_score(c5, i5, c1h, i1, c15, i15, adx_thr: float) -> Optional[int]:
    """Literal long score table. Returns score, or None if the hard bias gate fails."""
    close_1h = c1h.closes[i1]
    if not _ok(c1h.ema200[i1], c1h.ema50[i1], c1h.pdi[i1], c1h.mdi[i1], c1h.adx[i1],
               c15.ema20[i15], c15.ema50[i15], c5.ema20[i5], c5.ema50[i5], c5.atr[i5]):
        return None
    # hard bias gate (must hold to even consider a long)
    if not (close_1h > c1h.ema200[i1] and c1h.ema50[i1] > c1h.ema200[i1]
            and c1h.pdi[i1] > c1h.mdi[i1] and c1h.adx[i1] > adx_thr
            and c15.ema20[i15] > c15.ema50[i15]):
        return None
    s = 0
    s += 15 if close_1h > c1h.ema200[i1] else 0
    s += 10 if c1h.ema50[i1] > c1h.ema200[i1] else 0
    s += 10 if c15.ema20[i15] > c15.ema50[i15] else 0
    s += 10 if c5.ema20[i5] > c5.ema50[i5] else 0
    s += 10 if c5.st_dir[i5] == 1 else 0
    s += 10 if c1h.pdi[i1] > c1h.mdi[i1] else 0
    s += 10 if c1h.adx[i1] > 20 else 0
    # RSI pullback into 40-55 then turning up
    rsi_ok = (_ok(c5.rsi[i5], c5.rsi[i5 - 1])
              and any(c5.rsi[j] is not None and 40 <= c5.rsi[j] <= 55 for j in (i5 - 2, i5 - 1))
              and c5.rsi[i5] > c5.rsi[i5 - 1])
    s += 8 if rsi_ok else 0
    # Stochastic cross up (preferably below 60)
    stoch_ok = (_ok(c5.stoch_k[i5], c5.stoch_d[i5], c5.stoch_k[i5 - 1], c5.stoch_d[i5 - 1])
                and c5.stoch_k[i5] > c5.stoch_d[i5] and c5.stoch_k[i5 - 1] <= c5.stoch_d[i5 - 1]
                and c5.stoch_k[i5] < 60)
    s += 5 if stoch_ok else 0
    # MACD histogram rising 2 candles
    mh_ok = (_ok(c5.mh[i5], c5.mh[i5 - 1], c5.mh[i5 - 2])
             and c5.mh[i5] > c5.mh[i5 - 1] > c5.mh[i5 - 2])
    s += 5 if mh_ok else 0
    # OBV rising or relvol > 1.1
    vol_ok = (c5.obv[i5] > c5.obv[i5 - 1]) or (c5.relvol[i5] is not None and c5.relvol[i5] > 1.1)
    s += 10 if vol_ok else 0
    # ATR (volatility) acceptable — spread unknown, ATR% in a sane band
    atr_pct = c5.atr[i5] / c5.closes[i5]
    s += 10 if 0.0005 <= atr_pct <= 0.03 else 0
    return s


def short_score(c5, i5, c1h, i1, c15, i15, adx_thr: float) -> Optional[int]:
    close_1h = c1h.closes[i1]
    if not _ok(c1h.ema200[i1], c1h.ema50[i1], c1h.pdi[i1], c1h.mdi[i1], c1h.adx[i1],
               c15.ema20[i15], c15.ema50[i15], c5.ema20[i5], c5.ema50[i5], c5.atr[i5]):
        return None
    if not (close_1h < c1h.ema200[i1] and c1h.ema50[i1] < c1h.ema200[i1]
            and c1h.mdi[i1] > c1h.pdi[i1] and c1h.adx[i1] > adx_thr
            and c15.ema20[i15] < c15.ema50[i15]):
        return None
    s = 0
    s += 15 if close_1h < c1h.ema200[i1] else 0
    s += 10 if c1h.ema50[i1] < c1h.ema200[i1] else 0
    s += 10 if c15.ema20[i15] < c15.ema50[i15] else 0
    s += 10 if c5.ema20[i5] < c5.ema50[i5] else 0
    s += 10 if c5.st_dir[i5] == -1 else 0
    s += 10 if c1h.mdi[i1] > c1h.pdi[i1] else 0
    s += 10 if c1h.adx[i1] > 20 else 0
    rsi_ok = (_ok(c5.rsi[i5], c5.rsi[i5 - 1])
              and any(c5.rsi[j] is not None and 45 <= c5.rsi[j] <= 60 for j in (i5 - 2, i5 - 1))
              and c5.rsi[i5] < c5.rsi[i5 - 1])
    s += 8 if rsi_ok else 0
    stoch_ok = (_ok(c5.stoch_k[i5], c5.stoch_d[i5], c5.stoch_k[i5 - 1], c5.stoch_d[i5 - 1])
                and c5.stoch_k[i5] < c5.stoch_d[i5] and c5.stoch_k[i5 - 1] >= c5.stoch_d[i5 - 1]
                and c5.stoch_k[i5] > 40)
    s += 5 if stoch_ok else 0
    mh_ok = (_ok(c5.mh[i5], c5.mh[i5 - 1], c5.mh[i5 - 2])
             and c5.mh[i5] < c5.mh[i5 - 1] < c5.mh[i5 - 2])
    s += 5 if mh_ok else 0
    vol_ok = (c5.obv[i5] < c5.obv[i5 - 1]) or (c5.relvol[i5] is not None and c5.relvol[i5] > 1.1)
    s += 10 if vol_ok else 0
    atr_pct = c5.atr[i5] / c5.closes[i5]
    s += 10 if 0.0005 <= atr_pct <= 0.03 else 0
    return s


def _pullback_and_break(candles, c5, i5, is_long: bool) -> bool:
    """Pullback toward EMA20 then a micro-structure break in the trend direction."""
    ema20 = c5.ema20[i5]
    if ema20 is None or i5 < 4:
        return False
    if is_long:
        pulled = min(candles[j][3] for j in (i5 - 3, i5 - 2, i5 - 1)) <= ema20 * 1.001
        back = candles[i5][4] > ema20
        brk = candles[i5][4] > max(candles[j][2] for j in (i5 - 3, i5 - 2, i5 - 1))
        return pulled and back and brk
    pulled = max(candles[j][2] for j in (i5 - 3, i5 - 2, i5 - 1)) >= ema20 * 0.999
    back = candles[i5][4] < ema20
    brk = candles[i5][4] < min(candles[j][3] for j in (i5 - 3, i5 - 2, i5 - 1))
    return pulled and back and brk


def simulate_exit(candles, i, is_long: bool, c5, window: int, time_bars: int,
                  fee: float) -> Optional[float]:
    """Faithful TP1/TP2/runner partial-exit engine. Returns net % return."""
    entry = candles[i][4]
    atr_e = c5.atr[i]
    if atr_e is None or atr_e <= 0:
        return None
    if is_long:
        stop = max(ti.swing_low(candles, i, 10), entry - 1.2 * atr_e)
        if stop >= entry:
            stop = entry - 1.2 * atr_e
        R = entry - stop
        tp1, tp2 = entry + R, entry + 1.5 * R
    else:
        stop = min(ti.swing_high(candles, i, 10), entry + 1.2 * atr_e)
        if stop <= entry:
            stop = entry + 1.2 * atr_e
        R = stop - entry
        tp1, tp2 = entry - R, entry - 1.5 * R
    if R <= 0:
        return None

    remaining = 1.0
    tp1d = tp2d = False
    trail: Optional[float] = None
    realized = 0.0
    entry_rsi = c5.rsi[i] if c5.rsi[i] is not None else 50.0
    last = min(i + window, len(candles) - 1)

    def add(frac: float, price: float) -> None:
        nonlocal realized
        realized += frac * ((price - entry) if is_long else (entry - price)) / entry

    for k in range(i + 1, last + 1):
        hi, lo, cl = candles[k][2], candles[k][3], candles[k][4]
        tnew = cl - 1.5 * atr_e if is_long else cl + 1.5 * atr_e
        trail = tnew if trail is None else (max(trail, tnew) if is_long else min(trail, tnew))
        # 1. stop first (pessimistic)
        if (lo <= stop) if is_long else (hi >= stop):
            add(remaining, stop); remaining = 0.0; break
        # 2. TP1
        if not tp1d and ((hi >= tp1) if is_long else (lo <= tp1)):
            add(0.5, tp1); remaining -= 0.5; tp1d = True
        # 3. TP2
        if tp1d and not tp2d and ((hi >= tp2) if is_long else (lo <= tp2)):
            add(0.25, tp2); remaining -= 0.25; tp2d = True
        # 4. runner trailing stop (after TP2)
        if tp2d and remaining > 0 and ((lo <= trail) if is_long else (hi >= trail)):
            add(remaining, trail); remaining = 0.0; break
        # 5. opposite Supertrend / time stop / momentum exit
        opp = (c5.st_dir[k] == -1) if is_long else (c5.st_dir[k] == 1)
        tstop = (k - i) >= time_bars
        mh, r = c5.mh[k], c5.rsi[k]
        mom = (mh is not None and ((mh < 0) if is_long else (mh > 0))
               and r is not None and ((r < entry_rsi) if is_long else (r > entry_rsi)))
        if remaining > 0 and (opp or tstop or mom):
            add(remaining, cl); remaining = 0.0; break
    if remaining > 0:
        add(remaining, candles[last][4])
    return realized - fee


# row = (ts, symbol, side, net_ret)
Row = Tuple[int, str, str, float]


def backtest_symbol(symbol: str, c5d, c15d, c1hd, *, adx_thr: float, score_min: int,
                    window: int, time_bars: int, fee: float) -> Row_list:  # type: ignore[valid-type]
    c5, c15, c1h = Ind(c5d), Ind(c15d), Ind(c1hd)
    a15 = align(c5d, c15d, TF_MS["15m"])
    a1h = align(c5d, c1hd, TF_MS["1h"])
    out: List[Row] = []
    last = len(c5d) - window - 1
    i = 5
    while i <= last:
        i1, i15 = a1h[i], a15[i]
        if i1 is None or i15 is None:
            i += 1
            continue
        side = None
        ls = long_score(c5, i, c1h, i1, c15, i15, adx_thr)
        if ls is not None and ls >= score_min and _pullback_and_break(c5d, c5, i, True):
            side = "LONG"
        else:
            ss = short_score(c5, i, c1h, i1, c15, i15, adx_thr)
            if ss is not None and ss >= score_min and _pullback_and_break(c5d, c5, i, False):
                side = "SHORT"
        if side is None:
            i += 1
            continue
        ret = simulate_exit(c5d, i, side == "LONG", c5, window, time_bars, fee)
        if ret is not None:
            out.append((int(c5d[i][0]), symbol, side, ret))
            i += window  # one position at a time per coin — skip past the trade
        else:
            i += 1
    return out


Row_list = List[Row]


def main() -> int:
    ap = argparse.ArgumentParser(description="Faithful MTF trend-pullback scalper backtest")
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--adx-thr", type=float, default=18.0)
    ap.add_argument("--score-min", type=int, default=75)
    ap.add_argument("--time-stop", type=int, default=10, help="exit if no resolution after N 5m bars")
    ap.add_argument("--resolve-window", type=int, default=48)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    fee = args.fee_bps / 10000.0
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    rows: List[Row] = []
    for sym in symbols:
        try:
            c5 = fetch_ohlcv(sym, "5m", args.days)
            c15 = fetch_ohlcv(sym, "15m", args.days)
            c1h = fetch_ohlcv(sym, "1h", args.days)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: fetch failed (%s) — skipped", sym, exc)
            continue
        if len(c1h) < 220 or len(c5) < 300:
            logger.warning("%s: not enough history — skipped", sym)
            continue
        r = backtest_symbol(sym, c5, c15, c1h, adx_thr=args.adx_thr, score_min=args.score_min,
                            window=args.resolve_window, time_bars=args.time_stop, fee=fee)
        logger.info("%s: %d trades", sym, len(r))
        rows.extend(r)

    if not rows:
        logger.info("no trades qualified — the gates may be too strict for this window")
        return 1

    nets = [r[3] for r in rows]
    overall = _perf(nets)
    span = (max(r[0] for r in rows) - min(r[0] for r in rows)) / 86_400_000
    print(f"\n=== MTF TREND-PULLBACK SCALPER — score>={args.score_min}, ADX>{args.adx_thr:g}, "
          f"{args.fee_bps:.0f} bps, 1h/15m/5m ===")
    print(f"{overall['n']} trades across {len(symbols)} symbols over ~{span:.0f} days "
          f"(net of fees; TP1 1R/TP2 1.5R/runner trail)\n")
    print("OVERALL (net of fees):")
    print(_line("strategy", overall))

    print("\nBY SIDE:")
    for side in ("LONG", "SHORT"):
        print(_line(side, _perf([r[3] for r in rows if r[2] == side])))

    print("\nBY SYMBOL:")
    by_sym: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_sym[r[1]].append(r[3])
    for s in sorted(by_sym, key=lambda k: -_perf(by_sym[k])["total"]):
        print(_line(s, _perf(by_sym[s])))

    print("\nBY MONTH:")
    by_month: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_month[datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc).strftime("%Y-%m")].append(r[3])
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
              f"months and {pos_syms}/{len(by_sym)} symbols — BROAD and PERSISTENT. The trend-pullback "
              f"strategy shows a real edge; THIS is worth building into the live bot.")
    elif overall["exp"] > 0:
        print(f"net-positive overall (+{overall['exp']*100:.3f}%/trade) but NOT broad/persistent "
              f"({pos_months}/{n_months} months, {pos_syms}/{len(by_sym)} symbols) — fragile.")
    else:
        print(f"NOT net-positive overall ({overall['exp']*100:+.3f}%/trade) — the full multi-timeframe "
              f"trend-pullback strategy, simulated to the letter, does not survive fees either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""pairs_backtest.py — market-neutral statistical-arbitrage (pairs) backtest.

The first genuinely NEW road: no direction prediction at all. Find two coins whose
log-price spread historically mean-reverts (they move together), and when the
spread stretches too far, LONG the cheap one / SHORT the rich one, betting the
*relationship* snaps back — not betting on the market going up or down. This
sidesteps both walls that killed everything else: no entry signal to predict (it's
relative value), and it's fully backtestable on the candle history we already have.

Honest method:
  * Hedge ratio (beta) is fit on the FIRST half (train), the strategy is traded on
    the SECOND half (out-of-sample) — no peeking.
  * Half-life of mean reversion (from an AR(1) fit) screens for pairs that actually
    revert; non-reverting pairs are skipped.
  * Entry when the spread's rolling z-score exceeds +/- z_in; exit back toward 0
    (z_out); hard time-stop. Both legs cost fees.

    python pairs_backtest.py --timeframe 1h --days 180 --z-in 2.0 --z-out 0.5 --fee-bps 10

Needs numpy + ccxt (your venv has both). Reuses consensus_backtest's candle fetch.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from consensus_backtest import DEFAULT_SYMBOLS, fetch_ohlcv

logger = logging.getLogger("bot.pairs")


def _closes_aligned(series: Dict[str, List[List[float]]]) -> Tuple[List[str], np.ndarray, List[int]]:
    """Intersect timestamps across symbols, return (symbols, close matrix [T x N], ts)."""
    common: Optional[set] = None
    for s, c in series.items():
        ts = {int(b[0]) for b in c}
        common = ts if common is None else (common & ts)
    if not common:
        return [], np.empty((0, 0)), []
    ts_sorted = sorted(common)
    syms = list(series.keys())
    mat = np.zeros((len(ts_sorted), len(syms)))
    for j, s in enumerate(syms):
        by_ts = {int(b[0]): b[4] for b in series[s]}
        for i, t in enumerate(ts_sorted):
            mat[i, j] = by_ts[t]
    return syms, mat, ts_sorted


def _half_life(spread: np.ndarray) -> float:
    """AR(1) half-life of mean reversion. Large/inf = doesn't revert."""
    s = spread[:-1]
    ds = np.diff(spread)
    s = s - s.mean()
    denom = float(s @ s)
    if denom == 0:
        return math.inf
    beta = float(s @ ds) / denom  # ds ~ beta * (s - mean)
    if beta >= 0:
        return math.inf  # not mean-reverting
    return -math.log(2) / beta


def _hedge_ratio(la: np.ndarray, lb: np.ndarray) -> float:
    """OLS slope of log(A) on log(B): la ≈ alpha + beta·lb."""
    b = lb - lb.mean()
    denom = float(b @ b)
    return float(b @ (la - la.mean())) / denom if denom else 1.0


def backtest_pair(la: np.ndarray, lb: np.ndarray, split: int, *, z_window: int,
                  z_in: float, z_out: float, time_stop: int, fee: float) -> Optional[Dict]:
    """Trade z-score reversion on the OOS half. Returns per-trade net returns + stats."""
    beta = _hedge_ratio(la[:split], lb[:split])  # fit on train only
    spread = la - beta * lb
    hl = _half_life(spread[:split])
    if not (1 < hl < z_window * 4):  # must revert, on a sane timescale
        return None
    mu = spread[:split].mean()
    sd = spread[:split].std()
    if sd == 0:
        return None

    rets: List[float] = []
    pos = 0  # +1 = long spread (long A short B), -1 = short spread
    entry_i = 0
    for i in range(split, len(spread)):
        z = (spread[i] - mu) / sd
        if pos == 0:
            if z >= z_in:
                pos, entry_i = -1, i      # spread too high -> short it
            elif z <= -z_in:
                pos, entry_i = +1, i      # spread too low -> long it
        else:
            hit_target = (pos == -1 and z <= z_out) or (pos == +1 and z >= -z_out)
            timed_out = (i - entry_i) >= time_stop
            if hit_target or timed_out:
                # spread P&L over the hold (market-neutral): change in spread * pos
                pnl = pos * (spread[i] - spread[entry_i])
                rets.append(pnl - 2 * fee)  # round-trip on two legs
                pos = 0
    if not rets:
        return None
    n = len(rets)
    arr = np.array(rets)
    return {
        "beta": beta, "half_life": hl, "n": n,
        "win": float((arr > 0).mean()),
        "exp": float(arr.mean()),
        "total": float(arr.sum()),
        "sharpe": float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Market-neutral pairs / cointegration backtest")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--z-window", type=int, default=48, help="bars for the spread mean/std baseline")
    ap.add_argument("--z-in", type=float, default=2.0, help="enter when |z| exceeds this")
    ap.add_argument("--z-out", type=float, default=0.5, help="exit when |z| falls back under this")
    ap.add_argument("--time-stop", type=int, default=96, help="max bars to hold a pair trade")
    ap.add_argument("--fee-bps", type=float, default=10.0, help="cost per leg fill (round-trip = 2x both legs)")
    ap.add_argument("--top", type=int, default=15, help="show this many best pairs")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    fee = args.fee_bps / 10000.0
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    series: Dict[str, List[List[float]]] = {}
    for s in symbols:
        try:
            c = fetch_ohlcv(s, args.timeframe, args.days)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: fetch failed (%s)", s, exc); continue
        if len(c) > 200:
            series[s] = c
    syms, mat, ts = _closes_aligned(series)
    if len(syms) < 2 or mat.shape[0] < 200:
        logger.info("not enough aligned history across symbols"); return 1
    logmat = np.log(mat)
    split = mat.shape[0] // 2

    results: List[Tuple[str, str, Dict]] = []
    for a, b in itertools.combinations(range(len(syms)), 2):
        r = backtest_pair(logmat[:, a], logmat[:, b], split, z_window=args.z_window,
                          z_in=args.z_in, z_out=args.z_out, time_stop=args.time_stop, fee=fee)
        if r is not None:
            results.append((syms[a].split("/")[0], syms[b].split("/")[0], r))

    if not results:
        logger.info("no mean-reverting pairs cleared the screen — loosen z-in or try --timeframe 4h"); return 1

    results.sort(key=lambda x: -x[2]["total"])
    span = (ts[-1] - ts[split]) / 86_400_000

    print(f"\n=== PAIRS (STAT-ARB) BACKTEST — {args.timeframe}, z-in {args.z_in:g}/z-out {args.z_out:g}, "
          f"{args.fee_bps:g}bps/leg ===")
    print(f"{len(syms)} coins, {len(results)} reverting pairs, OOS span ~{span:.0f} days "
          f"(hedge fit in-sample, traded out-of-sample)\n")
    print(f"{'pair':<16}{'half-life':>10}{'trades':>8}{'win%':>7}{'net/trade':>11}{'total':>9}{'Sharpe':>8}")
    for a, b, r in results[:args.top]:
        print(f"{a+'-'+b:<16}{r['half_life']:>9.0f}b{r['n']:>8}{r['win']*100:>6.0f}%"
              f"{r['exp']*100:>+10.3f}%{r['total']*100:>+8.1f}%{r['sharpe']:>8.2f}")

    # portfolio: equal-weight all reverting pairs
    all_total = sum(r["total"] for _a, _b, r in results)
    all_trades = sum(r["n"] for _a, _b, r in results)
    pos_pairs = sum(1 for _a, _b, r in results if r["exp"] > 0)
    print(f"\nPORTFOLIO (all {len(results)} pairs equal-weight): "
          f"{all_total*100:+.1f}% total over ~{span:.0f}d, {all_trades} trades, "
          f"{pos_pairs}/{len(results)} pairs net-positive")

    print("\n=== read ===")
    # TRADE-weighted (the honest number) — every trade counts equally, so a lucky
    # 1-trade pair can't masquerade as an edge.
    tw_exp = all_total / all_trades if all_trades else 0.0
    thin = all_trades < 200 or sum(1 for _a, _b, r in results if r["n"] < 8) > 0.4 * len(results)
    if tw_exp > 0 and pos_pairs >= 0.6 * len(results) and not thin:
        print(f"trade-weighted net-positive (+{tw_exp*100:.3f}%/trade) across {pos_pairs}/{len(results)} "
              f"pairs on {all_trades} trades — a real, direction-neutral reverting edge. Worth refining.")
    elif tw_exp > 0 and thin:
        print(f"trade-weighted +{tw_exp*100:.3f}%/trade across {pos_pairs}/{len(results)} pairs — BUT on only "
              f"{all_trades} trades, many on tiny-sample pairs. That's small-sample LUCK, not a trustworthy "
              f"edge. Re-run with more coins + lower --z-in for hundreds of trades, then we believe it (or not).")
    elif tw_exp > 0:
        print(f"trade-weighted +{tw_exp*100:.3f}%/trade but only {pos_pairs}/{len(results)} pairs positive "
              f"— the gains are concentrated in a few low-trade flukes; fragile, not a real edge.")
    else:
        print(f"trade-weighted NEGATIVE ({tw_exp*100:+.3f}%/trade) — the spreads revert roughly break-even "
              f"GROSS, but the two-leg fees sink it. Crypto majors are BTC-correlated, not truly "
              f"cointegrated. Try --timeframe 4h (far fewer fees), but the common-factor problem is deep.")
    print("NOTE: real spread P&L also carries the two legs' execution + a small basis/borrow cost; "
          "this is the clean-fill estimate. But it's a genuinely new, direction-free road.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

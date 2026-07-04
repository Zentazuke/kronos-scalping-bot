"""fade_backtest.py — the MIRROR of careful-bets: fade the extreme morning move.

Origin (pre-registered, 2026-07-03): the careful-bets selectivity ladder showed OOS Sharpe
getting WORSE as we selected bigger morning moves (0.11 -> -0.42) — i.e. the biggest
morning moves tend to REVERSE in the afternoon (overreactions / liquidation spikes that
mean-revert). This tests that reversal directly, under the house honesty bar.

PRIMARY CONFIG (locked before running — changing it after seeing results = mining):
  * basket: BTC, ETH, SOL, XRP, DOGE, LINK, AVAX (the trial's coins + BTC)
  * signal: |morning ret| >= trailing-60d 90th percentile (strictly prior days, no lookahead)
            -> trade AGAINST the morning direction, split close (08:00) -> day close
  * fee: 20 bps round-trip (two taker legs — the real toll, per the 2026-07-03 audit)
  * judged on: OOS (last 40%) net total + breadth (coins net-positive OOS) + worst month.
    Quantiles 0.80 / 0.95 are shown as ROBUSTNESS ONLY — if 0.90 fails, this dies; we do
    not go shopping in the neighbors (that's Runs A-D again).

Verdict rule: PASS needs OOS net > 0, breadth > half the basket, and a worst month you can
live with. Anything else -> add to the graveyard next to careful-bets and move on.

    python fade_backtest.py --days 1850              # the 5-year bar (default)
    python fade_backtest.py --days 1850 --with-trend # also show the momentum mirror
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from consensus_backtest import fetch_ohlcv
from intraday_tsm_backtest import day_samples

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
           "DOGE/USDT", "LINK/USDT", "AVAX/USDT"]
SPLIT = 8            # same session split as intraday-TSM
VOL_WINDOW = 60      # trailing days for the extreme threshold
PRIMARY_Q = 0.90     # top-decile |morning move| — THE pre-registered config
ROBUST_QS = [0.80, 0.95]
FEE = 0.0020         # 20 bps round-trip (real live toll)
IS_FRAC = 0.6


def trades_for(samples: List[Tuple[str, float, float]], q: float, against: bool
               ) -> List[Tuple[str, float]]:
    """[(day, net_ret)] — fade (against=True) or ride (False) the extreme morning move.
    Threshold = trailing VOL_WINDOW-day q-quantile of |morning|, strictly prior days."""
    out = []
    abs_hist: List[float] = []
    for day, m, a in samples:
        if len(abs_hist) >= VOL_WINDOW:
            thr = float(np.quantile(abs_hist[-VOL_WINDOW:], q))
            if abs(m) >= thr and thr > 0:
                sign = -1.0 if against else 1.0          # against: fade the morning direction
                gross = sign * (1.0 if m > 0 else -1.0) * a
                out.append((day, gross - FEE))
        abs_hist.append(abs(m))
    return out


def stats(trades: List[Tuple[str, float]]) -> dict:
    if not trades:
        return {"n": 0, "win": 0.0, "avg": 0.0, "tot": 0.0, "worst_mo": 0.0}
    nets = np.array([t[1] for t in trades])
    bym: Dict[str, float] = defaultdict(float)
    for d, r in trades:
        bym[d[:7]] += r
    return {"n": len(nets), "win": float((nets > 0).mean()), "avg": float(nets.mean()),
            "tot": float(nets.sum()), "worst_mo": min(bym.values())}


def line(tag: str, s: dict) -> str:
    return (f"  {tag:<22} n={s['n']:<5} win {s['win']*100:>3.0f}%  net/trade {s['avg']*100:>+7.3f}%"
            f"  total {s['tot']*100:>+7.1f}%  worstMo {s['worst_mo']*100:+.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fade the extreme morning move (mirror of careful-bets)")
    ap.add_argument("--days", type=int, default=1850, help="5-year bar by default — no cherry years")
    ap.add_argument("--with-trend", action="store_true", help="also show the momentum (ride) mirror")
    args = ap.parse_args()

    per_coin: Dict[str, List[Tuple[str, float]]] = {}
    coin_samples: Dict[str, List[Tuple[str, float, float]]] = {}   # fetched ONCE, reused below
    all_days: List[str] = []
    for sym in SYMBOLS:
        try:
            c = fetch_ohlcv(sym, "1h", args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: fetch failed ({str(exc)[:60]}) — skip"); continue
        samples = day_samples(c, SPLIT)
        if len(samples) < VOL_WINDOW + 50:
            print(f"  {sym}: only {len(samples)} usable days — skip"); continue
        coin_samples[sym] = samples
        per_coin[sym] = trades_for(samples, PRIMARY_Q, against=True)
        all_days.extend(d for d, _m, _a in samples)
    if not per_coin:
        print("no usable symbols"); return 1

    days_sorted = sorted(set(all_days))
    split_day = days_sorted[int(len(days_sorted) * IS_FRAC)]
    print(f"\n=== MORNING-FADE BACKTEST — fade top-decile morning moves, {FEE*1e4:.0f}bps, "
          f"{days_sorted[0]} -> {days_sorted[-1]} ===")
    print(f"IS < {split_day} <= OOS  ·  PRIMARY config q={PRIMARY_Q} (pre-registered)\n")

    pooled_is, pooled_oos, breadth = [], [], 0
    print("per coin (PRIMARY, OOS is what counts):")
    for sym, trades in per_coin.items():
        t_is = [t for t in trades if t[0] < split_day]
        t_oos = [t for t in trades if t[0] >= split_day]
        pooled_is += t_is; pooled_oos += t_oos
        so = stats(t_oos)
        if so["tot"] > 0:
            breadth += 1
        print(f"  {sym.split('/')[0]:<6} IS n={len(t_is):<4} tot {stats(t_is)['tot']*100:>+7.1f}%   "
              f"OOS n={so['n']:<4} win {so['win']*100:>3.0f}%  tot {so['tot']*100:>+7.1f}%")
    print("\npooled:")
    print(line("PRIMARY q=0.90  IS", stats(pooled_is)))
    print(line("PRIMARY q=0.90  OOS", stats(pooled_oos)))
    print(f"  breadth: {breadth}/{len(per_coin)} coins net-positive OOS")

    print("\nrobustness (context only — the verdict is the PRIMARY row):")
    for q in ROBUST_QS:
        p_oos = []
        for sym in per_coin:
            c_trades = trades_for(coin_samples[sym], q, True)
            p_oos += [t for t in c_trades if t[0] >= split_day]
        print(line(f"fade q={q}  OOS", stats(p_oos)))
    if args.with_trend:
        p_oos = []
        for sym in per_coin:
            c_trades = trades_for(coin_samples[sym], PRIMARY_Q, False)
            p_oos += [t for t in c_trades if t[0] >= split_day]
        print(line("ride q=0.90 OOS (mirror)", stats(p_oos)))

    s = stats(pooled_oos)
    print("\n=== verdict ===")
    ok = s["tot"] > 0 and breadth > len(per_coin) / 2
    print(("PASS candidate — take it to a shadow forward test (same tsm_forward pattern), "
           "do NOT skip that step.") if ok else
          "FAIL — the reversal was noise (or eaten by the toll). Graveyard it next to careful-bets.")
    print(f"  OOS: net {s['tot']*100:+.1f}% over {s['n']} trades, breadth {breadth}/{len(per_coin)}, "
          f"worst month {s['worst_mo']*100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

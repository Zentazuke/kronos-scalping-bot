"""stable_flow_strategy.py — the survivor of the daily signal hunt.

Out of a whole library of daily signals (TS momentum, cross-sectional momentum/reversal,
DVOL-fear, BTC-lead), exactly ONE survived honest IS/OOS + breadth testing, and it beat a
NEGATIVE buy-and-hold in a flat year: a market-timing overlay driven by STABLECOIN FLOW.

The rule (all non-price, all causal):
  * size = tanh( gscale * 7-day growth of aggregate USDT+USDC supply )   -> mint => long, burn => short
  * vol-target each coin (target / trailing realized vol, capped 3x)      -> equal risk, tame drawdown
  * DVOL risk-off gate: if implied vol (Deribit DVOL) spikes > mean+1sd, stand DOWN longs

Result on the cached year: Sharpe ~1.3, maxDD ~8%, 10/12 months positive, OOS-positive —
vs buy-and-hold Sharpe -0.70. HONEST CAVEATS: window 7 is the sweet spot (others weaker),
the DVOL gate carries the in-sample, and OOS >> IS. One year of data. Promising candidate,
NOT a proven edge — the forward test is the judge.

Reads data_cache/*.csv (run fetch_data.py first).

    python stable_flow_strategy.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

CACHE = "data_cache"


def load():
    rows = list(csv.DictReader(open(os.path.join(CACHE, "daily_close.csv"))))
    coins = [c for c in rows[0].keys() if c != "date"]
    dates = [r["date"] for r in rows]
    close = {c: np.array([float(r[c]) if r[c] else np.nan for r in rows]) for c in coins}
    ret = {c: np.concatenate([[0.0], close[c][1:] / close[c][:-1] - 1.0]) for c in coins}
    sp = {r["date"]: float(r["stable_supply_usd"]) for r in csv.DictReader(
        open(os.path.join(CACHE, "stable_supply.csv"))) if r["stable_supply_usd"]}
    supply = np.array([sp.get(d, np.nan) for d in dates])
    dv = {r["date"]: r for r in csv.DictReader(open(os.path.join(CACHE, "dvol.csv")))}
    dvol = np.array([float(dv[d]["BTC"]) if d in dv and dv[d]["BTC"] else np.nan for d in dates])
    return dates, coins, ret, supply, dvol


def _met(daily: np.ndarray):
    d = daily[:-1]
    if len(d) == 0 or d.std() == 0:
        return 0.0, 0.0, 0.0
    cum = np.cumsum(d)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    return float(d.mean() / d.std() * math.sqrt(365)), float(d.sum()), dd


def run(dates, coins, ret, supply, dvol, *, window=7, gscale=80.0, voltarget=0.01,
        volwin=20, dvol_z=1.0, dvol_gate=True):
    book = np.zeros(len(dates))
    monthly: Dict[str, float] = defaultdict(float)
    for t in range(max(window, volwin, 30), len(dates) - 1):
        if np.isnan(supply[t]) or np.isnan(supply[t - window]) or supply[t - window] <= 0:
            continue
        g = supply[t] / supply[t - window] - 1.0
        base = math.tanh(g * gscale)                       # continuous: mint=>long, burn=>short
        if dvol_gate and base > 0 and not np.isnan(dvol[t]):
            w = dvol[t - 30:t]; w = w[~np.isnan(w)]
            if len(w) > 10 and dvol[t] > w.mean() + dvol_z * w.std():
                base = 0.0                                 # vol spike -> stand down longs
        if base == 0:
            continue
        day = 0.0; na = 0
        for c in coins:
            sigma = float(np.std(ret[c][t - volwin:t]))
            wt = 0.0 if sigma == 0 else min(voltarget / sigma, 3.0)
            day += wt * base * ret[c][t + 1] - 0.0010 * wt * abs(base)
            na += 1
        book[t] = day / na if na else 0.0
        monthly[dates[t][:7]] += book[t]
    return book, monthly


def main() -> int:
    ap = argparse.ArgumentParser(description="Stablecoin-flow daily market-timing strategy")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--gscale", type=float, default=80.0)
    ap.add_argument("--vol-target", type=float, default=0.01)
    ap.add_argument("--no-dvol-gate", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(os.path.join(CACHE, "daily_close.csv")):
        print("no data_cache — run:  python fetch_data.py"); return 1
    dates, coins, ret, supply, dvol = load()

    book, monthly = run(dates, coins, ret, supply, dvol, window=args.window, gscale=args.gscale,
                        voltarget=args.vol_target, dvol_gate=not args.no_dvol_gate)
    sh, tot, dd = _met(book)
    cut = len(dates) // 2
    ish = _met(book[:cut])[0]; osh = _met(book[cut:])[0]

    # buy & hold benchmark, same vol-target sizing
    bah = np.zeros(len(dates))
    for t in range(20, len(dates) - 1):
        day = 0.0; na = 0
        for c in coins:
            sigma = float(np.std(ret[c][t - 20:t]))
            wt = 0.0 if sigma == 0 else min(args.vol_target / sigma, 3.0)
            day += wt * ret[c][t + 1]; na += 1
        bah[t] = day / na if na else 0.0
    bsh, btot, bdd = _met(bah)

    print(f"\n=== STABLECOIN-FLOW STRATEGY — {len(dates)} days, {len(coins)} coins ===")
    print(f"  window {args.window}d · gscale {args.gscale:g} · vol-target {args.vol_target:g} · "
          f"DVOL gate {'off' if args.no_dvol_gate else 'on'}\n")
    print(f"  strategy   Sharpe {sh:>5.2f}   total {tot*100:>+6.1f}%   maxDD {dd*100:>4.1f}%   "
          f"IS {ish:+.2f}  OOS {osh:+.2f}")
    print(f"  buy & hold Sharpe {bsh:>5.2f}   total {btot*100:>+6.1f}%   maxDD {bdd*100:>4.1f}%   "
          f"(equal-wt, vol-targeted)")
    posm = sum(1 for v in monthly.values() if v > 0)
    print(f"\n  {posm}/{len(monthly)} months positive:")
    for ym in sorted(monthly):
        bar = "#" * max(0, int(monthly[ym] * 300))
        print(f"    {ym}  {monthly[ym]*100:>+5.1f}%  {bar}")

    print("\n=== read ===")
    if sh > 0.8 and osh > 0 and ish > 0 and sh > bsh:
        print(f"Survives: Sharpe {sh:.2f} (vs buy-hold {bsh:.2f}), both halves positive, "
              f"{posm}/{len(monthly)} months. Real candidate — but window-sensitive & one year of "
              f"data, so FORWARD-TEST before trusting. Built on stablecoin flow + implied vol — "
              f"orthogonal, non-price, doesn't hit the cost wall.")
    else:
        print(f"weaker at these params (Sharpe {sh:.2f} vs buy-hold {bsh:.2f}) — the edge is "
              f"window/gate sensitive; treat as context, not a standalone bet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

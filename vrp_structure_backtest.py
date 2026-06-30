"""vrp_structure_backtest.py — the TRADEABLE variance-risk-premium test.

The proxy backtest (vol_premium_backtest.py) measured the premium with a frictionless
variance swap (~1.8 Sharpe). This models the thing you'd actually put on: a monthly
short IRON CONDOR on BTC — sell a strangle, buy wings to CAP the loss — priced with
Black-Scholes off the real DVOL history, with the real capped payoff and the real left
tail. It answers the honest question: after defined-risk caps + fees, what's left?

Each cycle (every --dte days, non-overlapping):
  * entry: BTC = S0, implied vol = DVOL. Expected move sigma = IV*sqrt(T).
  * sell call & put at +/- k_short sigma ; buy wings at +/- k_long sigma.
  * collect the net credit (BS-priced at IV), minus fees.
  * at expiry (S_T from price history): pay the condor's intrinsic loss (capped by wings).
  * return = net P&L / capital-at-risk (the max loss you post).

Reports net Sharpe, win%, worst cycle, max drawdown, IS vs OOS — and a "fair-value"
benchmark (same structure priced at REALIZED vol, i.e. no premium) to prove the edge IS
the premium and not the structure. Reads data_cache/{daily_close,dvol}.csv (fetch_data.py).

    python vrp_structure_backtest.py --dte 30 --k-short 1.0 --k-long 2.0 --fee-bps 3
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

CACHE = "data_cache"


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs(S: float, K: float, T: float, iv: float, call: bool) -> float:
    """Black-Scholes price, r=0 (crypto). Falls back to intrinsic at T<=0/iv<=0."""
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if call else (K - S))
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    return (S * _ncdf(d1) - K * _ncdf(d2)) if call else (K * _ncdf(-d2) - S * _ncdf(-d1))


def load() -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    closes: Dict[str, float] = {}
    for r in csv.DictReader(open(os.path.join(CACHE, "daily_close.csv"))):
        if r.get("BTC"):
            closes[r["date"]] = float(r["BTC"])
    dvol: Dict[str, float] = {}
    for r in csv.DictReader(open(os.path.join(CACHE, "dvol.csv"))):
        if r.get("BTC"):
            dvol[r["date"]] = float(r["BTC"])
    dates = sorted(set(closes) & set(dvol))
    return dates, closes, dvol


def condor_pnl(S0, ST, iv_price, T, k_short, k_long, fee_bps):
    """Net P&L (in $ per 1 unit of BTC notional) and capital-at-risk, for a short condor
    whose premium is priced at iv_price and whose payoff is settled at ST."""
    sig = iv_price * math.sqrt(T)
    Kc_s, Kp_s = S0 * (1 + k_short * sig), S0 * (1 - k_short * sig)
    Kc_l, Kp_l = S0 * (1 + k_long * sig), S0 * (1 - k_long * sig)
    legs = [(Kc_s, True, +1), (Kp_s, False, +1), (Kc_l, True, -1), (Kp_l, False, -1)]
    prem = sum(sgn * bs(S0, K, T, iv_price, call) for K, call, sgn in legs)   # net credit (>0)
    # entry fee: Deribit ~0.0003*S0 per option, capped at 12.5% of the leg price
    fee = 0.0
    for K, call, _sgn in legs:
        p = bs(S0, K, T, iv_price, call)
        fee += min(0.0003 * S0, 0.125 * p) if p > 0 else 0.0003 * S0
    fee += fee_bps / 1e4 * S0 * 0  # (placeholder; per-leg model above is the cost)
    call_loss = max(0.0, ST - Kc_s) - max(0.0, ST - Kc_l)     # capped call-spread loss
    put_loss = max(0.0, Kp_s - ST) - max(0.0, Kp_l - ST)      # capped put-spread loss
    net = prem - call_loss - put_loss - fee
    car = max(Kc_l - Kc_s, Kp_s - Kp_l) - prem                # capital at risk (max loss)
    return net, max(car, 1e-9), prem


def realized_vol(dates, closes, i, dte):
    """Annualized realized vol over the cycle [i, i+dte] (for the fair-value benchmark)."""
    seg = [closes[dates[j]] for j in range(i, min(i + dte, len(dates)))]
    if len(seg) < 3:
        return None
    r = np.diff(np.log(seg))
    return float(r.std() * math.sqrt(365)) if r.std() > 0 else 0.0


def run(dates, closes, dvol, *, dte, k_short, k_long, fee_bps, price_at="implied"):
    T = dte / 365.0
    cyc: List[Tuple[str, float, float]] = []   # (day, return_on_risk, net$)
    i = 0
    while i + dte < len(dates):
        d0, dT = dates[i], dates[i + dte]
        S0, ST = closes[d0], closes[dT]
        iv = dvol[d0] / 100.0
        ivp = iv if price_at == "implied" else (realized_vol(dates, closes, i, dte) or iv)
        net, car, _prem = condor_pnl(S0, ST, ivp, T, k_short, k_long, fee_bps)
        cyc.append((d0, net / car, net))
        i += dte
    return cyc


def metrics(rets: np.ndarray, dte: int) -> dict:
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0.0, "mean": float(rets.mean()) if len(rets) else 0.0,
                "total": float(rets.sum()) if len(rets) else 0.0, "maxdd": 0.0, "worst": 0.0, "win": 0.0}
    cyc_yr = 365.0 / dte
    cum = np.cumsum(rets)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    return {"sharpe": float(rets.mean() / rets.std() * math.sqrt(cyc_yr)),
            "mean": float(rets.mean()), "total": float(rets.sum()), "maxdd": dd,
            "worst": float(rets.min()), "win": float((rets > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description="Tradeable VRP — monthly defined-risk short condor on BTC")
    ap.add_argument("--dte", type=int, default=30, help="days to expiry / cycle length")
    ap.add_argument("--k-short", type=float, default=1.0, help="short strikes at k*sigma")
    ap.add_argument("--k-long", type=float, default=2.0, help="wing strikes at k*sigma (cap)")
    ap.add_argument("--fee-bps", type=float, default=3.0)
    args = ap.parse_args()
    if not os.path.exists(os.path.join(CACHE, "dvol.csv")):
        print("no data_cache/dvol.csv — run:  python fetch_data.py --days 1000"); return 1
    dates, closes, dvol = load()
    if len(dates) < 4 * args.dte:
        print(f"only {len(dates)} days of BTC+DVOL data — need more"); return 1

    cyc = run(dates, closes, dvol, dte=args.dte, k_short=args.k_short, k_long=args.k_long, fee_bps=args.fee_bps)
    rets = np.array([r for _d, r, _n in cyc])
    nets = [n for _d, _r, n in cyc]
    m = metrics(rets, args.dte)
    cut = len(rets) // 2
    is_m, oos_m = metrics(rets[:cut], args.dte), metrics(rets[cut:], args.dte)
    # fair-value benchmark: same structure priced at REALIZED vol (no premium) -> should be ~0
    fair = run(dates, closes, dvol, dte=args.dte, k_short=args.k_short, k_long=args.k_long,
               fee_bps=args.fee_bps, price_at="realized")
    fm = metrics(np.array([r for _d, r, _n in fair]), args.dte)
    by_month: Dict[str, float] = defaultdict(float)
    for d, r, _n in cyc:
        by_month[d[:7]] += r

    print(f"\n=== TRADEABLE VRP — short BTC iron condor, {args.dte}d cycles, "
          f"shorts {args.k_short}sigma / wings {args.k_long}sigma, {len(rets)} cycles ===")
    print(f"  spans {dates[0]} -> {dates[-1]}\n")
    print(f"  net Sharpe   {m['sharpe']:>6.2f}   (annualized, on capital-at-risk)")
    print(f"  win rate     {m['win']*100:>5.1f}%   of cycles green")
    print(f"  mean / cycle {m['mean']*100:>+5.1f}%  of capital-at-risk")
    print(f"  total        {m['total']*100:>+5.0f}%  cumulative (on risk)")
    print(f"  worst cycle  {m['worst']*100:>+5.0f}%  (the capped tail — one bad expiry)")
    print(f"  max drawdown {m['maxdd']*100:>5.0f}%")
    print(f"  IS Sharpe {is_m['sharpe']:+.2f}   OOS Sharpe {oos_m['sharpe']:+.2f}   "
          f"(both positive = stable, the key test)")
    print(f"  fair-value benchmark (priced at REALIZED vol, no premium): Sharpe {fm['sharpe']:+.2f}, "
          f"mean {fm['mean']*100:+.1f}%")
    pos = sum(1 for v in by_month.values() if v > 0)
    print(f"  {pos}/{len(by_month)} months net-positive\n")

    print("=== read ===")
    real = m['sharpe'] > 0.5 and is_m['sharpe'] > 0 and oos_m['sharpe'] > 0 and m['mean'] > 0
    edge_is_premium = m['mean'] > fm['mean'] + 1e-6
    if real and edge_is_premium:
        print(f"TRADEABLE EDGE HOLDS: net Sharpe {m['sharpe']:.2f} after the defined-risk cap + fees, "
              f"positive in BOTH halves (IS {is_m['sharpe']:+.2f}, OOS {oos_m['sharpe']:+.2f}). And selling at "
              f"IMPLIED beats selling at REALIZED (mean {m['mean']*100:+.1f}% vs {fm['mean']*100:+.1f}%) — the "
              f"edge IS the premium, not the structure. This is the real, harvestable number (thinner than the "
              f"~1.8 proxy because the cap + fees are now in). Next: paper-trade this exact structure on Deribit.")
    elif real:
        print(f"positive (Sharpe {m['sharpe']:.2f}) but the premium isn't cleanly isolated vs the realized-vol "
              f"benchmark — sweep --k-short / --dte and re-check before trusting it.")
    else:
        print(f"the defined-risk version does NOT clear the bar (Sharpe {m['sharpe']:.2f}, IS {is_m['sharpe']:+.2f}, "
              f"OOS {oos_m['sharpe']:+.2f}). The cap + fees ate the proxy edge. Sweep the structure, but the "
              f"frictionless ~1.8 was optimistic — this is the honest tradeable picture.")
    print("\nCaveats: flat IV (no skew — real puts are richer, so put-side credit is understated); cash-settled "
          "European-style held to expiry; no early management. A real paper-trade on Deribit is the next judge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

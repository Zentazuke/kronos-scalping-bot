"""vol_premium_backtest.py — harvest the volatility risk premium (be the insurer).

The most different road we've tried: not prediction, not fast, not directional. Option
buyers systematically OVERPAY for protection — implied volatility trades above the
volatility that actually shows up. Selling that insurance harvests the gap. It's one
of the most robust premia in all of finance, and crucially it does NOT hit the cost
wall that killed every fast idea, because the edge is a structural premium, not a
micro-inefficiency.

Honest model (a daily-marked short variance swap — the clean way to measure the VRP):
  * implied daily variance = (DVOL_t / 100)^2 / 365         <- what you SELL (Deribit DVOL)
  * realized daily variance = r_{t+1}^2                      <- what you PAY (next day's move^2)
  * daily P&L (short vol) ∝ implied_var - realized_var
You earn the implied variance every day and pay the realized squared return. Positive
on average = the premium is real; but it has a FAT LEFT TAIL — when vol spikes, the
realized move dwarfs the premium and you take a big hit. This shows both.

Free data, no keys: DVOL from Deribit's public API, price from Binance.

    python vol_premium_backtest.py --days 365 --currency BTC
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from consensus_backtest import fetch_ohlcv


def _day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def fetch_dvol(currency: str, days: int) -> Dict[str, float]:
    """Daily DVOL (30-day implied vol, annualized %) by UTC date — Deribit public API."""
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    url = (f"https://www.deribit.com/api/v2/public/get_volatility_index_data"
           f"?currency={currency}&start_timestamp={start}&end_timestamp={end}&resolution=1D")
    req = urllib.request.Request(url, headers={"User-Agent": "kronos-vrp/1.0"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        data = json.load(resp)
    out: Dict[str, float] = {}
    for row in data.get("result", {}).get("data", []):
        ts, _o, _h, _l, close = row[0], row[1], row[2], row[3], row[4]
        out[_day(int(ts))] = float(close)        # IV in annualized %
    return out


def price_returns(symbol: str, days: int) -> Dict[str, float]:
    """Daily close-to-close return by UTC date (Binance)."""
    c = fetch_ohlcv(symbol, "1d", days)
    out: Dict[str, float] = {}
    for i in range(1, len(c)):
        if c[i - 1][4] > 0:
            out[_day(int(c[i][0]))] = c[i][4] / c[i - 1][4] - 1.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Volatility risk premium backtest (short vol)")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--currency", default="BTC", help="BTC or ETH (Deribit DVOL)")
    ap.add_argument("--notional", type=float, default=10000.0, help="vega notional for the $ curve")
    args = ap.parse_args()

    print(f"fetching {args.currency} DVOL (Deribit) + price (Binance) ...")
    try:
        iv = fetch_dvol(args.currency, args.days)
    except Exception as exc:  # noqa: BLE001
        print(f"DVOL fetch failed ({str(exc)[:80]}).")
        print("Deribit public API may be unreachable from your network; retry, or check the URL.")
        return 1
    rets = price_returns(f"{args.currency}/USDT", args.days)
    days = sorted(set(iv) & set(rets))
    if len(days) < 60:
        print(f"only {len(days)} aligned days — try a larger --days"); return 1

    # daily short-variance P&L (in variance units), then scaled to $ on a vega notional
    pnl: List[Tuple[str, float, float, float]] = []   # (day, iv, rv_ann, daily_pnl_$)
    for d in days:
        implied_daily_var = (iv[d] / 100.0) ** 2 / 365.0
        realized_daily_var = rets[d] ** 2
        # $ P&L: short variance, scaled so 1 "vol point" of edge ~ notional * vol * dvol-ish.
        # Use vega-style scaling: notional * (implied_var - realized_var) / (2 * implied_vol_daily)
        iv_daily = (iv[d] / 100.0) / np.sqrt(365.0)
        scale = args.notional / (2.0 * iv_daily) if iv_daily > 0 else 0.0
        daily = scale * (implied_daily_var - realized_daily_var)
        rv_ann = abs(rets[d]) * np.sqrt(365.0) * 100.0
        pnl.append((d, iv[d], rv_ann, daily))

    pnls = np.array([p[3] for p in pnl])
    ivs = np.array([p[1] for p in pnl])
    rv_real = np.array([p[2] for p in pnl])
    pos_days = float((pnls > 0).mean())
    equity = args.notional + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    maxdd = float((peak - equity).max())
    worst = float(pnls.min())
    cut = len(days) // 2
    is_mean, oos_mean = float(pnls[:cut].mean()), float(pnls[cut:].mean())

    print(f"\n=== {args.currency} VOLATILITY RISK PREMIUM — short vol, {len(days)} days, "
          f"${args.notional:,.0f} vega notional ===")
    print(f"  avg implied vol (DVOL)   {ivs.mean():.1f}%")
    print(f"  avg realized vol         {rv_real.mean():.1f}%   "
          f"(premium {ivs.mean()-rv_real.mean():+.1f} vol pts {'<-- IV > RV, premium exists' if ivs.mean()>rv_real.mean() else ''})")
    print(f"  short-vol P&L total      ${pnls.sum():+,.0f}   (on ${args.notional:,.0f})")
    print(f"  days premium positive    {pos_days*100:.0f}%")
    print(f"  WORST single day         ${worst:+,.0f}   <- the fat left tail (a vol spike)")
    print(f"  max drawdown             -${maxdd:,.0f}")
    print(f"  IS ${is_mean:+.1f}/day · OOS ${oos_mean:+.1f}/day")

    print("\nBY MONTH ($):")
    by_month: Dict[str, float] = defaultdict(float)
    for d, _iv, _rv, p in pnl:
        by_month[d[:7]] += p
    posm = 0
    for ym in sorted(by_month):
        v = by_month[ym]; posm += 1 if v > 0 else 0
        print(f"  {ym}   {v:>+10,.0f}")
    nmonths = len(by_month)

    print("\n=== read ===")
    mean_pos = pnls.mean() > 0
    if mean_pos and pos_days > 0.55 and oos_mean > 0 and posm >= 0.6 * nmonths:
        print(f"REAL PREMIUM: implied vol ran {ivs.mean()-rv_real.mean():+.1f} pts above realized; selling it "
              f"paid +${pnls.sum():,.0f} on ${args.notional:,.0f}, positive {pos_days*100:.0f}% of days and "
              f"{posm}/{nmonths} months, holding OOS. The VRP is real here — BUT that worst day "
              f"(${worst:,.0f}) is the catch: trade it DEFINED-RISK (spreads, not naked) and small, or one "
              f"vol spike erases months. This is the 'be the insurer' edge — structural, not predictive.")
    elif mean_pos and pos_days > 0.5:
        print(f"premium present but TAIL-HEAVY: positive on average ({pos_days*100:.0f}% of days) yet the "
              f"worst day (${worst:,.0f}) / maxDD (${maxdd:,.0f}) shows the left tail bites hard. Only viable "
              f"defined-risk and small. Net edge is thin once you pay for tail protection.")
    else:
        print(f"no clean premium this window (mean ${pnls.mean():+.1f}/day, {pos_days*100:.0f}% positive days) "
              f"— realized vol kept up with or exceeded implied. The VRP can invert in choppy/crisis regimes; "
              f"not a free lunch here.")
    print("\nNote: a daily-marked short-variance proxy from DVOL — it captures the PREMIUM and the TAIL, not "
          "exact option fills. Real selling needs Deribit options + defined-risk structures (spreads).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

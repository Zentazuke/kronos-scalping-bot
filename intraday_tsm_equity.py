"""intraday_tsm_equity.py — the HONEST dollar answer: run the hardened intraday-TSM
strategy on a real start-capital account and show the equity curve, in dollars.

Everything else reported per-trade or as a pooled sum; this is the faithful version:
a single account that compounds, holds several coins at once, and can't exceed a gross
exposure cap (so 8 coins firing the same day don't secretly become 8x leverage).

Each UTC day:
  * find the coins whose signal fires (trailing vol gate + trailing autocorr regime gate),
  * size each by vol-target (target / trailing_sigma), then scale the whole book down so
    total notional <= --max-gross x equity (default 1.0 = unleveraged),
  * P&L = sum(notional_i * pos_i * afternoon_ret_i) - costs,  equity compounds.

Drawdown is shown in real dollars and dates — the number that actually decides whether
you could hold this. No lookahead (every gate uses only past days).

    .venv\\Scripts\\python.exe intraday_tsm_equity.py --capital 5000 --days 360
    .venv\\Scripts\\python.exe intraday_tsm_equity.py --capital 5000 --max-gross 2 --vol-target 0.012
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from typing import Dict, List, Tuple

import numpy as np

from consensus_backtest import DEFAULT_SYMBOLS, fetch_ohlcv
from intraday_tsm_backtest import day_samples


def coin_signals(samples, *, vol_window: int, vol_q: float, regime_window: int,
                 vol_target: float) -> Dict[str, Tuple[float, float, float]]:
    """day -> (pos, afternoon_ret, weight) for days that FIRE. Causal, no lookahead."""
    morns = [m for _d, m, _a in samples]
    abs_m = [abs(m) for m in morns]
    afts = [a for _d, _m, a in samples]
    out: Dict[str, Tuple[float, float, float]] = {}
    for i, (day, m, a) in enumerate(samples):
        if i < max(vol_window, regime_window):
            continue
        if abs(m) < float(np.quantile(abs_m[i - vol_window:i], vol_q)):
            continue
        if regime_window:
            pm = np.array(morns[i - regime_window:i]); pa = np.array(afts[i - regime_window:i])
            if pm.std() == 0 or pa.std() == 0 or float(np.corrcoef(pm, pa)[0, 1]) <= 0:
                continue
        sigma = float(np.std(afts[i - vol_window:i]))
        w = 0.0 if sigma == 0 else min(vol_target / sigma, 3.0)
        out[day] = (1.0 if m > 0 else -1.0, a, w)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Account-level $ equity sim for hardened intraday-TSM")
    ap.add_argument("--capital", type=float, default=5000.0)
    ap.add_argument("--days", type=int, default=360)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--split", type=int, default=8)
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--vol-q", type=float, default=0.667)
    ap.add_argument("--vol-target", type=float, default=0.010, help="per-trade target vol for sizing")
    ap.add_argument("--regime-window", type=int, default=30, help="0 to disable the regime filter")
    ap.add_argument("--max-gross", type=float, default=1.0, help="max total notional as a multiple of equity")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    sig: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for s in symbols:
        try:
            c = fetch_ohlcv(s, args.timeframe, args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"{s.split('/')[0]:<7} fetch failed ({str(exc)[:30]})"); continue
        samp = day_samples(c, args.split)
        if len(samp) >= 80:
            sig[s.split("/")[0]] = coin_signals(
                samp, vol_window=args.vol_window, vol_q=args.vol_q,
                regime_window=args.regime_window, vol_target=args.vol_target)
    if not sig:
        print("not enough data"); return 1

    all_days = sorted({d for c in sig.values() for d in c})
    equity = args.capital
    peak = equity
    maxdd_usd = 0.0; maxdd_pct = 0.0; trough = equity; dd_peak_day = trough_day = all_days[0]
    cur_peak_day = all_days[0]
    curve: List[Tuple[str, float]] = []
    monthly: Dict[str, float] = defaultdict(float)
    n_trades = 0; days_deployed = 0; gross_used: List[float] = []

    for day in all_days:
        firing = {coin: sig[coin][day] for coin in sig if day in sig[coin]}
        pnl = 0.0
        if firing:
            desired = {coin: w * equity for coin, (_p, _a, w) in firing.items()}
            gross = sum(desired.values())
            scale = min(1.0, args.max_gross * equity / gross) if gross > 0 else 0.0
            for coin, (pos, aft, _w) in firing.items():
                notion = desired[coin] * scale
                pnl += notion * pos * aft - notion * fee
            n_trades += len(firing)
            days_deployed += 1
            gross_used.append(min(gross / equity, args.max_gross))
        equity += pnl
        monthly[day[:7]] += pnl
        curve.append((day, equity))
        if equity > peak:
            peak = equity; cur_peak_day = day
        dd = peak - equity
        if dd > maxdd_usd:
            maxdd_usd = dd; maxdd_pct = dd / peak; trough = equity
            dd_peak_day = cur_peak_day; trough_day = day

    final = equity
    cal_days = max(1, (date.fromisoformat(all_days[-1]) - date.fromisoformat(all_days[0])).days + 1)
    trading_days = len(all_days)
    total_ret = final / args.capital - 1.0
    cagr = (final / args.capital) ** (365.0 / cal_days) - 1.0

    print(f"\n=== INTRADAY-TSM — ${args.capital:,.0f} ACCOUNT SIM "
          f"(vol-target {args.vol_target:g}, regime {args.regime_window}d, max-gross {args.max_gross:g}x) ===")
    print(f"{len(sig)} coins · split {args.split:02d}:00 UTC · {args.fee_bps:g}bps · "
          f"{cal_days} calendar days · {n_trades} trades on {trading_days} trading days\n")
    print(f"  start equity     ${args.capital:,.2f}")
    print(f"  final equity     ${final:,.2f}")
    print(f"  total return     {total_ret*100:+.1f}%   (CAGR {cagr*100:+.1f}%/yr)")
    print(f"  max drawdown     -${maxdd_usd:,.2f}  ({maxdd_pct*100:.1f}%)  "
          f"peak {dd_peak_day} -> trough {trough_day} (${trough:,.0f})")
    avg_gross = (sum(gross_used) / len(gross_used) * 100) if gross_used else 0.0
    print(f"  avg exposure     {avg_gross:.0f}% of equity on trading days · "
          f"deployed {100*days_deployed/len(all_days):.0f}% of days")

    print("\nBY MONTH (account P&L, $):")
    eq_run = args.capital
    for ym in sorted(monthly):
        pnl = monthly[ym]
        start_m = eq_run; eq_run += pnl
        print(f"  {ym}   {('+' if pnl>=0 else '')}{pnl:>9,.0f}   "
              f"({pnl/start_m*100:>+6.1f}%)   -> ${eq_run:,.0f}")

    print("\n=== read ===")
    print(f"Taken at face value, ${args.capital:,.0f} would be ~${final:,.0f} — BUT you'd have ridden a "
          f"-${maxdd_usd:,.0f} ({maxdd_pct*100:.0f}%) drawdown to get there, and this assumes clean fills, "
          f"no slippage, and that the backtested edge repeats. It is a paper curve on past data — the live "
          f"forward test is the only number that counts. Size to the drawdown you can actually stomach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

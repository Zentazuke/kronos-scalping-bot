"""intraday_tsm_careful.py — does trading MORE CAREFULLY (fewer, higher-conviction
intraday bets) actually improve the edge out-of-sample, or just curve-fit?

The 5-year test left intraday-TSM at Sharpe ~0.22 — too thin to trade. "Be more careful"
has a principled version worth testing: raise the volatility gate so we only take the
VERY biggest morning moves (the days early->late momentum is strongest), instead of the
top third. This ladders that one selectivity knob on the HARDENED config
(vol-target + regime [+ on-chain]) over the full history and reads the OUT-OF-SAMPLE
column at each rung.

How to read it honestly:
  * If OOS Sharpe RISES as we get pickier (and stays positive on the 2nd-half data the
    gate never saw), the conditional edge is real — careful bets are worth it.
  * If the gain only shows in-sample while OOS stays flat/negative, it's noise from
    slicing to a smaller, luckier subset — and we stop. Fewer trades = easier to fool
    ourselves, so the OOS column is the only judge.

Note on sizing: Sharpe is ~size-invariant, so "careful" here means PICKIER ENTRIES, not
smaller bets. Smaller size shrinks return and drawdown together; it doesn't create edge.

    python intraday_tsm_careful.py --days 1825 \
        --symbols "ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,LINK/USDT,AVAX/USDT" --onchain-gate
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np

from consensus_backtest import DEFAULT_SYMBOLS, fetch_ohlcv
from intraday_tsm_backtest import day_samples
from intraday_tsm_strategy import build_trades, metrics, split_is_oos

Sample = Tuple[str, float, float]

# selectivity ladder: quantile of trailing |morning move| a day must exceed to trade
LADDER = [
    (0.667, "top 1/3  (base)"),
    (0.750, "top 1/4"),
    (0.850, "top 15%"),
    (0.900, "top 10%"),
    (0.950, "top 5%  (very picky)"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Ladder intraday-TSM selectivity, judged OOS")
    ap.add_argument("--days", type=int, default=1825)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--split", type=int, default=8)
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--vol-target", type=float, default=0.012)
    ap.add_argument("--regime-window", type=int, default=30)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--onchain-gate", action="store_true")
    ap.add_argument("--onchain-window", type=int, default=7)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    per_coin: Dict[str, List[Sample]] = {}
    for s in symbols:
        try:
            c = fetch_ohlcv(s, args.timeframe, args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"{s.split('/')[0]:<7} fetch failed ({str(exc)[:30]})"); continue
        samp = day_samples(c, args.split)
        if len(samp) >= 80:
            per_coin[s.split("/")[0]] = samp
    if not per_coin:
        print("not enough data"); return 1

    risk_off: Optional[set] = None
    if args.onchain_gate:
        try:
            from onchain_flow_scanner import aggregate_supply
            supply = aggregate_supply(args.days)
            sd = sorted(supply)
            risk_off = {sd[i] for i in range(args.onchain_window, len(sd))
                        if supply[sd[i - args.onchain_window]] > 0
                        and supply[sd[i]] / supply[sd[i - args.onchain_window]] - 1.0 < 0}
            print(f"on-chain gate ON: {len(risk_off)} risk-off days of {len(sd)}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"on-chain gate disabled ({str(exc)[:50]})\n"); risk_off = None

    span_days = sorted({d for samp in per_coin.values() for d, _m, _a in samp})
    print(f"=== INTRADAY-TSM 'CAREFUL BETS' — selectivity ladder, hardened config ===")
    print(f"{len(per_coin)} coins · split {args.split:02d}:00 UTC · {args.fee_bps:g}bps · "
          f"{len(span_days)} trading days · vol-target {args.vol_target:g} · "
          f"regime {args.regime_window}d{' · on-chain gate' if risk_off is not None else ''}")
    print("higher rung = pickier (fewer, bigger-conviction days). TRUST THE OOS COLUMNS.\n")
    print(f"{'selectivity':<20}{'trades':>7}{'Sharpe':>8}{'total':>9}{'maxDD':>8}"
          f"{'OOSsh':>7}{'OOStot':>8}{'OOSdd':>7}")
    print("-" * 74)

    results = []
    for q, label in LADDER:
        pooled: List[Tuple[str, float]] = []
        for _coin, samp in per_coin.items():
            pooled += build_trades(samp, fee=fee, vol_window=args.vol_window, vol_q=q,
                                   vol_target=args.vol_target, regime_window=args.regime_window,
                                   risk_off=risk_off)
        pooled.sort(key=lambda t: t[0])
        m = metrics(pooled)
        _is, oos = split_is_oos(pooled)
        mo = metrics(oos)
        results.append((label, m, mo))
        print(f"{label:<20}{m['n']:>7}{m['sharpe']:>8.2f}{m['total']*100:>+8.1f}%"
              f"{m['maxdd']*100:>7.1f}%{mo['sharpe']:>7.2f}{mo['total']*100:>+7.1f}%"
              f"{mo['maxdd']*100:>6.1f}%")

    base_oos = results[0][2]["sharpe"]
    oos_sharpes = [mo["sharpe"] for _l, _m, mo in results]
    best_i = int(np.argmax(oos_sharpes))
    best_label, _bm, best_oos = results[best_i]
    monotonic = all(oos_sharpes[i] <= oos_sharpes[i + 1] + 1e-9 for i in range(len(oos_sharpes) - 1))

    print("\n=== read ===")
    if best_i > 0 and best_oos["sharpe"] > base_oos and best_oos["sharpe"] > 0 and best_oos["total"] > 0:
        trend = "rises monotonically" if monotonic else "improves (not perfectly monotonic)"
        print(f"Careful bets HELP: OOS Sharpe {trend} as we get pickier — best at '{best_label.strip()}' "
              f"(OOS Sharpe {best_oos['sharpe']:.2f} vs base {base_oos:.2f}, OOS total {best_oos['total']*100:+.1f}%). "
              f"The biggest-conviction days carry a stronger, real conditional edge. "
              f"Forward-test THIS rung — don't trust until it survives live.")
        if not monotonic:
            print("  Caveat: the lift isn't clean across every rung, so part may be sampling luck. "
                  "Re-run on a different split / more coins before committing.")
    else:
        print(f"Careful bets do NOT help out-of-sample: getting pickier raised in-sample shine but OOS "
              f"Sharpe stayed flat/worse (base OOS {base_oos:.2f}, best OOS {best_oos['sharpe']:.2f} at "
              f"'{best_label.strip()}'). That's the overfitting tell — a smaller, luckier subset, not a "
              f"stronger edge. Honest conclusion: selectivity isn't the missing piece here.")
    print("\nNote: OOS = 2nd half of the calendar the gate never saw. Fewer trades at the top rungs "
          "means noisier estimates — weight the OOS column, and the trade count, over the in-sample total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

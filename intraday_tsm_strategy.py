"""intraday_tsm_strategy.py — turn the raw intraday-TSM edge into a tradeable strategy.

The base signal (early-session return predicts late-session, traded both directions
on high-vol days) survived every test but has two weaknesses: flat sizing and ugly
momentum-crash months. This hardens it with two upgrades, each causal (no lookahead)
and judged on risk-adjusted terms (Sharpe, max drawdown), in-sample AND out-of-sample:

  1. VOL-TARGET sizing — weight each trade by target/trailing_sigma (clipped), so every
     trade risks roughly the same. Calms the equity curve and shrinks the bad months.

  2. REGIME filter tied to the mechanism — the edge IS positive early->late
     autocorrelation. So only trade when the trailing correlation of (morning, afternoon)
     over the prior --regime-window days is positive; sit out the reverting regimes where
     momentum crashes. One parameter, principled, trailing-only.

It runs four variants — base / +vol-target / +regime / +both — side by side so you can
SEE whether the hardening actually improves Sharpe and drawdown or just curve-fits.

    .venv\\Scripts\\python.exe intraday_tsm_strategy.py --days 360 --split 8 --fee-bps 10
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from consensus_backtest import DEFAULT_SYMBOLS, fetch_ohlcv
from intraday_tsm_backtest import day_samples

Sample = Tuple[str, float, float]   # (day, morning_ret, afternoon_ret)


def build_trades(samples: List[Sample], *, fee: float, vol_window: int, vol_q: float,
                 vol_target: float, regime_window: int,
                 risk_off: Optional[set] = None) -> List[Tuple[str, float]]:
    """Causal per-trade (day, net_ret) for one coin under the chosen options.

    vol_target = 0   -> equal weight (no sizing)
    regime_window = 0 -> no regime filter
    risk_off = set of days when stablecoin supply is contracting -> suppress LONGs
              (the on-chain risk-off gate; uses only the sign of trailing supply growth,
              so no lookahead). None -> gate off.
    Every gate/threshold uses only PAST samples; nothing peeks ahead.
    """
    morns = [m for _d, m, _a in samples]
    abs_m = [abs(m) for m in morns]
    afts = [a for _d, _m, a in samples]
    out: List[Tuple[str, float]] = []
    for i, (day, m, a) in enumerate(samples):
        if i < max(vol_window, regime_window):
            continue                                    # warm-up: not enough trailing history
        # --- volatility gate (trailing): only trade big-move days
        thr = float(np.quantile(abs_m[i - vol_window:i], vol_q))
        if abs(m) < thr:
            continue
        # --- regime filter (trailing): trade only when early->late autocorr is positive
        if regime_window:
            pm = np.array(morns[i - regime_window:i])
            pa = np.array(afts[i - regime_window:i])
            if pm.std() == 0 or pa.std() == 0:
                continue
            if float(np.corrcoef(pm, pa)[0, 1]) <= 0:
                continue                                # reverting regime -> sit out
        pos = 1.0 if m > 0 else -1.0
        # --- on-chain risk-off gate: don't take LONGs when stablecoin supply is contracting
        if risk_off is not None and pos > 0 and day in risk_off:
            continue
        # --- vol-target sizing (trailing realized vol of afternoon moves)
        w = 1.0
        if vol_target > 0:
            sigma = float(np.std(afts[i - vol_window:i]))
            w = 0.0 if sigma == 0 else min(vol_target / sigma, 3.0)   # cap 3x leverage
        out.append((day, w * pos * a - w * fee))
    return out


def metrics(trades: List[Tuple[str, float]]) -> Dict[str, float]:
    if not trades:
        return {"n": 0, "mean": 0.0, "sharpe": 0.0, "total": 0.0, "maxdd": 0.0, "worst_month": 0.0}
    arr = np.array([r for _d, r in trades])
    mean, sd = float(arr.mean()), float(arr.std())
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    maxdd = float((peak - cum).max())
    by_month: Dict[str, float] = defaultdict(float)
    for day, r in trades:
        by_month[day[:7]] += r
    worst = min(by_month.values()) if by_month else 0.0
    # per-trade Sharpe, annualized assuming ~2 trades/coin/day across the book
    sharpe = (mean / sd * math.sqrt(252.0)) if sd > 0 else 0.0
    return {"n": len(arr), "mean": mean, "sharpe": sharpe, "total": float(arr.sum()),
            "maxdd": maxdd, "worst_month": worst}


def split_is_oos(trades: List[Tuple[str, float]]) -> Tuple[List, List]:
    if not trades:
        return [], []
    days = sorted({d for d, _r in trades})
    cut = days[len(days) // 2]
    return ([t for t in trades if t[0] < cut], [t for t in trades if t[0] >= cut])


def main() -> int:
    ap = argparse.ArgumentParser(description="Harden the intraday-TSM edge (vol-target + regime)")
    ap.add_argument("--days", type=int, default=360)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--split", type=int, default=8)
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--vol-q", type=float, default=0.667)
    ap.add_argument("--vol-target", type=float, default=0.012, help="per-trade target vol for sizing")
    ap.add_argument("--regime-window", type=int, default=30, help="trailing days for the autocorr regime gate")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--onchain-gate", action="store_true",
                    help="layer the stablecoin risk-off gate (suppress longs when supply contracts)")
    ap.add_argument("--onchain-window", type=int, default=7, help="trailing days for supply-growth sign")
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

    # On-chain risk-off gate: days when aggregate stablecoin supply is contracting
    # (sign of trailing growth -> no lookahead). Suppress LONGs on those days.
    risk_off = None
    if args.onchain_gate:
        try:
            from onchain_flow_scanner import aggregate_supply
            supply = aggregate_supply(args.days)
            sdays = sorted(supply)
            risk_off = {sdays[i] for i in range(args.onchain_window, len(sdays))
                        if supply[sdays[i - args.onchain_window]] > 0
                        and supply[sdays[i]] / supply[sdays[i - args.onchain_window]] - 1.0 < 0}
            print(f"on-chain gate: {len(risk_off)} risk-off (supply-contracting) days "
                  f"of {len(sdays)} — longs suppressed on those days\n")
        except Exception as exc:  # noqa: BLE001
            print(f"on-chain gate disabled (stablecoin fetch failed: {str(exc)[:50]})\n"); risk_off = None

    variants = {
        "base (equal wt)":      dict(vol_target=0.0, regime_window=0),
        "+ vol-target":         dict(vol_target=args.vol_target, regime_window=0),
        "+ regime filter":      dict(vol_target=0.0, regime_window=args.regime_window),
        "+ both (hardened)":    dict(vol_target=args.vol_target, regime_window=args.regime_window),
    }
    if risk_off is not None:
        variants["+ both + onchain"] = dict(vol_target=args.vol_target,
                                            regime_window=args.regime_window, risk_off=risk_off)

    print(f"\n=== INTRADAY-TSM HARDENING — split {args.split:02d}:00 UTC, {len(per_coin)} coins, "
          f"{args.fee_bps:g}bps, {args.days}d ===")
    print("Sharpe = annualized (higher better) · maxDD / worst-month in cumulative-return units "
          "(closer to 0 better)\n")
    print(f"{'variant':<20}{'trades':>7}{'Sharpe':>8}{'total':>9}{'maxDD':>8}{'worstMo':>9}"
          f"{'OOS Sh':>8}{'OOS tot':>9}")
    rows = {}
    for name, opt in variants.items():
        pooled: List[Tuple[str, float]] = []
        for coin, samp in per_coin.items():
            pooled += build_trades(samp, fee=fee, vol_window=args.vol_window, vol_q=args.vol_q, **opt)
        pooled.sort(key=lambda t: t[0])
        m = metrics(pooled)
        is_t, oos_t = split_is_oos(pooled)
        mo = metrics(oos_t)
        rows[name] = (m, mo)
        print(f"{name:<20}{m['n']:>7}{m['sharpe']:>8.2f}{m['total']*100:>+8.1f}%"
              f"{m['maxdd']*100:>7.1f}%{m['worst_month']*100:>+8.1f}%{mo['sharpe']:>8.2f}{mo['total']*100:>+8.1f}%")

    base = rows["base (equal wt)"][0]
    hard = rows["+ both (hardened)"][0]
    hard_oos = rows["+ both (hardened)"][1]
    print("\n=== read ===")
    better_sharpe = hard["sharpe"] > base["sharpe"]
    better_dd = hard["maxdd"] < base["maxdd"]
    oos_ok = hard_oos["sharpe"] > 0 and hard_oos["total"] > 0
    if better_sharpe and better_dd and oos_ok:
        print(f"hardening HELPS: Sharpe {base['sharpe']:.2f} -> {hard['sharpe']:.2f}, "
              f"maxDD {base['maxdd']*100:.1f}% -> {hard['maxdd']*100:.1f}%, worst month "
              f"{base['worst_month']*100:+.1f}% -> {hard['worst_month']*100:+.1f}%, and it holds OOS "
              f"(Sharpe {hard_oos['sharpe']:.2f}). Vol-target + regime turn the raw edge into a "
              f"smoother, more tradeable curve. This is the config to forward-test next.")
    elif better_sharpe or better_dd:
        print(f"partial: hardening improves {'Sharpe' if better_sharpe else 'drawdown'} but not both "
              f"cleanly (or OOS is shaky: OOS Sharpe {hard_oos['sharpe']:.2f}). Worth keeping the piece "
              f"that helps; don't stack knobs that only flatter the in-sample.")
    else:
        print(f"hardening does NOT improve risk-adjusted return (Sharpe {base['sharpe']:.2f} -> "
              f"{hard['sharpe']:.2f}) — the extra knobs are curve-fitting, not real. Keep the base, "
              f"size it small, and accept the rough months. Simpler is more honest here.")
    if "+ both + onchain" in rows:
        oc, oc_oos = rows["+ both + onchain"]
        helps = oc["sharpe"] >= hard["sharpe"] and oc["maxdd"] <= hard["maxdd"] and oc_oos["sharpe"] > 0
        print(f"\nON-CHAIN GATE: hardened {hard['sharpe']:.2f} Sharpe / {hard['maxdd']*100:.1f}% maxDD "
              f"-> +onchain {oc['sharpe']:.2f} / {oc['maxdd']*100:.1f}% (OOS Sharpe {oc_oos['sharpe']:.2f}).")
        if helps:
            print("  Two independent legs (price momentum + on-chain flow) stack: suppressing longs when "
                  "stablecoin supply contracts improves risk-adjusted return and holds OOS. Real diversification.")
        else:
            print("  The on-chain gate doesn't cleanly improve the hardened config here — the two signals "
                  "may overlap more than hoped, or the gate is too blunt. Keep it as context, not a hard gate.")
    print("\nNote: trailing gates only (no lookahead); OOS = second half of the calendar. "
          "Sweep --vol-target / --regime-window, but trust the OOS column over the in-sample shine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

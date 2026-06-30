"""analyst_gate_verify.py — is the analyst-gate OOS lift REAL or just trade-count luck?

intraday_tsm_analyst.py showed the analyst-agreement gate lifting intraday OOS Sharpe
0.13 -> 1.03 while halving the trades. Before believing it, rule out the obvious
artifact: maybe trading half as often just happens to dodge bad periods, and ANY gate
of that selectivity would look good.

Two controls:
  1. PLACEBO (the decisive one): shuffle the lean across dates — same bullish/bearish mix,
     same ~halving of trades, but the day-alignment is destroyed. Run it N times. If the
     REAL gate's OOS Sharpe sits inside the placebo cloud, the lift is an artifact. If it
     sits well ABOVE the cloud, the analyst is genuinely picking the right trades.
  2. BY-MONTH OOS: is the OOS profit spread across the period, or carried by one explosive
     month (the Jan-2026 mirage we've seen twice)?

    python analyst_gate_verify.py --days 1825 \
        --symbols "ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,LINK/USDT,AVAX/USDT" \
        --gate-symbol BTC_USDT --placebo 50
"""
from __future__ import annotations

import argparse
import bisect
import csv
import os
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from consensus_backtest import DEFAULT_SYMBOLS, fetch_ohlcv
from intraday_tsm_backtest import day_samples
from intraday_tsm_strategy import metrics, split_is_oos
from intraday_tsm_analyst import build_trades


def read_rows(path: str, gate_symbol: str, timeframe: str = "1D"):
    rows = []
    for r in csv.DictReader(open(path)):
        if r["symbol"] == gate_symbol and r["timeframe"] == timeframe:
            rows.append((r["date"], r["lean"], float(r["score"]), r["regime"]))
    rows.sort(key=lambda t: t[0])
    return rows


def make_lookup(rows):
    dates = [r[0] for r in rows]

    def at(day: str):
        i = bisect.bisect_left(dates, day)
        if i == 0:
            return None
        return rows[i - 1][1], rows[i - 1][2], rows[i - 1][3]
    return at


def run_gate(per_coin, base, analyst_at, gate_kwargs) -> List[Tuple[str, float]]:
    pooled: List[Tuple[str, float]] = []
    for _coin, samp in per_coin.items():
        pooled += build_trades(samp, **base, analyst_at=analyst_at, **gate_kwargs)
    pooled.sort(key=lambda t: t[0])
    return pooled


def main() -> int:
    ap = argparse.ArgumentParser(description="Placebo + by-month verification of the analyst gate")
    ap.add_argument("--days", type=int, default=1825)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--split", type=int, default=8)
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--vol-q", type=float, default=0.667)
    ap.add_argument("--vol-target", type=float, default=0.012)
    ap.add_argument("--regime-window", type=int, default=30)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--gate-symbol", default="BTC_USDT")
    ap.add_argument("--gate-tf", default="1D", help="analyst timeframe (1D or 4h)")
    ap.add_argument("--mode", default="trend", choices=["trend", "agree"],
                    help="gate on the regime trend label (trend) or the ensemble lean (agree)")
    ap.add_argument("--lean-csv", default="data_cache/analyst_lean.csv")
    ap.add_argument("--placebo", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0
    gate_kwargs = {"require_trend": True} if args.mode == "trend" else {"require_agree": True}

    rows = read_rows(args.lean_csv, args.gate_symbol, args.gate_tf)
    if not rows:
        print(f"no analyst rows for {args.gate_symbol} {args.gate_tf} in {args.lean_csv}"); return 1
    base = dict(fee=fee, vol_window=args.vol_window, vol_q=args.vol_q,
                vol_target=args.vol_target, regime_window=args.regime_window)

    per_coin = {}
    for s in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        try:
            c = fetch_ohlcv(s, args.timeframe, args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"{s} fetch failed ({str(exc)[:30]})"); continue
        samp = day_samples(c, args.split)
        if len(samp) >= 80:
            per_coin[s.split("/")[0]] = samp
    if not per_coin:
        print("not enough data"); return 1

    # --- REAL gate ---
    real = run_gate(per_coin, base, make_lookup(rows), gate_kwargs)
    _is, oos = split_is_oos(real)
    real_oos = metrics(oos)
    print(f"\n=== REAL analyst gate ({args.mode}, {args.gate_symbol} {args.gate_tf}) ===")
    print(f"  trades {len(real)} · OOS Sharpe {real_oos['sharpe']:.2f} · OOS total "
          f"{real_oos['total']*100:+.1f}% · OOS maxDD {real_oos['maxdd']*100:.1f}%")

    # by-month OOS — is it spread, or one month?
    bym = defaultdict(float)
    for d, r in oos:
        bym[d[:7]] += r
    months = sorted(bym)
    pos = sum(1 for v in bym.values() if v > 0)
    top = max(bym.values()); top_m = [m for m in months if bym[m] == top][0]
    tot = sum(bym.values())
    print(f"  OOS by-month: {pos}/{len(months)} positive · biggest month {top_m} "
          f"{top*100:+.1f}% = {top/tot*100:.0f}% of the OOS total")
    for m in months:
        bar = "#" * max(0, int(bym[m] * 200))
        print(f"    {m}  {bym[m]*100:>+6.1f}%  {bar}")

    # --- PLACEBO control ---
    random.seed(args.seed)
    leans = [(r[1], r[2], r[3]) for r in rows]
    dates = [r[0] for r in rows]
    placebo_oos = []
    placebo_trades = []
    for _ in range(args.placebo):
        perm = leans[:]
        random.shuffle(perm)
        prows = [(dates[j], perm[j][0], perm[j][1], perm[j][2]) for j in range(len(dates))]
        pooled = run_gate(per_coin, base, make_lookup(prows), gate_kwargs)
        _i, po = split_is_oos(pooled)
        placebo_oos.append(metrics(po)["sharpe"])
        placebo_trades.append(len(pooled))
    arr = np.array(placebo_oos)
    above = int(np.sum(arr >= real_oos["sharpe"]))
    print(f"\n=== PLACEBO control ({args.placebo} shuffled-lean trials) ===")
    print(f"  placebo trade count ~{int(np.mean(placebo_trades))} (real {len(real)}) — same selectivity")
    print(f"  placebo OOS Sharpe: mean {arr.mean():+.2f} · std {arr.std():.2f} · "
          f"min {arr.min():+.2f} · max {arr.max():+.2f}")
    print(f"  real OOS Sharpe {real_oos['sharpe']:.2f}  ·  placebos >= real: {above}/{args.placebo} "
          f"(empirical p = {above/args.placebo:.3f})")

    print("\n=== verdict ===")
    if above <= max(1, int(0.05 * args.placebo)) and real_oos["sharpe"] > arr.mean() + 2 * arr.std():
        print(f"REAL: the gate's OOS Sharpe ({real_oos['sharpe']:.2f}) sits well above the placebo cloud "
              f"(mean {arr.mean():+.2f}); shuffling the lean destroys the lift. The analyst is picking the "
              f"RIGHT trades, not just fewer trades. This is a genuine, orthogonal edge — forward-test it.")
        if pos < 0.5 * len(months) or top / tot > 0.6:
            print("  CAUTION: but the OOS profit is concentrated in few months — confirm it's not one stretch.")
    else:
        print(f"ARTIFACT: placebos reach the real gate's OOS Sharpe {above}/{args.placebo} of the time "
              f"(real {real_oos['sharpe']:.2f} vs placebo mean {arr.mean():+.2f}). The lift is mostly from "
              f"trading less, not from the analyst's information. Don't trust the headline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

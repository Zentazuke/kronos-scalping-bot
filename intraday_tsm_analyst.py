"""intraday_tsm_analyst.py — does gating intraday-TSM on the crypto-analyst's
higher-timeframe lean improve it out-of-sample?

The selectivity test failed (pickier-by-morning-size = worse OOS). This tries a
DIFFERENT kind of filter: an orthogonal, independently-calibrated signal — the
analyst's daily ensemble lean + regime — used as a higher-timeframe gate.

Hypothesis: only press the intraday momentum bet when the higher-timeframe regime
agrees (buy a morning-up day when the 1D lean is bullish; sell a morning-down day
when bearish), and stand down in flash / low-vol regimes.

Causal by construction:
  * the analyst lean comes from export_lean.py, which replays the analyst's own
    causal functions on an expanding window (row for day d uses only data <= d);
  * here we LAG it one more day — the 08:00 decision on day d is gated on the
    analyst's lean as of the last completed daily bar (date < d). No lookahead.

Reads data_cache/analyst_lean.csv (from the analyst's export_lean.py). Judged on the
same 5-year OOS column as every other test.

    python intraday_tsm_analyst.py --days 1825 \
        --symbols "ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,LINK/USDT,AVAX/USDT" \
        --gate-symbol BTC_USDT --lean-csv data_cache/analyst_lean.csv
"""
from __future__ import annotations

import argparse
import bisect
import csv
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from consensus_backtest import DEFAULT_SYMBOLS, fetch_ohlcv
from intraday_tsm_backtest import day_samples
from intraday_tsm_strategy import metrics, split_is_oos

Sample = Tuple[str, float, float]
STANDDOWN_REGIMES = {"flash_risk", "low_vol_drift"}


def load_lean(path: str, gate_symbol: str, timeframe: str = "1D"):
    """Return analyst_at(day) -> (lean, score, regime) for the most recent analyst
    date STRICTLY BEFORE `day` (the 1-day lag), or None if none exists yet."""
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run the analyst's export_lean.py first and copy it here.")
    rows = []
    for r in csv.DictReader(open(path)):
        if r["symbol"] == gate_symbol and r["timeframe"] == timeframe:
            rows.append((r["date"], r["lean"], float(r["score"]), r["regime"]))
    rows.sort(key=lambda t: t[0])
    dates = [t[0] for t in rows]
    if not dates:
        raise SystemExit(f"no rows for {gate_symbol} {timeframe} in {path}")

    def analyst_at(day: str):
        i = bisect.bisect_left(dates, day)   # first index with date >= day
        if i == 0:
            return None                      # no analyst data before this day
        d, lean, score, regime = rows[i - 1]
        return lean, score, regime
    return analyst_at, (dates[0], dates[-1], len(dates))


def build_trades(samples: List[Sample], *, fee: float, vol_window: int, vol_q: float,
                 vol_target: float, regime_window: int,
                 analyst_at: Optional[Callable] = None, require_agree: bool = False,
                 strict: bool = False, regime_standdown: bool = False,
                 require_trend: bool = False) -> List[Tuple[str, float]]:
    """Causal per-trade (day, net_ret) for one coin, hardened config + optional analyst gate.

    require_trend: the analyst's regime TREND LABEL must agree with the direction —
        take a LONG only when the (lagged) regime is 'trend_up', a SHORT only when
        'trend_down'. This is the '4h + trend labels' gate (the model card's strongest edge).
    require_agree: same idea but on the ensemble lean (bullish/bearish) instead of the label.
    """
    morns = [m for _d, m, _a in samples]
    abs_m = [abs(m) for m in morns]
    afts = [a for _d, _m, a in samples]
    out: List[Tuple[str, float]] = []
    for i, (day, m, a) in enumerate(samples):
        if i < max(vol_window, regime_window):
            continue
        thr = float(np.quantile(abs_m[i - vol_window:i], vol_q))
        if abs(m) < thr:
            continue
        if regime_window:
            pm = np.array(morns[i - regime_window:i]); pa = np.array(afts[i - regime_window:i])
            if pm.std() == 0 or pa.std() == 0 or float(np.corrcoef(pm, pa)[0, 1]) <= 0:
                continue
        pos = 1.0 if m > 0 else -1.0
        # --- analyst higher-timeframe gate (lagged, causal) ---
        if (require_agree or require_trend or regime_standdown) and analyst_at is not None:
            ar = analyst_at(day)
            if ar is not None:
                lean, _score, regime = ar
                if regime_standdown and regime in STANDDOWN_REGIMES:
                    continue
                if require_trend:
                    if pos > 0 and regime != "trend_up":
                        continue
                    if pos < 0 and regime != "trend_down":
                        continue
                if require_agree:
                    if strict:
                        if pos > 0 and lean != "bullish":
                            continue
                        if pos < 0 and lean != "bearish":
                            continue
                    else:
                        if pos > 0 and lean == "bearish":
                            continue
                        if pos < 0 and lean == "bullish":
                            continue
            # ar is None (no analyst history yet) -> fail-open, take the trade
        w = 1.0
        if vol_target > 0:
            sigma = float(np.std(afts[i - vol_window:i]))
            w = 0.0 if sigma == 0 else min(vol_target / sigma, 3.0)
        out.append((day, w * pos * a - w * fee))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate intraday-TSM on the analyst lean; judge OOS")
    ap.add_argument("--days", type=int, default=1825)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--split", type=int, default=8)
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--vol-q", type=float, default=0.667)
    ap.add_argument("--vol-target", type=float, default=0.012)
    ap.add_argument("--regime-window", type=int, default=30)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--gate-symbol", default="BTC_USDT", help="analyst symbol used as the market gate")
    ap.add_argument("--gate-tf", default="1D", help="analyst timeframe to gate on (1D or 4h)")
    ap.add_argument("--lean-csv", default="data_cache/analyst_lean.csv")
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    analyst_at, (a0, a1, an) = load_lean(args.lean_csv, args.gate_symbol, args.gate_tf)
    print(f"analyst gate: {args.gate_symbol} {args.gate_tf} lean+regime, {an} rows {a0} -> {a1} (lagged)\n")

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

    base = dict(fee=fee, vol_window=args.vol_window, vol_q=args.vol_q,
                vol_target=args.vol_target, regime_window=args.regime_window)
    variants = {
        "base (hardened)":        dict(),
        "+ trend-label agree":    dict(analyst_at=analyst_at, require_trend=True),
        "+ trend + stand-down":   dict(analyst_at=analyst_at, require_trend=True, regime_standdown=True),
        "+ lean agree":           dict(analyst_at=analyst_at, require_agree=True),
        "+ regime stand-down":    dict(analyst_at=analyst_at, regime_standdown=True),
    }

    span = sorted({d for s in per_coin.values() for d, _m, _a in s})
    print(f"=== INTRADAY-TSM x ANALYST GATE — {len(per_coin)} coins · split {args.split:02d}:00 UTC · "
          f"{args.fee_bps:g}bps · {len(span)} trading days ===")
    print("Does the analyst's higher-TF lean improve intraday OOS? TRUST THE OOS COLUMNS.\n")
    print(f"{'variant':<24}{'trades':>7}{'Sharpe':>8}{'total':>9}{'maxDD':>8}{'OOSsh':>7}{'OOStot':>8}{'OOSdd':>7}")
    print("-" * 78)

    results = []
    for name, gate in variants.items():
        pooled: List[Tuple[str, float]] = []
        for _coin, samp in per_coin.items():
            pooled += build_trades(samp, **base, **gate)
        pooled.sort(key=lambda t: t[0])
        m = metrics(pooled)
        _is, oos = split_is_oos(pooled)
        mo = metrics(oos)
        results.append((name, m, mo))
        print(f"{name:<24}{m['n']:>7}{m['sharpe']:>8.2f}{m['total']*100:>+8.1f}%"
              f"{m['maxdd']*100:>7.1f}%{mo['sharpe']:>7.2f}{mo['total']*100:>+7.1f}%{mo['maxdd']*100:>6.1f}%")

    base_oos = results[0][2]["sharpe"]
    best = max(results[1:], key=lambda r: r[2]["sharpe"]) if len(results) > 1 else results[0]
    bname, _bm, boos = best
    print("\n=== read ===")
    if boos["sharpe"] > base_oos and boos["sharpe"] > 0 and boos["total"] > 0:
        print(f"The analyst gate HELPS out-of-sample: best variant '{bname.strip()}' lifts OOS Sharpe "
              f"{base_oos:.2f} -> {boos['sharpe']:.2f} (OOS total {boos['total']*100:+.1f}%). Two orthogonal "
              f"edges (intraday momentum + higher-TF regime) stacking is real diversification. "
              f"Forward-test this gate live before trusting it.")
    else:
        print(f"The analyst gate does NOT improve intraday out-of-sample (base OOS {base_oos:.2f}, best "
              f"gated {boos['sharpe']:.2f} at '{bname.strip()}'). The higher-TF lean and the intraday edge "
              f"don't combine into something tradeable here — record it and move on, no knob-twisting.")
    print("\nNote: analyst lean is lagged 1 day (08:00 decision sees only the prior completed daily bar). "
          "OOS = 2nd half of the calendar. Fewer trades in gated variants = noisier — weight OOS + trade count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

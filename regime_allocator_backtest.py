"""regime_allocator_backtest.py — "the market was pumping and the bot sat flat."

The honest fix for that feeling is NOT loosening intraday-TSM's gates (it's a
session-momentum strategy; missing pumps is the cost of its edge claim). It's a
SEPARATE sleeve: use the regime classifier — the analyst's one validated skill —
to decide when to simply BE LONG, and when to stand aside.

Sleeves compared per coin, net of taker+slippage on every flip:
  * buy & hold                — the benchmark that made +40% while timing lost money
  * trend-rider               — long ONLY while regime == trend_up, else cash
  * default-long              — long UNLESS regime is trend_down/flash (catches pumps
                                by default, steps aside only in damage regimes)

The bar (pre-registered): a sleeve earns a live shadow test only if, OUT-OF-SAMPLE
and across most of the basket, it beats buy & hold risk-adjusted (higher Sharpe) —
or matches its return with clearly lower drawdown. Otherwise: buy & hold wins,
verdict recorded, move on. This has a LOW bar to add value (it doesn't need to
predict, just not be long in crashes) — which is exactly why it's worth testing.

Regime source: internal efficiency-ratio classifier (lagged, causal — same as the
fixed grid backtest) by default; --regime-csv uses THE ANALYST'S causal export
(data_cache/analyst_lean.csv — regime + confidence, 2021->now, all 7 coins). This is
the "pump confirmation" Ricardo asked for: the analyst's regime signal is its one
RELIABLE call (SIGNAL_CARD: 66% vs 52% base on BTC 1h/24-bar, stable both halves),
`trend_up`/`breakout` = confirmed pump, and it fires on ANY day, no fixed hour.
Causality: the export's row for day D is computed on D's close (its own docstring
says "lag by one day"), so each row is shifted +1 bar-period before use here.
--min-conf N treats low-confidence trend_up/breakout as UNconfirmed (stays flat).

    python regime_allocator_backtest.py --days 1850
    python regime_allocator_backtest.py --days 1850 --regime-csv data_cache/analyst_lean.csv --min-conf 0.6
"""
from __future__ import annotations

import argparse
import csv as _csv
import math
import os
from typing import Dict, List, Optional

import numpy as np

from consensus_backtest import fetch_ohlcv
from costs import SLIPPAGE_BPS, taker_bps
from grid_backtest import (TREND_REGIMES, atr_pct, internal_regime, lag1,
                           regime_series, smooth_regime)

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
           "DOGE/USDT", "LINK/USDT", "AVAX/USDT"]
FLIP_FEE = (taker_bps() + SLIPPAGE_BPS) / 1e4      # per position change (one taker leg)
IS_FRAC = 0.6
DAMAGE = {"trend_down", "flash", "flash_risk"}
CONFIRM = {"trend_up", "breakout"}                 # the analyst's "pump confirmed" labels

SLEEVES = {
    "buy&hold":     lambda reg: True,
    "trend-rider":  lambda reg: reg in CONFIRM,
    "default-long": lambda reg: reg not in DAMAGE,
}

_PERIOD_MS = {"1D": 86_400_000, "1d": 86_400_000, "4h": 14_400_000, "1h": 3_600_000}


def load_analyst_regime(path: str, symbol: str, min_conf: float) -> list:
    """(effective_ts, regime) from the analyst's lean export, causally shifted:
    the row for bar D is computed on D's CLOSE, so it becomes usable one full
    bar-period later. Low-confidence trend_up/breakout is downgraded to 'range'
    (an unconfirmed pump is not a confirmation)."""
    sym = symbol.replace("/", "_")
    rows = []
    for r in _csv.DictReader(open(path)):
        if r.get("symbol") not in (sym, symbol) or not r.get("regime"):
            continue
        if not r.get("ts"):
            continue
        ts = int(r["ts"]) + _PERIOD_MS.get(r.get("timeframe", "1D"), 86_400_000)
        reg = r["regime"]
        try:
            conf = float(r.get("confidence") or 0.0)
        except ValueError:
            conf = 0.0
        if reg in CONFIRM and conf < min_conf:
            reg = "range"
        rows.append((ts, reg))
    rows.sort()
    return rows


def run_sleeve(closes: np.ndarray, regime: List[str], want_long) -> np.ndarray:
    """Equity curve for one sleeve. Position for bar i is decided by regime[i]
    (already causal/lagged); a flip pays FLIP_FEE on the whole stake."""
    eq = np.ones(len(closes))
    pos = 1 if want_long(regime[0]) else 0
    val = 1.0 - (FLIP_FEE if pos else 0.0)          # entering at the start costs a leg
    eq[0] = val
    flips = 0
    for i in range(1, len(closes)):
        ret = closes[i] / closes[i - 1] - 1.0
        val *= (1.0 + (ret if pos else 0.0))
        want = 1 if want_long(regime[i]) else 0
        if want != pos:
            val *= (1.0 - FLIP_FEE)
            pos = want
            flips += 1
        eq[i] = val
    run_sleeve.last_flips = flips                    # cheap side-channel for reporting
    return eq


def seg_metrics(eq: np.ndarray) -> Dict[str, float]:
    ret = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(ret) < 2 or ret.std() == 0:
        return {"total": float(eq[-1] / eq[0] - 1), "sharpe": 0.0, "maxdd": 0.0}
    cum = eq / eq[0]
    dd = float((np.maximum.accumulate(cum) - cum).max() / np.maximum.accumulate(cum).max())
    return {"total": float(eq[-1] / eq[0] - 1),
            "sharpe": float(ret.mean() / ret.std() * math.sqrt(8760)),
            "maxdd": dd}


def main() -> int:
    ap = argparse.ArgumentParser(description="Regime classifier as a long/cash allocator")
    ap.add_argument("--days", type=int, default=1850)
    ap.add_argument("--er-win", type=int, default=24)
    ap.add_argument("--er-thr", type=float, default=0.35)
    ap.add_argument("--regime-persist", type=int, default=24)
    ap.add_argument("--regime-csv", default="", help="use the analyst's regime export "
                    "(e.g. data_cache/analyst_lean.csv)")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="min analyst confidence for trend_up/breakout to count as confirmed")
    args = ap.parse_args()

    per_sleeve_oos: Dict[str, List[Dict[str, float]]] = {k: [] for k in SLEEVES}
    beats: Dict[str, int] = {k: 0 for k in SLEEVES if k != "buy&hold"}
    n_coins = 0
    for sym in SYMBOLS:
        try:
            c = fetch_ohlcv(sym, "1h", args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: fetch failed ({str(exc)[:60]}) — skip"); continue
        if len(c) < 500:
            print(f"  {sym}: only {len(c)} bars — skip"); continue
        n_coins += 1
        ts = [int(x[0]) for x in c]
        closes = np.array([x[4] for x in c], float)
        ap_ = atr_pct(c)
        if args.regime_csv and os.path.exists(args.regime_csv):
            rows = load_analyst_regime(args.regime_csv, sym, args.min_conf)
            if rows:
                # analyst labels: already causal (+1 bar shift in the loader) and already
                # persistent from the classifier's own hysteresis — no extra smoothing.
                regime = regime_series(rows, ts)
            else:
                print(f"  {sym}: no rows in {args.regime_csv} — internal fallback")
                regime = smooth_regime(lag1(internal_regime(c, args.er_win, args.er_thr, ap_)),
                                       args.regime_persist)
        else:
            regime = smooth_regime(lag1(internal_regime(c, args.er_win, args.er_thr, ap_)),
                                   args.regime_persist)
        split = int(len(c) * IS_FRAC)

        print(f"\n=== {sym} ({len(c)} bars) ===")
        print(f"  {'sleeve':<13} | {'IS tot':>8} {'IS Sh':>6} | {'OOS tot':>8} {'OOS Sh':>6} "
              f"{'OOS DD':>6} {'flips':>6}")
        bh_oos: Optional[Dict[str, float]] = None
        for name, fn in SLEEVES.items():
            eq = run_sleeve(closes, regime, fn)
            flips = run_sleeve.last_flips
            mi, mo = seg_metrics(eq[:split]), seg_metrics(eq[split:])
            per_sleeve_oos[name].append(mo)
            if name == "buy&hold":
                bh_oos = mo
            elif bh_oos is not None and (
                    mo["sharpe"] > bh_oos["sharpe"]
                    or (mo["total"] >= bh_oos["total"] * 0.9 and mo["maxdd"] < bh_oos["maxdd"] * 0.6)):
                beats[name] += 1
            print(f"  {name:<13} | {mi['total']*100:>+7.1f}% {mi['sharpe']:>6.2f} | "
                  f"{mo['total']*100:>+7.1f}% {mo['sharpe']:>6.2f} {mo['maxdd']*100:>5.0f}% {flips:>6}")

    if not n_coins:
        print("no usable symbols"); return 1
    med = lambda xs: float(np.median(xs)) if xs else 0.0
    print("\n=== basket medians (OOS) ===")
    for name, ms in per_sleeve_oos.items():
        print(f"  {name:<13} total {med([m['total'] for m in ms])*100:>+7.1f}%  "
              f"Sharpe {med([m['sharpe'] for m in ms]):>5.2f}  "
              f"maxDD {med([m['maxdd'] for m in ms])*100:>4.0f}%"
              + (f"  beats B&H on {beats[name]}/{n_coins} coins" if name in beats else "  (benchmark)"))
    print("\n=== read (pre-registered) ===")
    print("A sleeve goes to a shadow forward test ONLY if it beats buy & hold OOS (Sharpe, or")
    print("~equal return with much lower DD) on MOST of the basket. If buy & hold wins, that IS")
    print("the answer to 'the market pumped and we were flat': the honest pump-catcher is owning")
    print("the asset, gated only by the damage regimes — and if even that fails, just owning it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

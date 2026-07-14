"""sleeve_consumption_study.py — HOW should the validated regime signal be consumed?

Pre-registered 2026-07-14 (see Strategy_Roadmap_2026-07-14.md — the variant list and
the success bar were locked before any results). This optimizes the IMPLEMENTATION of
a signal that already passed its bar; it is not a search for a new edge. It does NOT
touch the running shadow test: a winning variant becomes a THIRD shadow sleeve after
the ~Aug 1 sleeve read, never an edit to the live ones.

Variants (only these):
  baseline  binary trend-rider (long iff trend_up/breakout) and default-long
            (long unless damage) — exactly what runs live in shadow.
  A         confidence-scaled: exposure = analyst confidence when confirmed, else 0
            (TR) / 1 - conf when damage-labeled, else 1 (DL uses damage confidence
            to scale OUT: high-confidence damage -> flat, low-confidence -> partial).
  B         vol-targeted: binary exposure x min(1, TARGET_VOL / realized_30d).
  C         A x B.
  D         portfolio-level: basket exposure = share of coins confirmed (TR frame),
            equal-weight, rebalanced only when the target shifts > 15pp (no-trade band).

Costs: one taker+slip leg per exposure change, scaled by |delta exposure|.
Success bar: beat BASELINE OOS Sharpe with maxDD no worse, on the basket AND >=4/7
coins. Anything less -> keep the baseline.

    python sleeve_consumption_study.py --days 1650 --regime-csv data_cache/analyst_lean.csv
"""
from __future__ import annotations

import argparse
import bisect
import csv as _csv
import math
import os
from typing import Dict, List

import numpy as np

from consensus_backtest import fetch_ohlcv
from costs import SLIPPAGE_BPS, taker_bps

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
           "DOGE/USDT", "LINK/USDT", "AVAX/USDT"]
CONFIRM = {"trend_up", "breakout"}
DAMAGE = {"trend_down", "flash", "flash_risk"}
FLIP_FEE = (taker_bps() + SLIPPAGE_BPS) / 1e4
IS_FRAC = 0.6
TARGET_VOL = 0.40          # annualized, for variant B/C
VOL_WIN = 30 * 24          # 30d of hourly bars
BAND = 0.15                # variant D no-trade band (15pp)
_PERIOD_MS = {"1D": 86_400_000, "1d": 86_400_000, "4h": 14_400_000, "1h": 3_600_000}


def load_regime_conf(path: str, symbol: str) -> list:
    """[(effective_ts, regime, confidence)] with the +1-bar causal shift."""
    sym = symbol.replace("/", "_")
    rows = []
    for r in _csv.DictReader(open(path)):
        if r.get("symbol") not in (sym, symbol) or not r.get("regime") or not r.get("ts"):
            continue
        ts = int(r["ts"]) + _PERIOD_MS.get(r.get("timeframe", "1D"), 86_400_000)
        try:
            conf = float(r.get("confidence") or 0.0)
        except ValueError:
            conf = 0.0
        rows.append((ts, r["regime"], conf))
    rows.sort()
    return rows


def per_bar(rows: list, ts_list: List[int]) -> tuple:
    """(regime[], conf[]) aligned to bars, latest row strictly before each bar."""
    keys = [r[0] for r in rows]
    reg, conf = [], []
    for t in ts_list:
        i = bisect.bisect_left(keys, t)
        if i > 0:
            reg.append(rows[i - 1][1]); conf.append(rows[i - 1][2])
        else:
            reg.append("range"); conf.append(0.0)
    return reg, np.array(conf)


def realized_vol(closes: np.ndarray) -> np.ndarray:
    """Trailing 30d annualized vol per bar (uses only past bars)."""
    rets = np.diff(np.log(np.maximum(closes, 1e-12)), prepend=np.log(closes[0]))
    out = np.full(len(closes), np.nan)
    for i in range(VOL_WIN, len(closes)):
        out[i] = rets[i - VOL_WIN:i].std() * math.sqrt(8760)
    return out


def exposures(variant: str, regime: List[str], conf: np.ndarray,
              rvol: np.ndarray, frame: str) -> np.ndarray:
    """Target exposure in [0,1] per bar for one coin. frame: 'TR' or 'DL'."""
    n = len(regime)
    base = np.zeros(n)
    for i, r in enumerate(regime):
        if frame == "TR":
            base[i] = 1.0 if r in CONFIRM else 0.0
        else:
            base[i] = 0.0 if r in DAMAGE else 1.0
    if variant == "baseline":
        return base
    if variant in ("A", "C"):
        scaled = np.zeros(n)
        for i, r in enumerate(regime):
            if frame == "TR":
                scaled[i] = conf[i] if r in CONFIRM else 0.0
            else:
                scaled[i] = (1.0 - conf[i]) if r in DAMAGE else 1.0
        base = scaled
    if variant in ("B", "C"):
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.minimum(1.0, TARGET_VOL / rvol)
        scale = np.where(np.isfinite(scale), scale, 1.0)
        base = base * scale
    return np.clip(base, 0.0, 1.0)


def run_curve(closes: np.ndarray, expo: np.ndarray) -> np.ndarray:
    """Equity for a [0,1] exposure path; |delta expo| pays FLIP_FEE pro-rata."""
    eq = np.ones(len(closes))
    val, pos = 1.0, 0.0
    for i in range(1, len(closes)):
        ret = closes[i] / closes[i - 1] - 1.0
        val *= 1.0 + pos * ret
        want = float(expo[i])
        if abs(want - pos) > 1e-9:
            val *= 1.0 - FLIP_FEE * abs(want - pos)
            pos = want
        eq[i] = val
    return eq


def seg(eq: np.ndarray) -> dict:
    ret = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(ret) < 2 or ret.std() == 0:
        return {"total": float(eq[-1] / eq[0] - 1), "sharpe": 0.0, "maxdd": 0.0}
    cum = eq / eq[0]
    dd = float((np.maximum.accumulate(cum) - cum).max() / np.maximum.accumulate(cum).max())
    return {"total": float(eq[-1] / eq[0] - 1),
            "sharpe": float(ret.mean() / ret.std() * math.sqrt(8760)), "maxdd": dd}


def flips(expo: np.ndarray) -> int:
    return int(np.sum(np.abs(np.diff(expo)) > 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-registered sleeve consumption study")
    ap.add_argument("--days", type=int, default=1650)
    ap.add_argument("--regime-csv", default="data_cache/analyst_lean.csv")
    args = ap.parse_args()
    if not os.path.exists(args.regime_csv):
        print(f"{args.regime_csv} not found"); return 1

    data = []
    for sym in SYMBOLS:
        try:
            c = fetch_ohlcv(sym, "1h", args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: fetch failed ({str(exc)[:50]}) — skip"); continue
        if len(c) < 2000:
            continue
        ts = [int(x[0]) for x in c]
        closes = np.array([x[4] for x in c], float)
        rows = load_regime_conf(args.regime_csv, sym)
        if not rows:
            print(f"  {sym}: no regime rows — skip"); continue
        regime, conf = per_bar(rows, ts)
        data.append((sym, closes, regime, conf, realized_vol(closes)))
    if not data:
        print("no usable symbols"); return 1

    n = len(data[0][1])
    split = int(n * IS_FRAC)
    variants = ["baseline", "A", "B", "C"]
    frames = [("TR", "trend-rider"), ("DL", "default-long")]

    print(f"\n=== SLEEVE CONSUMPTION STUDY — {len(data)} coins · IS {IS_FRAC*100:.0f}% / OOS rest · "
          f"pre-registered variants only ===")
    for fr, fr_name in frames:
        print(f"\n--- {fr_name} frame ---")
        print(f"  {'variant':<9} | {'OOS Sh(med)':>11} {'OOS tot(med)':>12} {'OOS DD(med)':>11} "
              f"{'flips(med)':>10} {'coins>base':>10} | basket: Sh / tot / DD")
        base_oos: Dict[str, dict] = {}
        for v in variants:
            per_sh, per_tot, per_dd, per_fl, beats = [], [], [], [], 0
            basket = None
            for (sym, closes, regime, conf, rvol) in data:
                ex = exposures(v, regime, conf, rvol, fr)
                eq = run_curve(closes, ex)
                m = seg(eq[split:])
                per_sh.append(m["sharpe"]); per_tot.append(m["total"])
                per_dd.append(m["maxdd"]); per_fl.append(flips(ex[split:]))
                if v != "baseline":
                    b = base_oos[sym]
                    if m["sharpe"] > b["sharpe"] and m["maxdd"] <= b["maxdd"] + 1e-9:
                        beats += 1
                else:
                    base_oos[sym] = m
                # equal-weight basket: average per-bar exposure-returns
                r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
                basket = r if basket is None else basket + r
            basket = basket / len(data)
            beq = np.concatenate([[1.0], np.cumprod(1 + basket)])
            bm = seg(beq[split:])
            med = lambda xs: float(np.median(xs))
            tag = "" if v == "baseline" else f" {beats}/{len(data)}"
            print(f"  {v:<9} | {med(per_sh):>11.2f} {med(per_tot)*100:>+11.1f}% {med(per_dd)*100:>10.0f}% "
                  f"{med(per_fl):>10.0f} {tag:>10} | {bm['sharpe']:.2f} / {bm['total']*100:+.1f}% / {bm['maxdd']*100:.0f}%")

        # Variant D: portfolio-level (TR frame only — share of confirmed coins)
        if fr == "TR":
            share = np.zeros(n)
            for (sym, closes, regime, conf, rvol) in data:
                share += np.array([1.0 if r in CONFIRM else 0.0 for r in regime])
            share /= len(data)
            target = np.zeros(n)
            cur = 0.0
            for i in range(n):
                if abs(share[i] - cur) > BAND:
                    cur = share[i]
                target[i] = cur
            # apply the banded exposure to the equal-weight basket price path
            avg_ret = None
            for (sym, closes, regime, conf, rvol) in data:
                r = np.diff(np.log(np.maximum(closes, 1e-12)))
                avg_ret = r if avg_ret is None else avg_ret + r
            avg_ret = np.expm1(avg_ret / len(data))
            eq = np.ones(n); val, pos = 1.0, 0.0
            for i in range(1, n):
                val *= 1.0 + pos * avg_ret[i - 1]
                if abs(target[i] - pos) > 1e-9:
                    val *= 1.0 - FLIP_FEE * abs(target[i] - pos)
                    pos = target[i]
                eq[i] = val
            dm = seg(eq[split:])
            print(f"  {'D(port)':<9} | {'—':>11} {'—':>12} {'—':>11} {flips(target[split:]):>10} "
                  f"{'—':>10} | {dm['sharpe']:.2f} / {dm['total']*100:+.1f}% / {dm['maxdd']*100:.0f}%")

    print("\n=== read (pre-registered bar) ===")
    print("A variant wins ONLY if it beats baseline OOS Sharpe with maxDD no worse, on the")
    print("basket AND >=4/7 coins. Otherwise the baseline stays — it is simpler and already")
    print("live-validated. A winner becomes a THIRD shadow sleeve after the Aug 1 read;")
    print("nothing about the running shadow test changes before then.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""analyst_swing_backtest.py — trade the ANALYST'S OWN signal directly (not as an
intraday gate): go long when its conviction is strongly positive, short when strongly
negative, flatten when it decays back toward neutral. "Buy/sell when strong, stop on
reversal."

Why this is a different (and fairer) test than the gate experiments:
  * it trades the FULL period, holding for days and flipping only on regime change, so
    per-trade cost is amortized over long holds — the slow, structural category that can
    actually clear the cost wall (unlike gated intraday, which shrank to ~400 noisy trades);
  * it tests the analyst exactly where its model card shows a measured edge — its own
    1D/4h directional calls.

Signal (pick with --signal):
  * normalized  — the ensemble conviction in [-1, 1]  (needs an export WITH the ensemble)
  * score       — raw ensemble score
  * regime      — +1 trend_up / -1 trend_down / 0 else (works on a labels-only export)

Position: enter when |signal| >= --enter, hold while it stays beyond --exit in the same
direction, flatten when it falls back inside --exit, flip if it crosses to the other side.
Strictly causal: position at bar t uses signal[t]; P&L is earned on bar t+1's return.

Controls printed every run:
  * BUY & HOLD benchmark (same vol scale)
  * PLACEBO: block-shuffle the signal (preserve its persistence, destroy its alignment to
    returns) N times. If the real strategy doesn't beat the placebo cloud, the analyst's
    *timing* adds nothing.

    python analyst_swing_backtest.py --lean-csv data_cache/analyst_lean.csv \
        --symbols BTC_USDT,ETH_USDT --signal normalized --enter 0.3 --exit 0.1 --placebo 50
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

BARS_PER_YEAR = {"1D": 365.0, "4h": 365.0 * 6, "1h": 365.0 * 24}


def read_csv(path: str, gate_tf: str) -> Dict[str, List[dict]]:
    per: Dict[str, List[dict]] = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r.get("timeframe") != gate_tf:
            continue
        if not r.get("close") or not r.get("ts"):
            continue
        per[r["symbol"]].append({
            "ts": int(r["ts"]), "close": float(r["close"]),
            "score": float(r.get("score") or 0.0),
            "normalized": float(r.get("normalized") or 0.0),
            "regime": r.get("regime") or "",
        })
    for s in per:
        per[s].sort(key=lambda x: x["ts"])
    return per


def signal_series(rows: List[dict], which: str) -> np.ndarray:
    if which == "regime":
        return np.array([1.0 if x["regime"] == "trend_up" else -1.0 if x["regime"] == "trend_down"
                         else 0.0 for x in rows])
    return np.array([x[which] for x in rows], float)


def positions(sig: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    pos = np.zeros(len(sig))
    cur = 0.0
    for t in range(len(sig)):
        s = sig[t]
        if cur == 0.0:
            if s >= enter:
                cur = 1.0
            elif s <= -enter:
                cur = -1.0
        elif cur > 0:
            if s <= -enter:
                cur = -1.0
            elif s < exit_:
                cur = 0.0
        else:  # cur < 0
            if s >= enter:
                cur = 1.0
            elif s > -exit_:
                cur = 0.0
        pos[t] = cur
    return pos


def pnl_from_positions(pos: np.ndarray, ret_next: np.ndarray, fee: float):
    """Per-bar net return = pos[t]*ret[t+1] - fee*|pos[t]-pos[t-1]|. Last bar dropped."""
    n = len(pos)
    out = np.zeros(n)
    prev = 0.0
    trades = 0
    for t in range(n - 1):
        turn = abs(pos[t] - prev)
        if turn > 0:
            trades += 1
        out[t] = pos[t] * ret_next[t] - fee * turn
        prev = pos[t]
    return out[:-1], trades


def metrics(daily: np.ndarray, tf: str) -> dict:
    d = daily[np.isfinite(daily)]
    if len(d) == 0 or d.std() == 0:
        return {"sharpe": 0.0, "total": float(d.sum()), "maxdd": 0.0}
    cum = np.cumsum(d)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    return {"sharpe": float(d.mean() / d.std() * math.sqrt(BARS_PER_YEAR.get(tf, 365))),
            "total": float(d.sum()), "maxdd": dd}


def run_symbol(rows: List[dict], which: str, enter: float, exit_: float, fee: float):
    close = np.array([x["close"] for x in rows], float)
    ret_next = np.concatenate([close[1:] / close[:-1] - 1.0, [0.0]])
    sig = signal_series(rows, which)
    pos = positions(sig, enter, exit_)
    pnl, trades = pnl_from_positions(pos, ret_next, fee)
    return pnl, ret_next[:-1], trades, [x["ts"] for x in rows[:-1]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Trade the analyst signal directly (swing)")
    ap.add_argument("--lean-csv", default="data_cache/analyst_lean.csv")
    ap.add_argument("--gate-tf", default="1D")
    ap.add_argument("--symbols", default="BTC_USDT,ETH_USDT")
    ap.add_argument("--signal", default="normalized", choices=["normalized", "score", "regime"])
    ap.add_argument("--enter", type=float, default=0.3, help="|signal| to OPEN a position")
    ap.add_argument("--exit", dest="exit_", type=float, default=0.1, help="|signal| to CLOSE back to flat")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--placebo", type=int, default=50)
    ap.add_argument("--block", type=int, default=20, help="placebo block length (preserve persistence)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0
    per = read_csv(args.lean_csv, args.gate_tf)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip() and s.strip() in per]
    if not syms:
        print(f"no matching symbols/timeframe in {args.lean_csv}"); return 1

    print(f"=== ANALYST SWING — {args.signal} signal, enter |{args.enter}| / exit |{args.exit_}|, "
          f"{args.gate_tf}, {args.fee_bps:g}bps ===\n")
    print(f"{'symbol':<10}{'bars':>6}{'trades':>7}{'Sharpe':>8}{'total':>9}{'maxDD':>8}"
          f"{'OOSsh':>7}{'B&H sh':>8}{'plcb>=':>8}")
    print("-" * 72)

    random.seed(args.seed)
    agg = []
    for s in syms:
        rows = per[s]
        if len(rows) < 200:
            print(f"{s:<10} only {len(rows)} bars — skip"); continue
        pnl, ret, trades, ts = run_symbol(rows, args.signal, args.enter, args.exit_, fee)
        m = metrics(pnl, args.gate_tf)
        cut = len(pnl) // 2
        oos = metrics(pnl[cut:], args.gate_tf)
        bh = metrics(ret, args.gate_tf)  # buy & hold (always long)

        # placebo: block-shuffle the signal, keep its persistence, destroy alignment
        sig = signal_series(rows, args.signal)
        ge = 0
        real_oos = oos["sharpe"]
        for _ in range(args.placebo):
            blocks = [sig[i:i + args.block] for i in range(0, len(sig), args.block)]
            random.shuffle(blocks)
            psig = np.concatenate(blocks)[:len(sig)]
            ppos = positions(psig, args.enter, args.exit_)
            close = np.array([x["close"] for x in rows], float)
            rn = np.concatenate([close[1:] / close[:-1] - 1.0, [0.0]])
            ppnl, _ = pnl_from_positions(ppos, rn, fee)
            if metrics(ppnl[len(ppnl) // 2:], args.gate_tf)["sharpe"] >= real_oos:
                ge += 1
        agg.append({"sym": s, "oos": oos["sharpe"], "bh": bh["sharpe"], "ge": ge})
        print(f"{s:<10}{len(rows):>6}{trades:>7}{m['sharpe']:>8.2f}{m['total']*100:>+8.1f}%"
              f"{m['maxdd']*100:>7.1f}%{oos['sharpe']:>7.2f}{bh['sharpe']:>8.2f}{ge:>6}/{args.placebo}")

    if agg:
        mean_oos = float(np.mean([a["oos"] for a in agg]))
        beat_bh = sum(1 for a in agg if a["oos"] > a["bh"])
        passed = sum(1 for a in agg if a["oos"] > 0 and a["oos"] > a["bh"]
                     and a["ge"] <= max(1, int(0.10 * args.placebo)))
        n = len(agg)
        print("-" * 72)
        print(f"AGGREGATE ({n} coins): mean OOS Sharpe {mean_oos:+.2f} · beat buy&hold {beat_bh}/{n} · "
              f"clean pass (OOS>0, >B&H, placebo<=10%) {passed}/{n}")
        print("\n=== read ===")
        if passed >= max(2, int(0.6 * n)) and mean_oos > 0:
            print(f"BREADTH HOLDS: {passed}/{n} coins pass cleanly and the average OOS Sharpe is "
                  f"{mean_oos:+.2f} — a real edge shows up BROADLY, not in one coin. This is the first "
                  f"thing in the project to survive breadth + placebo. Worth vol-targeting and forward-testing.")
        else:
            print(f"NOT BREADTH: only {passed}/{n} coins pass cleanly (mean OOS {mean_oos:+.2f}). The "
                  f"signal doesn't generalize across coins — whatever looked good is coin-specific luck, "
                  f"not a broad edge. Don't trade it.")
        return 0

    print("\n=== read ===")
    print("Real edge needs: full-sample AND OOS Sharpe positive, beating buy & hold, AND the")
    print("placebo count (plcb>=) LOW (<~5/50). If placebos routinely match it, the analyst's")
    print("timing adds nothing over a persistence-matched random signal. Sweep --enter/--exit,")
    print("but judge on OOS + placebo, never the in-sample total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

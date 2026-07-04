"""tsm_split_study.py — is 08:00 UTC actually special, or does session momentum
work at ANY entry hour?

The question (Ricardo, 2026-07-03): "intraday only looks at a specific hour — I need
it to be more free, it's not like a specific hour every day is the best time to enter."

The honest version of that wish is a TEST, not a live tweak. This generalizes the
TSM structure to a rolling session: at entry hour H, the signal is the return over
the PRIOR 8 hours [H-8 -> H], the trade holds the NEXT 16 hours [H -> H+16], with the
same trailing gates as the real strategy (60d top-tertile |signal| vol gate, 30d
signal->outcome autocorr regime gate) and the real 20 bps round-trip.

READ THE RESULT LIKE THIS (pre-registered):
  * Signal broad across hours (most H green OOS)  -> the edge is "session momentum,"
    not "08:00" — a multi-hour version is justified and can go to a shadow forward
    test. More entry hours = more trades = faster evidence, too.
  * ONLY 08:00 green                              -> 08:00 was likely a lucky pick
    (curve-fit warning for the CURRENT live config, not license to switch hours).
  * Nothing green                                 -> matches the 5-year backtest
    verdict (the 1yr profit was small-sample); the live forward test remains the
    only scorecard.
What this is NOT: permission to hop to whatever hour looks best this month — that's
the Runs A-D mining trap with a clock.

    python tsm_split_study.py --days 1850
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from consensus_backtest import fetch_ohlcv
from costs import round_trip_bps

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
           "DOGE/USDT", "LINK/USDT", "AVAX/USDT"]
HOURS = [0, 4, 8, 12, 16, 20]     # entry hours to test (8 = the live config's slot)
SIG_H = 8                          # signal window: prior 8 hours
HOLD_H = 16                        # hold window: next 16 hours
VOL_WINDOW = 60                    # trailing days for the top-tertile vol gate
VOL_Q = 0.667
REGIME_WINDOW = 30                 # trailing days for the autocorr regime gate
FEE = round_trip_bps() / 1e4       # 20 bps taker/taker + slippage, from costs.py
IS_FRAC = 0.6


def sessions(candles: List[list], entry_hour: int) -> List[Tuple[str, float, float]]:
    """[(day, signal_ret, outcome_ret)] for one entry hour, from hourly candles.
    signal = close[H-1] / close[H-1-SIG_H] - 1 ; outcome = close[H-1+HOLD_H] / close[H-1] - 1
    (all closes of completed bars — fully causal at H:00)."""
    by_ts = {int(b[0]) // 3600_000: float(b[4]) for b in candles}   # hour-index -> close
    out = []
    hours_sorted = sorted(by_ts)
    for h in hours_sorted:
        dt = datetime.fromtimestamp(h * 3600, tz=timezone.utc)
        if dt.hour != (entry_hour - 1) % 24:      # bar whose CLOSE lands on entry_hour
            continue
        h0, h1 = h - SIG_H, h + HOLD_H
        if h0 not in by_ts or h1 not in by_ts:
            continue
        sig = by_ts[h] / by_ts[h0] - 1.0
        outc = by_ts[h1] / by_ts[h] - 1.0
        out.append((dt.strftime("%Y-%m-%d"), sig, outc))
    return out


def gated_trades(samps: List[Tuple[str, float, float]]) -> List[Tuple[str, float]]:
    """Apply the real strategy's trailing gates; return [(day, net_ret)]."""
    out = []
    hist: List[Tuple[float, float]] = []       # (signal, outcome) history, prior days
    for day, sig, outc in samps:
        if len(hist) >= max(VOL_WINDOW, REGIME_WINDOW):
            thr = float(np.quantile([abs(s) for s, _ in hist[-VOL_WINDOW:]], VOL_Q))
            rec = hist[-REGIME_WINDOW:]
            ss = np.array([s for s, _ in rec]); oo = np.array([o for _, o in rec])
            regime_ok = ss.std() > 0 and oo.std() > 0 and np.corrcoef(ss, oo)[0, 1] > 0
            if thr > 0 and abs(sig) >= thr and regime_ok:
                direction = 1.0 if sig > 0 else -1.0
                out.append((day, direction * outc - FEE))
        hist.append((sig, outc))
    return out


def stats(tr: List[Tuple[str, float]]) -> dict:
    if not tr:
        return {"n": 0, "win": 0.0, "avg": 0.0, "tot": 0.0}
    a = np.array([x[1] for x in tr])
    return {"n": len(a), "win": float((a > 0).mean()), "avg": float(a.mean()), "tot": float(a.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description="Does session momentum depend on the entry hour?")
    ap.add_argument("--days", type=int, default=1850)
    args = ap.parse_args()

    data: Dict[str, List[list]] = {}
    for sym in SYMBOLS:
        try:
            data[sym] = fetch_ohlcv(sym, "1h", args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: fetch failed ({str(exc)[:60]}) — skip")
    if not data:
        print("no usable symbols"); return 1

    all_days = sorted({datetime.fromtimestamp(int(b[0]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                       for c in data.values() for b in c})
    split_day = all_days[int(len(all_days) * IS_FRAC)]
    print(f"\n=== TSM SPLIT-HOUR STUDY — signal {SIG_H}h -> hold {HOLD_H}h, gates as live, "
          f"{FEE*1e4:.0f} bps · {all_days[0]} -> {all_days[-1]} · IS < {split_day} <= OOS ===\n")
    print(f"  {'H':>3} | {'IS n':>5} {'IS net/tr':>10} | {'OOS n':>5} {'OOS win':>7} "
          f"{'OOS net/tr':>10} {'OOS total':>9} {'breadth':>8}")
    verdict_rows = []
    for H in HOURS:
        p_is, p_oos, breadth = [], [], 0
        for sym, c in data.items():
            tr = gated_trades(sessions(c, H))
            t_is = [t for t in tr if t[0] < split_day]
            t_oos = [t for t in tr if t[0] >= split_day]
            p_is += t_is; p_oos += t_oos
            if t_oos and stats(t_oos)["tot"] > 0:
                breadth += 1
        si, so = stats(p_is), stats(p_oos)
        tag = " <- live slot" if H == 8 else ""
        print(f"  {H:>3} | {si['n']:>5} {si['avg']*100:>+9.3f}% | {so['n']:>5} {so['win']*100:>6.0f}% "
              f"{so['avg']*100:>+9.3f}% {so['tot']*100:>+8.1f}% {breadth:>5}/{len(data)}{tag}")
        verdict_rows.append((H, so["avg"], so["tot"], breadth))

    green = [(h, t) for h, avg, t, b in verdict_rows if t > 0 and b > len(data) / 2]
    print("\n=== read (pre-registered) ===")
    if len(green) >= 4:
        print(f"BROAD: {len(green)}/{len(HOURS)} hours net-positive OOS with breadth — the edge is")
        print("session momentum itself, not 08:00. A multi-hour ensemble is justified: next step is")
        print("a shadow forward test logging ALL hours' decisions (same tsm_forward pattern).")
    elif any(h == 8 for h, _ in green) and len(green) <= 2:
        print("NARROW: 08:00 (± one neighbor) is the only green — treat the live config's hour as")
        print("possibly lucky. Do NOT hop hours; weigh this as evidence when judging the forward test.")
    elif green:
        print(f"MIXED: green at {[h for h, _ in green]} but not broad. No action — evidence noted.")
    else:
        print("NOTHING green OOS — consistent with the failed 5-year backtest. The live forward")
        print("test remains the only scorecard; 'more freedom' would just add more losing trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

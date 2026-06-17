"""intraday_tsm_backtest.py — the one intraday edge the literature says survives costs.

Source of the idea: Shen, Urquhart & Wang, "Bitcoin intraday time-series momentum"
(Financial Review, 2022). Their finding: the EARLY part of the session's return
positively predicts the LATE part of the session, and the strategy stays positive
AFTER fees — strongest on high-volume / high-volatility days. Mechanism: liquidity
provision, not informed trading. It's ONE trade per day, so fees barely bite — the
opposite of the scalp frequency where costs always won.

It also lines up with the realistic-cost crypto-momentum literature (Han/Kang/Ryu
2023): TIME-SERIES momentum (trend on a coin's OWN return) survives; CROSS-SECTIONAL
/ reversion (what pairs was) does not; and the edge is concentrated in WINNERS.

This test, honestly:
  * Split each UTC day at hour S. "morning" = open -> hour S. "afternoon" = hour S -> close.
  * Signal = sign(morning return). Trade the afternoon in that direction. Exit at day close.
  * One round-trip per day per coin, fee charged once.
  * --mode tsm     : long if morning up, short if morning down (full TSM)
    --mode winners : long only when morning up, else flat (the winner-concentration finding)
  * Reports the PREDICTIVE core first (corr of afternoon on morning return) — that's
    the edge independent of any trading rule — then net-of-fee P&L by month and coin.
  * Optional --vol-gate: only trade days whose morning move is in the top tertile of
    |morning return| (the "high-volatility session" condition the paper highlights).

    .venv\\Scripts\\python.exe intraday_tsm_backtest.py --timeframe 1h --days 360 --split 8 --fee-bps 10
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from consensus_backtest import DEFAULT_SYMBOLS, fetch_ohlcv


def _day_key(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _hour(ts_ms: int) -> int:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour


def day_samples(candles: List[List[float]], split_hour: int) -> List[Tuple[str, float, float]]:
    """Return [(day, morning_ret, afternoon_ret)] for days with usable coverage.

    morning   = open of first bar  -> close of the last bar at/under split_hour
    afternoon = that close          -> close of the last bar of the day
    """
    by_day: Dict[str, List[List[float]]] = defaultdict(list)
    for b in candles:
        by_day[_day_key(int(b[0]))].append(b)
    out: List[Tuple[str, float, float]] = []
    for day, bars in by_day.items():
        bars.sort(key=lambda b: b[0])
        if len(bars) < 6:
            continue
        open_px = bars[0][1]                       # first bar open
        # split point: last bar whose hour < split_hour
        pre = [b for b in bars if _hour(int(b[0])) < split_hour]
        post = [b for b in bars if _hour(int(b[0])) >= split_hour]
        if not pre or not post:
            continue
        split_px = pre[-1][4]                       # close at the split
        close_px = bars[-1][4]                      # day close
        if open_px <= 0 or split_px <= 0:
            continue
        out.append((day, split_px / open_px - 1.0, close_px / split_px - 1.0))
    return sorted(out, key=lambda x: x[0])   # chronological


def predictive_corr(samples: List[Tuple[str, float, float]]) -> float:
    """Rule-free edge: correlation of afternoon return on morning return."""
    if len(samples) < 20:
        return 0.0
    morns = np.array([m for _d, m, _a in samples])
    afts = np.array([a for _d, _m, a in samples])
    if morns.std() == 0 or afts.std() == 0:
        return 0.0
    return float(np.corrcoef(morns, afts)[0, 1])


def gen_trades(samples: List[Tuple[str, float, float]], *, mode: str, fee: float,
               vol_gate: bool, vol_window: int, vol_q: float) -> List[Tuple[str, float]]:
    """Chronological per-trade (day, net_ret).

    The vol gate uses a TRAILING window quantile (only days BEFORE the current one),
    so there is no lookahead — you could have computed the threshold live.
    """
    abs_m = [abs(m) for _d, m, _a in samples]
    out: List[Tuple[str, float]] = []
    for i, (day, m, a) in enumerate(samples):
        if vol_gate:
            if i < vol_window:
                continue                                  # warm-up: no trailing history yet
            thr = float(np.quantile(abs_m[i - vol_window:i], vol_q))
            if abs(m) < thr:
                continue
        if mode == "winners":
            if m <= 0:
                continue
            pos = 1.0
        else:
            pos = 1.0 if m > 0 else -1.0
        out.append((day, pos * a - fee))
    return out


def _stat(rets: List[float]) -> Tuple[int, float, float, float]:
    if not rets:
        return 0, 0.0, 0.0, 0.0
    arr = np.array(rets)
    return len(arr), float((arr > 0).mean()), float(arr.mean()), float(arr.sum())


def _pool_oos(trades_by_coin: Dict[str, List[Tuple[str, float]]]):
    """Return (pooled_rets, oos_rets, oos_pos_coins, oos_n_coins)."""
    allt = [(d, r) for ts in trades_by_coin.values() for (d, r) in ts]
    if not allt:
        return [], [], 0, 0
    days = sorted({d for d, _r in allt})
    cut = days[len(days) // 2]
    pooled = [r for _d, r in allt]
    oos = [r for d, r in allt if d >= cut]
    op = on = 0
    for ts in trades_by_coin.values():
        o = [r for d, r in ts if d >= cut]
        if len(o) >= 10:
            on += 1
            op += 1 if np.mean(o) > 0 else 0
    return pooled, oos, op, on


def _sweep(symbols, args, gk) -> int:
    """Fetch once, test the edge across a range of session splits — the cherry-pick guard."""
    splits = [int(x) for x in args.sweep_splits.split(",")]
    candles: Dict[str, List[List[float]]] = {}
    for s in symbols:
        try:
            candles[s] = fetch_ohlcv(s, args.timeframe, args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"{s.split('/')[0]}: fetch failed ({str(exc)[:30]})")
    print(f"\n=== INTRADAY TSM SPLIT SWEEP — mode={args.mode}"
          f"{', vol-gated' if args.vol_gate else ''}, {args.fee_bps:g}bps/day, {len(candles)} coins ===")
    print("a real session effect survives a RANGE of splits; one lucky hour = cherry-pick\n")
    print(f"{'split':>6}{'trades':>8}{'avgcorr':>9}{'pooled/day':>12}{'OOS/day':>10}{'OOScoins':>10}")
    good = 0
    for sp in splits:
        tbc: Dict[str, List[Tuple[str, float]]] = {}
        corrs = []
        for s, c in candles.items():
            samp = day_samples(c, sp)
            if len(samp) < 20:
                continue
            tr = gen_trades(samp, **gk)
            if tr:
                tbc[s.split("/")[0]] = tr
                corrs.append(predictive_corr(samp))
        if not tbc:
            print(f"{sp:>6}   (no trades)")
            continue
        pooled, oos, op, on = _pool_oos(tbc)
        ac = float(np.mean(corrs)) if corrs else 0.0
        flag = "  <-- OOS+ & broad" if (oos and np.mean(oos) > 0 and on and op >= 0.6 * on) else ""
        if flag:
            good += 1
        print(f"{sp:>6}{len(pooled):>8}{ac:>+9.3f}{np.mean(pooled)*100:>+11.3f}%"
              f"{np.mean(oos)*100:>+9.3f}%{op:>6}/{on}{flag}")
    print("\n=== read ===")
    if good >= max(2, len(splits) // 2):
        print(f"the edge holds at {good}/{len(splits)} session splits — it's a ROBUST session effect, "
              f"not a lucky hour. That's a real cherry-pick-proof intraday edge. Live-forward it.")
    elif good >= 1:
        print(f"only {good}/{len(splits)} splits work — the effect is real but session-specific/fragile. "
              f"Trust it only at the split(s) that survive, and live-forward small.")
    else:
        print("no split survives OOS broadly — the single-split green was likely a cherry-pick. Honest re-think.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Intraday time-series momentum backtest (hardened)")
    ap.add_argument("--days", type=int, default=360)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--split", type=int, default=8, help="UTC hour splitting morning/afternoon")
    ap.add_argument("--mode", choices=["tsm", "winners"], default="tsm")
    ap.add_argument("--vol-gate", action="store_true", help="only trade days whose |morning move| clears the trailing quantile")
    ap.add_argument("--vol-window", type=int, default=60, help="trailing days for the vol-gate threshold (no lookahead)")
    ap.add_argument("--vol-q", type=float, default=0.667, help="trailing quantile to clear (0.667 = top tertile)")
    ap.add_argument("--fee-bps", type=float, default=10.0, help="round-trip cost per day-trade")
    ap.add_argument("--sweep", action="store_true", help="test a range of session splits at once (robustness)")
    ap.add_argument("--sweep-splits", default="0,4,6,8,10,12,16,20", help="UTC hours to sweep")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    gk = dict(mode=args.mode, fee=fee, vol_gate=args.vol_gate,
              vol_window=args.vol_window, vol_q=args.vol_q)

    if args.sweep:
        return _sweep(symbols, args, gk)

    print(f"\n=== INTRADAY TSM (HARDENED) — split {args.split:02d}:00 UTC, mode={args.mode}"
          f"{f', trailing vol-gate q{args.vol_q:g}/{args.vol_window}d' if args.vol_gate else ''}, "
          f"{args.fee_bps:g}bps/day ===")
    print("trailing gate = NO lookahead; IS/OOS = first vs second half of the calendar\n")
    print(f"{'coin':<7}{'days':>6}{'corr':>8}{'win%':>7}{'net/day':>10}{'total':>9}")

    trades_by_coin: Dict[str, List[Tuple[str, float]]] = {}
    by_month: Dict[str, List[float]] = defaultdict(list)
    corrs: List[float] = []
    for s in symbols:
        try:
            c = fetch_ohlcv(s, args.timeframe, args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"{s.split('/')[0]:<7} fetch failed ({str(exc)[:30]})"); continue
        samples = day_samples(c, args.split)
        if len(samples) < 20:
            continue
        coin = s.split("/")[0]
        corr = predictive_corr(samples)
        trades = gen_trades(samples, **gk)
        if not trades:
            continue
        corrs.append(corr)
        trades_by_coin[coin] = trades
        n, win, exp, tot = _stat([r for _d, r in trades])
        print(f"{coin:<7}{n:>6}{corr:>+8.3f}{win*100:>6.0f}%{exp*100:>+9.3f}%{tot*100:>+8.1f}%")
        for day, ret in trades:
            by_month[day[:7]].append(ret)

    if not trades_by_coin:
        print("\nno tradeable days — loosen filters or add coins"); return 1

    all_trades = [(d, r) for ts in trades_by_coin.values() for (d, r) in ts]
    all_days = sorted({d for d, _r in all_trades})
    cutoff = all_days[len(all_days) // 2]                  # calendar midpoint
    avg_corr = float(np.mean(corrs)) if corrs else 0.0

    pooled = [r for _d, r in all_trades]
    is_rets = [r for d, r in all_trades if d < cutoff]
    oos_rets = [r for d, r in all_trades if d >= cutoff]
    pn, pw, pe, ptot = _stat(pooled)
    in_, iw, ie, itot = _stat(is_rets)
    on, ow, oe, otot = _stat(oos_rets)

    print(f"\nPOOLED: {pn} trades, avg predictive corr {avg_corr:+.3f}, "
          f"win {pw*100:.0f}%, net/day {pe*100:+.3f}%, total {ptot*100:+.1f}%")
    print(f"IN-SAMPLE  (< {cutoff}): n={in_:<5} win {iw*100:.0f}%  net/day {ie*100:+.3f}%  total {itot*100:+.1f}%")
    print(f"OUT-SAMPLE (>= {cutoff}): n={on:<5} win {ow*100:.0f}%  net/day {oe*100:+.3f}%  total {otot*100:+.1f}%")

    # per-coin OOS breadth (the honest persistence check)
    oos_pos = oos_n = 0
    for coin, ts in trades_by_coin.items():
        o = [r for d, r in ts if d >= cutoff]
        if len(o) >= 10:
            oos_n += 1
            oos_pos += 1 if np.mean(o) > 0 else 0
    print(f"OOS coins net-positive: {oos_pos}/{oos_n}\n")

    print("BY MONTH (net of fees):")
    pos_m = 0
    months = sorted(by_month)
    for ym in months:
        v = np.array(by_month[ym])
        pos_m += 1 if v.mean() > 0 else 0
        print(f"  {ym}   n={len(v):<4} net/day {v.mean()*100:>+7.3f}%  total {v.sum()*100:>+7.1f}%")

    print("\n=== read ===")
    broad_oos = oos_n > 0 and oos_pos >= 0.6 * oos_n
    if oe > 0 and avg_corr > 0.03 and broad_oos and ie > 0:
        print(f"SURVIVES the honest test: predictive corr {avg_corr:+.3f}, and net of fees it pays "
              f"in BOTH halves (IS {ie*100:+.3f}%/day, OOS {oe*100:+.3f}%/day), broad out-of-sample "
              f"({oos_pos}/{oos_n} coins) — with a trailing gate, no lookahead. This is a real, "
              f"direction-free-of-prediction intraday edge. Next: live-forward it on testnet.")
    elif oe > 0 and avg_corr > 0:
        print(f"half-survives: positive OOS (+{oe*100:.3f}%/day) but thin/uneven "
              f"(corr {avg_corr:+.3f}, OOS coins {oos_pos}/{oos_n}, IS {ie*100:+.3f}%/day) — a real but "
              f"fragile, regime-dependent edge. The honest play is a regime filter (trade it only when "
              f"the momentum regime is on), not always-on. Sweep --split / --vol-q before trusting size.")
    else:
        print(f"does NOT survive the trailing gate + holdout (OOS {oe*100:+.3f}%/day, corr {avg_corr:+.3f}) — "
              f"the in-sample shine came partly from vol-gate lookahead. The predictability is real in sign "
              f"but too weak to harvest net of costs out-of-sample. Honest no on always-on; only a regime "
              f"filter could rescue it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

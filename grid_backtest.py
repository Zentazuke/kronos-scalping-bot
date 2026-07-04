"""grid_backtest.py — Phase 0: does a regime-gated, ATR-spaced grid beat frictions?

The honest test the two research reports demand. A plain static grid is ~zero EV before
fees and negative after; the only question worth asking is whether ADAPTATION — volatility
spacing + a regime gate that stands the grid down in trends — turns it into something real,
net of fees, ACROSS regimes, and surviving a trending stretch.

Model (per coin, on hourly candles — fine enough to capture oscillation):
  * Grid = independent cells between adjacent geometric levels, spacing = --step-pct, inside
    a band of +/- (range_mult * ATR%) around the (re)center price.
  * Cell 'empty' rests a buy at its low; on a touch it buys (maker fee) and becomes 'holding'
    with a resting sell at its high; on a touch there it sells (maker fee) -> a completed
    round-trip. Inventory = size * holding-cells; cash constraint is real (a downtrend drains
    cash and piles inventory — the trap).
  * REGIME GATE: when the (lagged, causal) regime is trend/flash, the grid STOPS — market-
    sells inventory (taker fee) and waits; when range returns it RE-CENTERS at the new price
    (DGT-style). Gate off = grid always on (the naive baseline that should bleed in trends).

Reports, per gate mode: total, Sharpe, maxDD, round-trips, fee load, and return split by
regime (range vs trend) + worst month, vs buy-and-hold. The bar: the gated grid must beat
no-gate AND not bleed in trend segments, net of fees.

Regime source: internal efficiency-ratio classifier by default; pass --regime-csv to gate on
the Crypto-Analyst's exported regime instead (causal, timestamp-lagged).

    python grid_backtest.py --symbols BTC/USDT,ETH/USDT --days 1400 --step-pct 0.006 --range-mult 6
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from consensus_backtest import fetch_ohlcv

TREND_REGIMES = {"trend_up", "trend_down", "breakout", "flash", "flash_risk"}


def atr_pct(c: List[list], n: int = 14) -> np.ndarray:
    """Wilder ATR as a fraction of close."""
    h = np.array([x[2] for x in c], float); l = np.array([x[3] for x in c], float)
    cl = np.array([x[4] for x in c], float)
    pc = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.zeros(len(c)); atr[:n] = tr[:n].mean() if len(tr) >= n else tr.mean()
    for i in range(n, len(c)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr / np.maximum(cl, 1e-9)


def efficiency_ratio(c: List[list], n: int = 24) -> np.ndarray:
    """Kaufman ER: |net move| / summed path over n bars. ~1 = strong trend, ~0 = chop."""
    cl = np.array([x[4] for x in c], float)
    er = np.zeros(len(cl))
    for i in range(n, len(cl)):
        net = abs(cl[i] - cl[i - n])
        path = np.abs(np.diff(cl[i - n:i + 1])).sum()
        er[i] = net / path if path > 0 else 0.0
    return er


def internal_regime(c: List[list], er_win: int, er_thr: float, ap: np.ndarray) -> List[str]:
    er = efficiency_ratio(c, er_win)
    cl = np.array([x[4] for x in c], float)
    reg = []
    for i in range(len(c)):
        # flash: a single-bar move > 3x ATR
        bar_move = abs(cl[i] / cl[i - 1] - 1.0) if i > 0 else 0.0
        if bar_move > 3.0 * ap[i]:
            reg.append("flash")
        elif er[i] >= er_thr:
            reg.append("trend_up" if cl[i] >= cl[i - er_win] else "trend_down")
        else:
            reg.append("range")
    return reg


def smooth_regime(regime: List[str], persist: int) -> List[str]:
    """Debounce/hysteresis: only flip the confirmed regime after `persist` consecutive raw
    bars agree on the new class (trend vs range). Stops the gate thrashing on a jumpy signal.
    A flash bar flips immediately (that's the whole point of a flash halt)."""
    conf: List[str] = []
    state = "range"
    run = 0
    last_cls: Optional[str] = None
    for r in regime:
        if r == "flash" or r == "flash_risk":
            state = r; run = 0; last_cls = "trend"; conf.append(state); continue
        cls = "trend" if r in TREND_REGIMES else "range"
        run = run + 1 if cls == last_cls else 1
        last_cls = cls
        state_cls = "trend" if state in TREND_REGIMES else "range"
        if run >= persist and cls != state_cls:
            state = r if cls == "trend" else "range"
        conf.append(state)
    return conf


def load_regime_csv(path: str, symbol: str) -> Optional[list]:
    """(ts_ms, regime) sorted, for causal lookup. symbol like BTC/USDT -> BTC_USDT."""
    sym = symbol.replace("/", "_")
    rows = []
    for r in csv.DictReader(open(path)):
        if r.get("symbol") in (sym, symbol) and r.get("regime"):
            ts = int(r["ts"]) if r.get("ts") else None
            if ts:
                rows.append((ts, r["regime"]))
    rows.sort()
    return rows or None


def regime_series(reg_rows: list, ts_list: List[int]) -> List[str]:
    """Latest regime strictly before each bar (causal). Keys built once (was O(n^2))."""
    keys = [r[0] for r in reg_rows]
    out = []
    for t in ts_list:
        i = bisect.bisect_left(keys, t)
        out.append(reg_rows[i - 1][1] if i > 0 else "range")
    return out


def lag1(regime: List[str]) -> List[str]:
    """AUDIT FIX (2026-07-03): the internal classifier labels bar i using bar i's own
    close (ER window and the flash test both end at cl[i]), but the sim was gating fills
    INSIDE bar i on that label — same-bar lookahead that flattered the gate (especially
    flash, which flips with no hysteresis). Gate bar i on the regime known at i-1's close."""
    return regime[:1] + regime[:-1]


def simulate(c: List[list], regime: List[str], *, step: float, range_mult: float,
             ap: np.ndarray, maker: float, taker: float, capital: float, standdown: set,
             freeze: bool = False, hedge: bool = False, stop_loss: float = 0.0,
             inv_cap: float = 0.0, cooldown: int = 24):
    """Cash+inventory grid sim. Returns (equity_curve, n_roundtrips, fee_paid).

    standdown regimes: 'exit' mode (freeze=False) liquidates + rebuilds on entry/return;
    'freeze' mode (freeze=True) pauses NEW buys but keeps resting sells working, never
    liquidates.
    'hedge' mode (hedge=True): on stand-down, pause the whole book and SHORT a perp for
    the inventory notional — delta-neutral instead of dumping (no realized loss) or riding
    the trend down (freeze's flaw). Unwind when range returns. Modeled conservatively:
    taker fee on both hedge legs, NO funding income credited (shorts often COLLECT funding
    in downtrends, so real results should be a touch better).

    Extra risk knobs (independent of the regime gate — the surgical tail-caps):
      * inv_cap>0: pause NEW buys whenever inventory value >= inv_cap * capital. Caps how big
        the bag can grow in a fall without ever flattening — a soft ceiling, no realized loss.
      * stop_loss>0: if the open bag's UNREALIZED loss exceeds stop_loss (e.g. 0.15), market-
        sell the whole bag (taker) and stand down for `cooldown` bars before rebuilding. A hard
        tail-cut that intervenes RARELY (only in a deep fall), so no churn — but it locks the
        loss and can sell the bottom. Cost-basis tracked to compute the unrealized loss."""
    n = len(c)
    equity = np.zeros(n)
    cash = capital
    inv = 0.0
    inv_cost = 0.0              # running acquisition cost (incl fees) of held inventory
    cells: List[dict] = []      # each: {'lo','hi','state'}
    center = 0.0
    size = 0.0
    fee_paid = 0.0
    rt = 0
    active = False
    stop_until = 0              # bar index before which we stay stood-down after a stop-loss
    cap_val = inv_cap * capital if inv_cap > 0 else float("inf")

    def build_grid(price: float, i: int):
        nonlocal cells, center, size, inv, inv_cost, cash, fee_paid
        # AUDIT FIX (2026-07-03): carry REAL equity through rebuilds. The old code did
        # `cash = capital - cost`, silently resetting the account to initial capital at
        # every rebuild — realized losses from stand-downs/stop-losses were erased,
        # flattering exactly the gated modes under test.
        equity_now = cash + inv * price
        band = max(range_mult * ap[i], 2 * step)         # half-band in fraction
        floor, ceil = price * (1 - band), price * (1 + band)
        levels = []
        p = price
        while p > floor:
            p /= (1 + step); levels.append(p)
        levels = levels[::-1] + [price]
        p = price
        while p < ceil:
            p *= (1 + step); levels.append(p)
        levels = sorted(set(levels))
        ncell = max(len(levels) - 1, 1)
        size = (equity_now * 0.95) / (price * ncell)      # 95% of CURRENT equity
        cells = []
        holding = 0
        for k in range(len(levels) - 1):
            lo, hi = levels[k], levels[k + 1]
            if lo >= price:                               # above center -> pre-hold to sell
                cells.append({"lo": lo, "hi": hi, "state": "holding"}); holding += 1
            else:
                cells.append({"lo": lo, "hi": hi, "state": "empty"})
        want = size * holding                             # target pre-hold inventory
        need = max(0.0, want - inv)                       # buy only the shortfall (an existing
        cost = need * price * (1 + taker)                 #  bag re-arms sell cells for free)
        if cost > cash:                                   # can't afford full pre-hold
            need = max(cash, 0.0) / (price * (1 + taker))
            cost = need * price * (1 + taker)
        cash -= cost
        inv += need
        inv_cost += cost                                  # cost basis of the pre-hold
        fee_paid += need * price * taker

    def liquidate(price: float):
        nonlocal cash, inv, inv_cost, fee_paid
        if inv > 0:
            cash += inv * price * (1 - taker); fee_paid += inv * price * taker
            inv = 0.0; inv_cost = 0.0

    hedged_qty = 0.0
    hedge_entry = 0.0

    for i in range(n):
        lo, hi, cl = c[i][3], c[i][2], c[i][4]
        in_sd = regime[i] in standdown
        if hedge and in_sd:                               # HEDGE mode: pause book, go neutral
            if active and hedged_qty == 0.0 and inv > 0:
                hedged_qty = inv; hedge_entry = cl        # open perp short = inventory
                hfee = hedged_qty * cl * taker
                cash -= hfee; fee_paid += hfee
            equity[i] = cash + inv * cl + hedged_qty * (hedge_entry - cl)
            continue
        if hedge and hedged_qty > 0.0:                    # range returned: unwind hedge
            pnl = hedged_qty * (hedge_entry - cl)
            hfee = hedged_qty * cl * taker
            cash += pnl - hfee; fee_paid += hfee
            hedged_qty = 0.0; hedge_entry = 0.0
            if cells and not (cells[0]["lo"] <= cl <= cells[-1]["hi"]):
                build_grid(cl, i)                         # price left the band: re-center (DGT),
        #                                                   existing bag re-arms the sell side
        if in_sd and not freeze and not hedge:            # EXIT mode: dump inventory, wait
            if active:
                liquidate(cl); active = False; cells = []
            equity[i] = cash + inv * cl
            continue
        if not active and i > 30 and not in_sd and i >= stop_until:   # (re)start when allowed
            build_grid(cl, i); active = True
        frozen = in_sd and freeze                          # freeze: keep sells, pause new buys —
        #                                                    static grid, NO chasing/re-center churn
        if active:
            for cell in cells:
                if (cell["state"] == "empty" and not frozen and lo <= cell["lo"]
                        and inv * cl < cap_val):           # inv_cap: soft ceiling on the bag
                    cost = cell["lo"] * size * (1 + maker)
                    if cash >= cost:
                        cash -= cost; inv += size; inv_cost += cost
                        fee_paid += cell["lo"] * size * maker; cell["state"] = "holding"
                elif cell["state"] == "holding" and hi >= cell["hi"] and inv >= size:
                    proceeds = cell["hi"] * size * (1 - maker)
                    cash += proceeds
                    inv_cost -= inv_cost * (size / inv) if inv > 0 else 0.0   # relieve basis
                    inv -= size; fee_paid += cell["hi"] * size * maker
                    cell["state"] = "empty"; rt += 1
            # stop-loss: rare hard tail-cut on a deep unrealized drawdown of the open bag
            if stop_loss > 0 and inv > 0 and inv_cost > 0:
                unreal = (inv * cl - inv_cost) / inv_cost
                if unreal < -stop_loss:
                    liquidate(cl); active = False; cells = []; stop_until = i + cooldown
        equity[i] = cash + inv * cl
    return equity, rt, fee_paid


def metrics(eq: np.ndarray, ts: List[int], regime: List[str], hours_year=8760.0) -> dict:
    ret = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
    if len(ret) == 0 or ret.std() == 0:
        return {"total": 0.0, "sharpe": 0.0, "maxdd": 0.0, "range_ret": 0.0, "trend_ret": 0.0, "worst_mo": 0.0}
    cum = np.cumprod(1 + ret)
    dd = float((np.maximum.accumulate(cum) - cum).max() / np.maximum.accumulate(cum).max())
    # regime split (aligned to ret[i] = return over bar i+1)
    rr = tr = 0.0
    bym: Dict[str, float] = defaultdict(float)
    for i in range(len(ret)):
        if regime[i + 1] in TREND_REGIMES:
            tr += ret[i]
        else:
            rr += ret[i]
        mo = datetime.fromtimestamp(ts[i + 1] / 1000, tz=timezone.utc).strftime("%Y-%m")
        bym[mo] += ret[i]
    return {"total": float(cum[-1] - 1), "sharpe": float(ret.mean() / ret.std() * math.sqrt(hours_year)),
            "maxdd": dd, "range_ret": rr, "trend_ret": tr,
            "worst_mo": min(bym.values()) if bym else 0.0}


def run_symbol(sym: str, days: int, tf: str, step: float, range_mult: float, maker: float,
               taker: float, capital: float, er_win: int, er_thr: float, reg_csv: Optional[str],
               persist: int, slippage: float):
    c = fetch_ohlcv(sym, tf, days)
    if len(c) < 300:
        print(f"{sym}: only {len(c)} bars — skip"); return
    ts = [int(x[0]) for x in c]
    ap = atr_pct(c)
    if reg_csv and os.path.exists(reg_csv):
        rows = load_regime_csv(reg_csv, sym)
        regime = regime_series(rows, ts) if rows else lag1(internal_regime(c, er_win, er_thr, ap))
        src = "analyst" if rows else "internal(fallback)"
    else:
        regime = lag1(internal_regime(c, er_win, er_thr, ap)); src = "internal ER"
    raw_trend = sum(1 for r in regime if r in TREND_REGIMES)
    regime = smooth_regime(regime, persist)               # hysteresis so the gate can't thrash
    flips = sum(1 for i in range(1, len(regime))
                if (regime[i] in TREND_REGIMES) != (regime[i - 1] in TREND_REGIMES))
    rng = sum(1 for r in regime if r not in TREND_REGIMES) / len(regime)
    print(f"\n=== {sym}  ({len(c)} {tf} bars, {ts_span(ts)})  regime src: {src} (persist {persist})  "
          f"[{rng*100:.0f}% range / {(1-rng)*100:.0f}% trend, {flips} regime flips] ===")
    bh = metrics(np.array([x[4] for x in c]) / c[0][4] * capital, ts, regime)
    print(f"  buy & hold:            total {bh['total']*100:>+7.1f}%  Sharpe {bh['sharpe']:>5.2f}  maxDD {bh['maxdd']*100:4.0f}%")
    down = {"trend_down", "flash", "flash_risk"}
    modes = [
        ("GRID no gate",       set(), False, False),
        ("gate down (exit)",   down,  False, False),  # liquidate + rebuild (fee-heavy)
        ("gate down (freeze)", down,  True,  False),  # pause buys, keep sells, no dump
        ("gate down (hedge)",  down,  False, True),   # pause book + perp-short the bag
    ]
    for tag, standdown, frz, hdg in modes:
        eq, rt, fee = simulate(c, regime, step=step, range_mult=range_mult, ap=ap,
                               maker=maker + slippage, taker=taker + slippage,
                               capital=capital, standdown=standdown, freeze=frz, hedge=hdg)
        m = metrics(eq, ts, regime)
        print(f"  {tag:<18} total {m['total']*100:>+7.1f}%  Sharpe {m['sharpe']:>5.2f}  "
              f"maxDD {m['maxdd']*100:4.0f}%  RTs {rt:>5}  fees {fee/capital*100:5.1f}%  "
              f"| range {m['range_ret']*100:>+6.1f}%  trend {m['trend_ret']*100:>+6.1f}%  worstMo {m['worst_mo']*100:+.1f}%")


def ts_span(ts):
    f = lambda t: datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return f"{f(ts[0])}->{f(ts[-1])}"


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


GATES = {                        # name -> (standdown set, freeze flag, hedge flag)
    "none":   (set(), False, False),
    "freeze": ({"trend_down", "flash", "flash_risk"}, True, False),
    "exit":   ({"trend_down", "flash", "flash_risk"}, False, False),
    "hedge":  ({"trend_down", "flash", "flash_risk"}, False, True),
}


def sweep(symbols: List[str], days: int, tf: str, maker: float, taker: float, slip: float,
          capital: float, er_win: int, er_thr: float, reg_csv: Optional[str], persist: int,
          steps: List[float], rms: List[float], sls: List[float], caps: List[float],
          gates: List[str], is_frac: float, top: int):
    """Test every {gate x step x range x stop-loss x inv-cap} config across the basket, with an
    honest anti-overfit protocol:

      * Each coin's history is split IN-SAMPLE (first is_frac) / OUT-OF-SAMPLE (rest).
      * Configs are RANKED BY IN-SAMPLE median Sharpe — i.e. the choice you'd actually make
        with only the past visible — and their OUT-OF-SAMPLE result is shown beside it.
      * BREADTH: how many coins the config keeps net-positive OUT-OF-SAMPLE. A real edge holds
        across the basket AND out of sample; a curve-fit tops the IS column then falls apart OOS.

    The honest read is the OOS + breadth columns of whatever tops the IS column — not the single
    best backtest anywhere in the table (that one is guaranteed to be luck)."""
    # fetch + regime ONCE per coin (only simulate() varies across configs)
    data = []
    for s in symbols:
        try:
            c = fetch_ohlcv(s, tf, days)
        except Exception as exc:  # noqa: BLE001
            print(f"  {s}: fetch failed ({str(exc)[:60]}) — skip"); continue
        if len(c) < 300:
            print(f"  {s}: only {len(c)} bars — skip"); continue
        ts = [int(x[0]) for x in c]
        ap = atr_pct(c)
        if reg_csv and os.path.exists(reg_csv):
            rows = load_regime_csv(reg_csv, s)
            regime = regime_series(rows, ts) if rows else lag1(internal_regime(c, er_win, er_thr, ap))
        else:
            regime = lag1(internal_regime(c, er_win, er_thr, ap))
        regime = smooth_regime(regime, persist)
        split = int(len(c) * is_frac)
        data.append((s, c, ts, ap, regime, split))
    if not data:
        print("no usable symbols"); return

    span = ts_span(data[0][2])
    print(f"\n=== SWEEP — {len(data)} coins · {span} · IS first {is_frac*100:.0f}% / OOS rest · "
          f"maker {maker*1e4:g}+slip {slip*1e4:g}bps ===")
    grid = [(g, st, rm, sl, cp) for g in gates for st in steps for rm in rms
            for sl in sls for cp in caps]
    print(f"testing {len(grid)} configs x {len(data)} coins = {len(grid)*len(data)} sims "
          f"(this takes a few minutes)...\n")

    results = []
    for gi, (g, st, rm, sl, cp) in enumerate(grid):
        standdown, frz, hdg = GATES[g]
        is_sh, is_tot, oos_sh, oos_tot, oos_dd, oos_pos = [], [], [], [], [], 0
        for (s, c, ts, ap, regime, split) in data:
            eq, rt, fee = simulate(c, regime, step=st, range_mult=rm, ap=ap,
                                   maker=maker + slip, taker=taker + slip, capital=capital,
                                   standdown=standdown, freeze=frz, hedge=hdg,
                                   stop_loss=sl, inv_cap=cp)
            mi = metrics(eq[:split], ts[:split], regime[:split])
            mo = metrics(eq[split:], ts[split:], regime[split:])
            is_sh.append(mi["sharpe"]); is_tot.append(mi["total"])
            oos_sh.append(mo["sharpe"]); oos_tot.append(mo["total"]); oos_dd.append(mo["maxdd"])
            if mo["total"] > 0:
                oos_pos += 1
        results.append({
            "cfg": (g, st, rm, sl, cp),
            "is_sh": _median(is_sh), "is_tot": _median(is_tot),
            "oos_sh": _median(oos_sh), "oos_tot": _median(oos_tot),
            "oos_dd": _median(oos_dd), "breadth": oos_pos, "ncoin": len(data),
        })
        print(f"\r  {gi+1}/{len(grid)} configs done", end="", flush=True)
    print("\n")

    # rank by the honest selection criterion: in-sample median Sharpe
    results.sort(key=lambda r: r["is_sh"], reverse=True)
    hdr = (f"  {'gate':<7} {'step':>5} {'band':>4} {'SL':>5} {'cap':>4} | "
           f"{'IS_Sh':>6} {'IS_tot':>7} | {'OOS_Sh':>6} {'OOS_tot':>7} {'OOS_DD':>6} {'breadth':>8}")
    print("=== leaderboard (ranked by IN-SAMPLE Sharpe — the pick you'd actually make) ===")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in results[:top]:
        g, st, rm, sl, cp = r["cfg"]
        sl_s = f"{sl*100:.0f}%" if sl > 0 else "—"
        cp_s = f"{cp*100:.0f}%" if cp > 0 else "—"
        print(f"  {g:<7} {st*100:>4.1f}% {rm:>4.0f} {sl_s:>5} {cp_s:>4} | "
              f"{r['is_sh']:>6.2f} {r['is_tot']*100:>+6.1f}% | "
              f"{r['oos_sh']:>6.2f} {r['oos_tot']*100:>+6.1f}% {r['oos_dd']*100:>5.0f}% "
              f"{r['breadth']:>4}/{r['ncoin']}")

    baseline = next((r for r in results if r["cfg"][0] == "none" and r["cfg"][3] == 0
                     and r["cfg"][4] == 0 and abs(r["cfg"][1] - 0.006) < 1e-9
                     and abs(r["cfg"][2] - 6) < 1e-9), None)
    if baseline:
        b = baseline
        print("\n  plain no-gate grid (step 0.6% band 6, no SL/cap) — the anchor to beat:")
        print(f"    IS Sh {b['is_sh']:.2f}  IS {b['is_tot']*100:+.1f}%  |  "
              f"OOS Sh {b['oos_sh']:.2f}  OOS {b['oos_tot']*100:+.1f}%  "
              f"OOS_DD {b['oos_dd']*100:.0f}%  breadth {b['breadth']}/{b['ncoin']}")

    best = results[0]
    print("\n=== verdict ===")
    print("PASS only if the IS-top config ALSO has OOS Sharpe clearly > 0, OOS_tot > 0, and breadth")
    print(f"> half the basket — AND beats the plain no-gate anchor OOS. The IS winner here is "
          f"{best['cfg'][0]} step {best['cfg'][1]*100:.1f}% band {best['cfg'][2]:.0f} "
          f"SL {best['cfg'][3] or '—'} cap {best['cfg'][4] or '—'}:")
    print(f"  OOS Sharpe {best['oos_sh']:.2f} · OOS total {best['oos_tot']*100:+.1f}% · "
          f"breadth {best['breadth']}/{best['ncoin']}.")
    print("If OOS collapses vs IS, the tail-caps didn't create an edge — they curve-fit the past.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-0 regime-gated grid backtest")
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    ap.add_argument("--days", type=int, default=1850, help="1850d reaches back to ~2021 to include the 2022 bear")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--step-pct", type=float, default=0.006, help="geometric rung spacing (0.006 = 0.6%)")
    ap.add_argument("--range-mult", type=float, default=6.0, help="half-band = range_mult * ATR%")
    ap.add_argument("--maker-bps", type=float, default=7.5, help="maker fee per fill (7.5 = 0.075%, BNB)")
    ap.add_argument("--taker-bps", type=float, default=7.5, help="taker fee for stop-outs/setup")
    ap.add_argument("--capital", type=float, default=5000.0)
    ap.add_argument("--er-win", type=int, default=24)
    ap.add_argument("--er-thr", type=float, default=0.35, help="efficiency ratio >= thr => trend")
    ap.add_argument("--regime-persist", type=int, default=24,
                    help="bars a new regime must persist before the gate acts (hysteresis, anti-thrash)")
    ap.add_argument("--slippage-bps", type=float, default=2.0,
                    help="haircut per fill on top of fees (queue/adverse-selection realism)")
    ap.add_argument("--regime-csv", default="", help="use analyst regime export instead of internal")
    # --- risk knobs (single-run) + sweep controls ---
    ap.add_argument("--stop-loss", type=float, default=0.0,
                    help="single-run: flatten the bag if unrealized loss exceeds this (0.15 = 15%)")
    ap.add_argument("--inv-cap", type=float, default=0.0,
                    help="single-run: pause buys when inventory value exceeds this fraction of capital")
    ap.add_argument("--sweep", action="store_true", help="run the full IS/OOS breadth sweep instead")
    ap.add_argument("--sweep-gates", default="none,freeze,exit,hedge")
    ap.add_argument("--sweep-steps", default="0.006,0.01", help="comma list of step-pcts to test")
    ap.add_argument("--sweep-rms", default="4,6,8", help="comma list of range-mults (band widths)")
    ap.add_argument("--sweep-sl", default="0,0.15,0.25", help="comma list of stop-loss levels (0 = off)")
    ap.add_argument("--sweep-cap", default="0,0.6", help="comma list of inventory caps (0 = off)")
    ap.add_argument("--is-frac", type=float, default=0.6, help="in-sample fraction (rest is OOS)")
    ap.add_argument("--top", type=int, default=20, help="rows of the leaderboard to print")
    args = ap.parse_args()
    maker, taker, slip = args.maker_bps / 1e4, args.taker_bps / 1e4, args.slippage_bps / 1e4
    if args.step_pct < 2 * (maker + slip):
        print(f"WARNING: step {args.step_pct*100:.2f}% < 2x (maker+slip) {2*(maker+slip)*100:.2f}% — can't clear costs.\n")
    syms = [x.strip() for x in args.symbols.split(",") if x.strip()]
    if args.sweep:
        f = lambda s: [float(x) for x in s.split(",") if x.strip() != ""]
        sweep(syms, args.days, args.timeframe, maker, taker, slip, args.capital,
              args.er_win, args.er_thr, args.regime_csv or None, args.regime_persist,
              steps=f(args.sweep_steps), rms=f(args.sweep_rms), sls=f(args.sweep_sl),
              caps=f(args.sweep_cap), gates=[g.strip() for g in args.sweep_gates.split(",") if g.strip()],
              is_frac=args.is_frac, top=args.top)
        return 0
    print(f"=== GRID BACKTEST — step {args.step_pct*100:.2f}% · band {args.range_mult}xATR · "
          f"maker {args.maker_bps:g}bps + slip {args.slippage_bps:g}bps · ${args.capital:g} ===")
    print("Bar to pass: 'gate down+flash' beats no-gate (higher Sharpe / lower DD) AND stays net-positive"
          " through the 2022 bear, after fees+slippage.")
    for s in syms:
        try:
            run_symbol(s, args.days, args.timeframe, args.step_pct, args.range_mult, maker, taker,
                       args.capital, args.er_win, args.er_thr, args.regime_csv or None, args.regime_persist, slip)
        except Exception as exc:  # noqa: BLE001
            print(f"{s}: failed ({str(exc)[:80]})")
    print("\n=== read ===")
    print("If 'trend' column is deeply negative with no gate but the gate cuts that loss AND total")
    print("improves, regime-gating is doing real work. If the gated grid is still net-negative after")
    print("fees, the grid doesn't clear frictions here — sweep --step-pct / --range-mult, but a static")
    print("grid that only wins in a cherry-picked range is exactly what the research says to distrust.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# audit-fix 2026-07-03: equity carry-through, causal regime lag, hedge gate mode

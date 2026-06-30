"""daily_signal_lab.py — systematic hunt for a surviving daily edge, off cached data.

Reads data_cache/*.csv (run fetch_data.py first) and tests a LIBRARY of signals across
all coins with one honest scorecard each: net of fees, in-sample AND out-of-sample,
breadth (how many coins it works on), by-month. Then ranks what survives so we can mix
the winners. No live fetch — the agent can run this itself.

Honesty rules baked in:
  * positions are causal (use returns up to day t to hold t -> t+1),
  * fee charged on turnover (|position change|) every day,
  * IS/OOS = first vs second half of the calendar,
  * a signal only "passes" if it's OOS-positive AND broad (>=60% of coins).
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

CACHE = "data_cache"
FEE = 0.0010   # 10 bps per unit turnover


def load():
    rows = list(csv.DictReader(open(os.path.join(CACHE, "daily_close.csv"))))
    coins = [c for c in rows[0].keys() if c != "date"]
    dates = [r["date"] for r in rows]
    close = {c: np.array([float(r[c]) if r[c] else np.nan for r in rows]) for c in coins}
    # returns aligned so ret[c][i] = close[i]/close[i-1]-1 (ret[0]=0)
    ret = {}
    for c in coins:
        p = close[c]; r = np.zeros(len(p))
        r[1:] = p[1:] / p[:-1] - 1.0
        ret[c] = r
    dvol = {}
    if os.path.exists(os.path.join(CACHE, "dvol.csv")):
        dv = {r["date"]: r for r in csv.DictReader(open(os.path.join(CACHE, "dvol.csv")))}
        dvol = {"BTC": np.array([float(dv[d]["BTC"]) if d in dv and dv[d]["BTC"] else np.nan for d in dates]),
                "ETH": np.array([float(dv[d]["ETH"]) if d in dv and dv[d]["ETH"] else np.nan for d in dates])}
    supply = None
    if os.path.exists(os.path.join(CACHE, "stable_supply.csv")):
        sp = {r["date"]: float(r["stable_supply_usd"]) for r in csv.DictReader(
            open(os.path.join(CACHE, "stable_supply.csv"))) if r["stable_supply_usd"]}
        supply = np.array([sp.get(d, np.nan) for d in dates])
    return dates, coins, close, ret, dvol, supply


def backtest(positions: Dict[str, np.ndarray], ret: Dict[str, np.ndarray], dates: List[str]):
    """positions[c][t] = desired position for holding t->t+1 (causal). Returns metrics."""
    coins = list(positions.keys())
    per_coin_daily: Dict[str, np.ndarray] = {}
    pooled = np.zeros(len(dates))
    active = np.zeros(len(dates))
    for c in coins:
        pos = positions[c]
        pnl = np.zeros(len(dates))
        for t in range(len(dates) - 1):
            turn = abs(pos[t] - (pos[t - 1] if t > 0 else 0.0))
            pnl[t] = pos[t] * ret[c][t + 1] - FEE * turn
        per_coin_daily[c] = pnl
        pooled += pnl
        active += (np.abs(pos) > 0).astype(float)
    n_active = np.where(active > 0)[0]
    # average across coins that are active (equal-weight book)
    book = np.zeros(len(dates))
    for t in range(len(dates)):
        a = active[t]
        book[t] = pooled[t] / a if a > 0 else 0.0
    return per_coin_daily, book


def metrics(daily: np.ndarray, dates: List[str]):
    d = daily[:-1]  # last day has no forward return
    if d.std() == 0 or len(d) == 0:
        return dict(sharpe=0, total=0, maxdd=0, n=0)
    cum = np.cumsum(d)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    sharpe = float(d.mean() / d.std() * math.sqrt(365)) if d.std() > 0 else 0.0
    return dict(sharpe=sharpe, total=float(d.sum()), maxdd=dd, n=int((np.abs(d) > 0).sum()))


def split_metrics(daily: np.ndarray, dates: List[str]):
    cut = len(dates) // 2
    return metrics(daily[:cut], dates[:cut]), metrics(daily[cut:], dates[cut:])


# ---------- signal library (each returns positions dict, causal) ----------
def sig_tsm(ret, coins, dates, k):
    pos = {}
    for c in coins:
        r = ret[c]; p = np.zeros(len(dates))
        for t in range(k, len(dates)):
            p[t] = np.sign(r[t - k + 1:t + 1].sum())
        pos[c] = p
    return pos


def sig_tsrev(ret, coins, dates, k=1):
    pos = {}
    for c in coins:
        r = ret[c]; p = np.zeros(len(dates))
        for t in range(k, len(dates)):
            p[t] = -np.sign(r[t - k + 1:t + 1].sum())
        pos[c] = p
    return pos


def _xs_rank(ret, coins, dates, lookback, reverse):
    # cross-sectional: rank coins each day by trailing return; long top third / short bottom
    pos = {c: np.zeros(len(dates)) for c in coins}
    for t in range(lookback, len(dates)):
        scores = {c: ret[c][t - lookback + 1:t + 1].sum() for c in coins}
        order = sorted(coins, key=lambda c: scores[c])
        n = len(coins); third = max(1, n // 3)
        losers, winners = order[:third], order[-third:]
        for c in winners:
            pos[c][t] = -1.0 if reverse else 1.0
        for c in losers:
            pos[c][t] = 1.0 if reverse else -1.0
    return pos


def sig_xsmom(ret, coins, dates, lookback=5):
    return _xs_rank(ret, coins, dates, lookback, reverse=False)


def sig_xsrev(ret, coins, dates, lookback=1):
    return _xs_rank(ret, coins, dates, lookback, reverse=True)


def sig_stable(ret, coins, dates, supply, window=7, long_only=False):
    pos = {c: np.zeros(len(dates)) for c in coins}
    if supply is None:
        return pos
    for t in range(window, len(dates)):
        if np.isnan(supply[t]) or np.isnan(supply[t - window]) or supply[t - window] <= 0:
            continue
        g = supply[t] / supply[t - window] - 1.0
        val = 1.0 if g > 0 else (0.0 if long_only else -1.0)
        for c in coins:
            pos[c][t] = val
    return pos


def sig_dvol_fear(ret, coins, dates, dvol, lb=30, z=1.0, hold=3):
    # when BTC DVOL spikes above trailing mean+z*std, go long all coins for `hold` days (buy fear)
    pos = {c: np.zeros(len(dates)) for c in coins}
    if not dvol or "BTC" not in dvol:
        return pos
    v = dvol["BTC"]
    for t in range(lb, len(dates)):
        w = v[t - lb:t]
        w = w[~np.isnan(w)]
        if len(w) < lb // 2 or np.isnan(v[t]):
            continue
        if v[t] > w.mean() + z * w.std():
            for k in range(t, min(t + hold, len(dates))):
                for c in coins:
                    pos[c][k] = 1.0
    return pos


def sig_btc_lead(ret, coins, dates, k=2):
    pos = {c: np.zeros(len(dates)) for c in coins}
    b = ret["BTC"]
    for c in coins:
        if c == "BTC":
            continue
        p = np.zeros(len(dates))
        for t in range(k, len(dates)):
            p[t] = np.sign(b[t - k + 1:t + 1].sum())
        pos[c] = p
    return pos


def gate(base: Dict[str, np.ndarray], allow: Dict[str, np.ndarray]):
    """Combine: keep base position only where allow != 0 and agrees in sign-bias (long gate)."""
    out = {}
    for c in base:
        out[c] = np.where((allow[c] >= 0), base[c], np.where(base[c] > 0, 0.0, base[c]))
    return out


def evaluate(name, pos, ret, dates, results):
    per_coin, book = backtest(pos, ret, dates)
    m = metrics(book, dates)
    is_m, oos_m = split_metrics(book, dates)
    pos_coins = sum(1 for c in per_coin if metrics(per_coin[c], dates)["total"] > 0)
    n_coins = sum(1 for c in per_coin if (np.abs(pos[c]) > 0).any())
    breadth = pos_coins / n_coins if n_coins else 0
    results.append((name, m["sharpe"], m["total"], m["maxdd"], is_m["sharpe"],
                    oos_m["sharpe"], oos_m["total"], breadth))
    return book


def main():
    dates, coins, close, ret, dvol, supply = load()
    print(f"loaded {len(dates)} days, {len(coins)} coins: {coins}")
    print(f"DVOL: {list(dvol)} · supply: {'yes' if supply is not None else 'no'}\n")
    results = []
    books = {}

    for k in (2, 3, 5, 10, 20):
        books[f"TSM-{k}"] = evaluate(f"TSM-{k}d", sig_tsm(ret, coins, dates, k), ret, dates, results)
    books["TSREV-1"] = evaluate("TSREV-1d", sig_tsrev(ret, coins, dates, 1), ret, dates, results)
    books["TSREV-2"] = evaluate("TSREV-2d", sig_tsrev(ret, coins, dates, 2), ret, dates, results)
    for lb in (3, 5, 10):
        evaluate(f"XSMOM-{lb}d", sig_xsmom(ret, coins, dates, lb), ret, dates, results)
    evaluate("XSREV-1d", sig_xsrev(ret, coins, dates, 1), ret, dates, results)
    evaluate("XSREV-2d", sig_xsrev(ret, coins, dates, 2), ret, dates, results)
    if supply is not None:
        evaluate("STABLE-dir", sig_stable(ret, coins, dates, supply, 7, False), ret, dates, results)
        evaluate("STABLE-long", sig_stable(ret, coins, dates, supply, 7, True), ret, dates, results)
    if dvol:
        for z in (0.5, 1.0, 1.5):
            evaluate(f"DVOLFEAR-z{z}", sig_dvol_fear(ret, coins, dates, dvol, 30, z, 3), ret, dates, results)
    evaluate("BTCLEAD-2d", sig_btc_lead(ret, coins, dates, 2), ret, dates, results)

    results.sort(key=lambda r: -r[5])  # by OOS sharpe
    print(f"{'signal':<14}{'Sharpe':>8}{'total':>9}{'maxDD':>8}{'IS Sh':>7}{'OOS Sh':>8}{'OOS tot':>9}{'breadth':>8}")
    for name, sh, tot, dd, iss, oss, oost, br in results:
        flag = "  <==" if (oss > 0.3 and br >= 0.6 and iss > 0) else ""
        print(f"{name:<14}{sh:>8.2f}{tot*100:>+8.1f}%{dd*100:>7.1f}%{iss:>7.2f}{oss:>8.2f}"
              f"{oost*100:>+8.1f}%{br*100:>7.0f}%{flag}")

    survivors = [r[0] for r in results if r[5] > 0.3 and r[7] >= 0.6 and r[4] > 0]
    print(f"\nsurvivors (OOS Sharpe>0.3, IS>0, breadth>=60%): {survivors or 'none yet'}")
    return dates, coins, ret, dvol, supply, books, results


if __name__ == "__main__":
    main()

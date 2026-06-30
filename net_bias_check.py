"""net_bias_check.py — does the TA net bias (consensus) actually pay?

A focused, single-feature calibration for the eyeball impression that "when the
whole TA board agrees, the coin goes that way." It is reactive/momentum-friendly
by design: it does NOT ask whether the board predicts the future, it asks the
scalper's question — *when we reacted to a strongly-aligned board, did the move
continue enough to win the bracket?*

For every decided observation it computes the board's agreement WITH THE TRADE
TAKEN (direction-aware consensus: + = board agreed with the side we took),
buckets trades by that strength, and reports win rate + expectancy per bucket —
on a time hold-out, so a rising staircase has to survive out-of-sample and isn't
just the indicator shadowing price in-sample.

Read-only on observations.db; never trades.

    python net_bias_check.py --db observations.db
    python net_bias_check.py --db observations.db --buckets 5
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

HOLDOUT_FRAC = 0.30
MIN_PER_BUCKET = 15  # below this a bucket's win-rate is too noisy to read


def _f(v) -> Optional[float]:
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _ts_ms(ts: Optional[str]) -> int:
    if not ts:
        return 0
    s = ts.replace(" ", "T")
    tail = s[10:]
    if not (s.endswith("Z") or "+" in tail or "-" in tail):
        s += "+00:00"
    s = s.replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(s).timestamp() * 1000)
    except ValueError:
        return 0


def load(db_path: str) -> List[Tuple[int, float, float]]:
    """Return (ts_ms, aligned_consensus, ret) for every decided observation that
    carries a consensus value. aligned_consensus > 0 means the TA board agreed
    with the direction the trade actually took."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ts_open, direction, entry_price, pnl, ta_consensus "
        "FROM trades WHERE status IN ('WIN','LOSS') ORDER BY ts_open"
    ).fetchall()
    conn.close()
    out: List[Tuple[int, float, float]] = []
    for r in rows:
        entry, pnl, cons = _f(r["entry_price"]), _f(r["pnl"]), _f(r["ta_consensus"])
        if not entry or pnl is None or cons is None:
            continue
        is_long = str(r["direction"]).endswith("LONG")
        aligned = cons if is_long else -cons
        out.append((_ts_ms(r["ts_open"]), aligned, pnl / entry))
    return out


def _quantile_edges(values: Sequence[float], k: int) -> List[float]:
    """Inner cut points splitting `values` into k roughly-equal groups."""
    s = sorted(values)
    edges = []
    for i in range(1, k):
        idx = min(len(s) - 1, int(i / k * len(s)))
        edges.append(s[idx])
    # de-duplicate while preserving order (ties collapse buckets, which is fine)
    seen, uniq = set(), []
    for e in edges:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


def _bucket(aligned: float, edges: Sequence[float]) -> int:
    i = 0
    for e in edges:
        if aligned >= e:
            i += 1
        else:
            break
    return i


def _stats(rets: Sequence[float]) -> Tuple[int, float, float]:
    n = len(rets)
    if n == 0:
        return 0, 0.0, 0.0
    win = sum(1 for r in rets if r > 0) / n
    exp = sum(rets) / n
    return n, win, exp


def main() -> int:
    p = argparse.ArgumentParser(description="Net-bias (TA consensus) calibration check")
    p.add_argument("--db", default="observations.db")
    p.add_argument("--buckets", type=int, default=5, help="number of consensus strength buckets")
    p.add_argument("--fee-bps", type=float, default=0.0,
                   help="round-trip cost in basis points to subtract from expectancy "
                        "(e.g. 15 = 0.15%%); win%% stays gross")
    args = p.parse_args()
    fee = args.fee_bps / 10000.0  # bps -> fraction of price

    data = load(args.db)
    if len(data) < MIN_PER_BUCKET * args.buckets:
        print(f"only {len(data)} decided trades carry consensus — too few; keep harvesting")
        return 1

    data.sort(key=lambda t: t[0])
    split = int(len(data) * (1 - HOLDOUT_FRAC))
    train, hold = data[:split], data[split:]
    edges = _quantile_edges([a for _t, a, _r in train], args.buckets)
    nb = len(edges) + 1

    base_n, base_win, base_exp = _stats([r for _t, _a, r in data])
    print(f"\n=== NET-BIAS CHECK — {len(data)} decided trades "
          f"(take-all: {base_win*100:.0f}% win, net exp {(base_exp - fee)*100:+.3f}%) ===")
    print(f"bucket = TA board agreement WITH the trade taken (low → split, high → strongly aligned)")
    print(f"expectancy is NET of {args.fee_bps:.0f} bps round-trip fee" if fee
          else "expectancy is GROSS (pass --fee-bps to subtract costs)")
    print(f"\n{'bucket':<8}{'range':<18}{'n':>5}{'win%':>7}{'net exp%':>10}  |  "
          f"{'OOS n':>5}{'win%':>7}{'net exp%':>10}")

    def rng(b: int) -> str:
        lo = "-inf" if b == 0 else f"{edges[b-1]:.2f}"
        hi = "+inf" if b == nb - 1 else f"{edges[b]:.2f}"
        return f"[{lo}, {hi})"

    tr_win, ho_win, ho_net = [], [], []
    for b in range(nb):
        tr = [r for _t, a, r in train if _bucket(a, edges) == b]
        ho = [r for _t, a, r in hold if _bucket(a, edges) == b]
        tn, tw, te = _stats(tr)
        hn, hw, he = _stats(ho)
        tr_win.append(tw if tn >= MIN_PER_BUCKET else None)
        ho_win.append(hw if hn >= MIN_PER_BUCKET else None)
        ho_net.append((he - fee) if hn >= MIN_PER_BUCKET else None)
        flag = "" if hn >= MIN_PER_BUCKET else "  (OOS thin)"
        print(f"B{b:<7}{rng(b):<18}{tn:>5}{tw*100:>6.0f}%{(te-fee)*100:>+9.3f}%  |  "
              f"{hn:>5}{hw*100:>6.0f}%{(he-fee)*100:>+9.3f}%{flag}")

    # Verdict: does the strongest-aligned bucket clearly beat the most-split one,
    # in-sample AND out-of-sample? A reactive edge should rise with agreement.
    print("\n=== read ===")
    lo_in, hi_in = tr_win[0], tr_win[-1]
    lo_oo, ho_oo = ho_win[0], ho_win[-1]
    top_net = ho_net[-1]
    if None in (lo_in, hi_in, lo_oo, ho_oo):
        print("not enough trades in the end buckets to judge — keep harvesting.")
    elif hi_in > lo_in + 0.05 and ho_oo > lo_oo + 0.05:
        print(f"staircase HOLDS: strongly-aligned trades win more than split ones "
              f"both in-sample ({hi_in*100:.0f}% vs {lo_in*100:.0f}%) and out-of-sample "
              f"({ho_oo*100:.0f}% vs {lo_oo*100:.0f}%) — the net bias looks reactive-real.")
        if fee and top_net is not None:
            if top_net > 0:
                print(f"  AND the top bucket clears fees: net exp {top_net*100:+.3f}%/trade "
                      f"after {args.fee_bps:.0f} bps — worth a frozen forward test.")
            else:
                print(f"  BUT the top bucket does NOT clear fees: net exp {top_net*100:+.3f}%/trade "
                      f"after {args.fee_bps:.0f} bps — the edge is real but too small to trade as-is; "
                      f"needs tighter selectivity or cheaper fills.")
        else:
            print("  (pass --fee-bps to see whether it survives trading costs.)")
    elif hi_in > lo_in + 0.05:
        print(f"in-sample the staircase is there ({hi_in*100:.0f}% vs {lo_in*100:.0f}%) but it "
              f"does NOT hold out-of-sample ({ho_oo*100:.0f}% vs {lo_oo*100:.0f}%) — most likely "
              f"the indicator shadowing price, not a tradeable edge.")
    else:
        print("no staircase: strongly-aligned trades do NOT win more than split ones. "
              "The net bias looks like it's tracking price, not edging it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""feature_check.py — does ANY single recorded feature predict trade outcomes?

A generalisation of net_bias_check: pick any column the observation journal
records (order-book imbalance, OFI, microprice gap, depth, consensus, …),
orient it so positive = supports the trade direction, bucket trades by its
strength, and report win rate + expectancy net of fees, on a time hold-out.

This is how we test the **order-book / LOB angle** on the data we actually have:
the snapshot microstructure features (book_imbalance, ofi_rel, microprice_gap_bps,
depth_imbalance) the journal logs on every setup. It is NOT deep DeepLOB sequence
modelling (that needs high-frequency full-ladder history we don't have) — it's
the shallow, snapshot shadow of it, on ~the same short window as everything else.

    python feature_check.py --db observations.db --feature book_imbalance --mode centered --fee-bps 10
    python feature_check.py --db observations.db --feature ofi_rel --mode signed --fee-bps 10
    python feature_check.py --db observations.db --feature microprice_gap_bps --mode signed
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

HOLDOUT_FRAC = 0.30
MIN_PER_BUCKET = 15


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


def load(db_path: str, feature: str, mode: str, center: float) -> List[Tuple[int, float, float]]:
    """(ts_ms, aligned_feature, ret). aligned > 0 means the feature supported the
    direction the trade actually took."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    if feature not in cols:
        conn.close()
        raise SystemExit(f"column {feature!r} not in trades table. Available: {sorted(cols)}")
    rows = conn.execute(
        f"SELECT ts_open, direction, entry_price, pnl, {feature} AS feat "
        "FROM trades WHERE status IN ('WIN','LOSS') ORDER BY ts_open"
    ).fetchall()
    conn.close()
    out: List[Tuple[int, float, float]] = []
    for r in rows:
        entry, pnl, feat = _f(r["entry_price"]), _f(r["pnl"]), _f(r["feat"])
        if not entry or pnl is None or feat is None:
            continue
        is_long = str(r["direction"]).endswith("LONG")
        base = (feat - center) if mode == "centered" else feat
        aligned = base if is_long else -base
        out.append((_ts_ms(r["ts_open"]), aligned, pnl / entry))
    return out


def _edges(vals: Sequence[float], k: int) -> List[float]:
    s = sorted(vals)
    edges, seen = [], set()
    for i in range(1, k):
        e = s[min(len(s) - 1, int(i / k * len(s)))]
        if e not in seen:
            seen.add(e)
            edges.append(e)
    return edges


def _bucket(a: float, edges: Sequence[float]) -> int:
    i = 0
    for e in edges:
        if a >= e:
            i += 1
        else:
            break
    return i


def _stats(rets: Sequence[float]) -> Tuple[int, float, float]:
    n = len(rets)
    if n == 0:
        return 0, 0.0, 0.0
    return n, sum(1 for r in rets if r > 0) / n, sum(rets) / n


def main() -> int:
    p = argparse.ArgumentParser(description="Does a single recorded feature predict outcomes?")
    p.add_argument("--db", default="observations.db")
    p.add_argument("--feature", required=True, help="column name, e.g. book_imbalance / ofi_rel")
    p.add_argument("--mode", choices=("signed", "centered"), default="signed",
                   help="'centered' for 0..1 features like book_imbalance; 'signed' for OFI/gaps")
    p.add_argument("--center", type=float, default=0.5, help="neutral value for --mode centered")
    p.add_argument("--buckets", type=int, default=5)
    p.add_argument("--fee-bps", type=float, default=10.0)
    args = p.parse_args()
    fee = args.fee_bps / 10000.0

    data = load(args.db, args.feature, args.mode, args.center)
    if len(data) < MIN_PER_BUCKET * args.buckets:
        print(f"only {len(data)} decided trades carry {args.feature} — too few; keep harvesting")
        return 1
    data.sort(key=lambda t: t[0])
    split = int(len(data) * (1 - HOLDOUT_FRAC))
    train, hold = data[:split], data[split:]
    edges = _edges([a for _t, a, _r in train], args.buckets)
    nb = len(edges) + 1

    bn, bw, be = _stats([r for _t, _a, r in data])
    print(f"\n=== FEATURE CHECK — {args.feature} ({args.mode}), {len(data)} trades, "
          f"{args.fee_bps:.0f} bps ===")
    print(f"take-all: {bw*100:.0f}% win, net exp {(be-fee)*100:+.3f}%")
    print("bucket = feature support FOR the trade taken (low = against, high = strongly for)")
    print(f"\n{'bucket':<8}{'n':>6}{'win%':>7}{'net exp%':>10}  |  {'OOS n':>6}{'win%':>7}{'net exp%':>10}")

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
        print(f"B{b:<7}{tn:>6}{tw*100:>6.0f}%{(te-fee)*100:>+9.3f}%  |  "
              f"{hn:>6}{hw*100:>6.0f}%{(he-fee)*100:>+9.3f}%{flag}")

    print("\n=== read ===")
    lo_in, hi_in = tr_win[0], tr_win[-1]
    lo_oo, ho_oo, top_net = ho_win[0], ho_win[-1], ho_net[-1]
    if None in (lo_in, hi_in, lo_oo, ho_oo):
        print("end buckets too thin to judge — keep harvesting.")
    elif hi_in > lo_in + 0.05 and ho_oo > lo_oo + 0.05:
        msg = (f"staircase HOLDS in+out of sample ({hi_in*100:.0f}% vs {lo_in*100:.0f}% / "
               f"{ho_oo*100:.0f}% vs {lo_oo*100:.0f}%)")
        if top_net is not None and top_net > 0:
            print(msg + f", and the strong bucket clears fees (net {top_net*100:+.3f}%/trade) — "
                  f"{args.feature} carries a real microstructure signal; worth more data + a forward test.")
        else:
            print(msg + f", but the strong bucket does NOT clear fees (net {top_net*100:+.3f}%) — "
                  f"signal present but too small to trade as-is.")
    elif hi_in > lo_in + 0.05:
        print(f"in-sample staircase ({hi_in*100:.0f}% vs {lo_in*100:.0f}%) but it does NOT hold "
              f"out-of-sample — likely the feature shadowing price, not edge.")
    else:
        print(f"no staircase — {args.feature} does not separate winners from losers. No microstructure edge here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

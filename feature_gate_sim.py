"""feature_gate_sim.py — breadth + persistence test for a gated feature edge.

feature_check showed OFI's win-rate staircase holds out-of-sample and the top
deciles clear fees at 5 bps. But that was POOLED across all coins and all time —
exactly how the consensus 'edge' looked great until the per-symbol breakdown
revealed one coin carried it. This gates trades to the strongest slice of a
feature (e.g. top 30% of aligned OFI) and reports the result by symbol, by month,
and by side, net of fees — so concentration and regime-dependence can't hide.

    python feature_gate_sim.py --db observations.db --feature ofi_rel --mode signed --top-pct 30 --fee-bps 5
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple


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


# row = (ts, symbol, side, aligned, ret)
Row = Tuple[int, str, str, float, float]


def load(db_path: str, feature: str, mode: str, center: float) -> List[Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    if feature not in cols:
        conn.close()
        raise SystemExit(f"column {feature!r} not found. Available: {sorted(cols)}")
    rows = conn.execute(
        f"SELECT ts_open, symbol, direction, entry_price, pnl, {feature} AS feat "
        "FROM trades WHERE status IN ('WIN','LOSS') ORDER BY ts_open"
    ).fetchall()
    conn.close()
    out: List[Row] = []
    for r in rows:
        entry, pnl, feat = _f(r["entry_price"]), _f(r["pnl"]), _f(r["feat"])
        if not entry or pnl is None or feat is None:
            continue
        is_long = str(r["direction"]).endswith("LONG")
        base = (feat - center) if mode == "centered" else feat
        aligned = base if is_long else -base
        side = "LONG" if is_long else "SHORT"
        out.append((_ts_ms(r["ts_open"]), str(r["symbol"]), side, aligned, pnl / entry))
    return out


def _perf(nets: Sequence[float]) -> Dict[str, float]:
    n = len(nets)
    if n == 0:
        return {"n": 0, "win": 0.0, "exp": 0.0, "total": 0.0}
    return {"n": n, "win": sum(1 for x in nets if x > 0) / n,
            "exp": sum(nets) / n, "total": sum(nets)}


def _line(label: str, p: Dict[str, float]) -> str:
    if p["n"] == 0:
        return f"  {label:<14} (no trades)"
    return (f"  {label:<14} n={int(p['n']):<5} win {p['win']*100:>4.0f}%  "
            f"net/trade {p['exp']*100:>+7.3f}%  total {p['total']*100:>+7.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Breadth+persistence test for a gated feature")
    ap.add_argument("--db", default="observations.db")
    ap.add_argument("--feature", required=True)
    ap.add_argument("--mode", choices=("signed", "centered"), default="signed")
    ap.add_argument("--center", type=float, default=0.5)
    ap.add_argument("--top-pct", type=float, default=30.0,
                    help="gate to the strongest X%% of aligned feature (default 30)")
    ap.add_argument("--fee-bps", type=float, default=5.0)
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0

    data = load(args.db, args.feature, args.mode, args.center)
    if not data:
        print("no decided trades carry that feature — keep harvesting")
        return 1
    data.sort(key=lambda r: r[0])

    aligned_sorted = sorted(a for _t, _s, _sd, a, _r in data)
    idx = min(len(aligned_sorted) - 1, int((1 - args.top_pct / 100.0) * len(aligned_sorted)))
    thr = aligned_sorted[idx]
    gated = [r for r in data if r[3] >= thr]

    all_perf = _perf([r[4] - fee for r in data])
    g_perf = _perf([r[4] - fee for r in gated])
    print(f"\n=== FEATURE GATE — {args.feature} ({args.mode}), top {args.top_pct:g}% "
          f"(aligned >= {thr:.4f}), {args.fee_bps:.0f} bps ===")
    print(f"gate passes {g_perf['n']} of {all_perf['n']} trades\n")
    print("OVERALL (net of fees):")
    print(_line("take-all", all_perf))
    print(_line("gated", g_perf))

    print("\nBY SIDE:")
    for side in ("LONG", "SHORT"):
        print(_line(side, _perf([r[4] - fee for r in gated if r[2] == side])))

    print("\nBY SYMBOL:")
    by_sym: Dict[str, List[float]] = defaultdict(list)
    for r in gated:
        by_sym[r[1]].append(r[4] - fee)
    for s in sorted(by_sym, key=lambda k: -_perf(by_sym[k])["total"]):
        print(_line(s, _perf(by_sym[s])))

    print("\nBY MONTH:")
    by_month: Dict[str, List[float]] = defaultdict(list)
    for r in gated:
        by_month[datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc).strftime("%Y-%m")].append(r[4] - fee)
    pos_months = 0
    for ym in sorted(by_month):
        p = _perf(by_month[ym])
        pos_months += 1 if p["exp"] > 0 else 0
        print(_line(ym, p))

    n_months = len(by_month)
    pos_syms = sum(1 for s in by_sym if _perf(by_sym[s])["exp"] > 0)
    print("\n=== read ===")
    if g_perf["exp"] > 0 and pos_months >= max(2, n_months * 0.6) and pos_syms >= len(by_sym) * 0.6:
        print(f"gated edge is net-positive (+{g_perf['exp']*100:.3f}%/trade), in {pos_months}/{n_months} "
              f"months and {pos_syms}/{len(by_sym)} symbols — BROAD and PERSISTENT. This is the real "
              f"thing: worth capturing live order-book data and a forward test.")
    elif g_perf["exp"] > 0:
        print(f"gated edge is net-positive (+{g_perf['exp']*100:.3f}%/trade) but NOT broad/persistent "
              f"({pos_months}/{n_months} months, {pos_syms}/{len(by_sym)} symbols) — concentrated; "
              f"treat with the same suspicion as the consensus/XRP mirage.")
    else:
        print(f"gated edge is NOT net-positive ({g_perf['exp']*100:+.3f}%/trade) once gated — the pooled "
              f"staircase didn't hold up selective + net of fees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

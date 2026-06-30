"""conviction_gate_check.py — breadth+persistence+cost test for Kronos's
highest-conviction calls.

The search keeps surfacing `conviction >= 0.9` (Kronos's maximally confident
calls) as its best, OOS-positive theme — and the report card agrees the top
conviction bucket is the strongest. This puts that ONE signal through the same
make-or-break we used for OFI: gate to conviction >= threshold, then break the
result down by symbol, by month, and by side, net of fees — so concentration or
regime-dependence can't hide behind a pooled number.

conviction = the chosen side's probability (p_up for longs, p_down for shorts).
Outcomes are labelled at the offline labeler's 2.5/2.5 ATR bracket (R:R 1.0, so
50% win = breakeven before fees).

    python conviction_gate_check.py --db observations.db --min-conviction 0.9 --fee-bps 10
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


# row = (ts, symbol, side, conviction, ret)
Row = Tuple[int, str, str, float, float]


def load(db_path: str) -> List[Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ts_open, symbol, direction, entry_price, pnl, p_up, p_down "
        "FROM trades WHERE status IN ('WIN','LOSS') ORDER BY ts_open"
    ).fetchall()
    conn.close()
    out: List[Row] = []
    for r in rows:
        entry, pnl = _f(r["entry_price"]), _f(r["pnl"])
        pu, pd = _f(r["p_up"]), _f(r["p_down"])
        if not entry or pnl is None or pu is None or pd is None:
            continue
        is_long = str(r["direction"]).endswith("LONG")
        conviction = pu if is_long else pd
        out.append((_ts_ms(r["ts_open"]), str(r["symbol"]),
                    "LONG" if is_long else "SHORT", conviction, pnl / entry))
    return out


def _perf(nets: Sequence[float]) -> Dict[str, float]:
    n = len(nets)
    if n == 0:
        return {"n": 0, "win": 0.0, "exp": 0.0, "total": 0.0}
    return {"n": n, "win": sum(1 for x in nets if x > 0) / n,
            "exp": sum(nets) / n, "total": sum(nets)}


def _line(label: str, p: Dict[str, float]) -> str:
    if p["n"] == 0:
        return f"  {label:<14} (none)"
    return (f"  {label:<14} n={int(p['n']):<5} win {p['win']*100:>4.0f}%  "
            f"net/trade {p['exp']*100:>+7.3f}%  total {p['total']*100:>+7.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Breadth/persistence test for high-conviction Kronos calls")
    ap.add_argument("--db", default="observations.db")
    ap.add_argument("--min-conviction", type=float, default=0.9)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0

    data = load(args.db)
    if not data:
        print("no decided trades with p_up/p_down — keep harvesting")
        return 1
    data.sort(key=lambda r: r[0])
    gated = [r for r in data if r[3] >= args.min_conviction]

    allp = _perf([r[4] - fee for r in data])
    gp = _perf([r[4] - fee for r in gated])
    print(f"\n=== CONVICTION GATE — conviction >= {args.min_conviction:g}, {args.fee_bps:g}bps "
          f"(2.5/2.5 geometry, 50% = breakeven) ===")
    print(f"gate passes {gp['n']} of {allp['n']} decided trades "
          f"({100*gp['n']/allp['n']:.0f}%)\n")
    print("OVERALL (net of fees):")
    print(_line("take-all", allp))
    print(_line(f">= {args.min_conviction:g}", gp))

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
    if gp["exp"] > 0 and pos_months >= max(2, n_months * 0.6) and pos_syms >= len(by_sym) * 0.6:
        print(f"high-conviction calls are net-positive (+{gp['exp']*100:.3f}%/trade), in {pos_months}/{n_months} "
              f"months and {pos_syms}/{len(by_sym)} symbols — BROAD and PERSISTENT. The strongest lead yet; "
              f"worth a live forward test (and check it survives the live 1.5/2.5 geometry too).")
    elif gp["exp"] > 0:
        print(f"net-positive (+{gp['exp']*100:.3f}%/trade) but not broad/persistent "
              f"({pos_months}/{n_months} months, {pos_syms}/{len(by_sym)} symbols) — concentrated; "
              f"treat with the same suspicion that caught the XRP/consensus mirage.")
    else:
        print(f"NOT net-positive once gated ({gp['exp']*100:+.3f}%/trade) — the search's in-sample "
              f"high-conviction edge doesn't survive selective + net of fees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

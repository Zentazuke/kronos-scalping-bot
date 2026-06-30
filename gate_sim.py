"""gate_sim.py — dry-run the net-bias gate on the harvested observations.

Before flipping the live gate on, this answers the direct question: *if the bot
had ONLY taken trades where the TA board agreed with the direction (aligned
consensus >= threshold), what would the whole harvest have done?* It reports
aggregate win rate and P&L net of fees, against the take-all baseline, on a time
hold-out, with an equity curve and max drawdown.

This is a PC-side backtest on data the signal was found on — NOT the final word.
The hold-out gives some out-of-sample read, but the real proof is still the live
forward test. Read-only on observations.db; never trades.

    python gate_sim.py --db observations.db --min-consensus 5 --fee-bps 10
    python gate_sim.py --db observations.db --min-consensus 5 --fee-bps 10 --csv equity.csv
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

HOLDOUT_FRAC = 0.30


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


# row = (ts_ms, symbol, aligned_consensus, ret)
Row = Tuple[int, str, float, float]


def load(db_path: str) -> List[Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ts_open, symbol, direction, entry_price, pnl, ta_consensus "
        "FROM trades WHERE status IN ('WIN','LOSS') ORDER BY ts_open"
    ).fetchall()
    conn.close()
    out: List[Row] = []
    for r in rows:
        entry, pnl, cons = _f(r["entry_price"]), _f(r["pnl"]), _f(r["ta_consensus"])
        if not entry or pnl is None or cons is None:
            continue
        is_long = str(r["direction"]).endswith("LONG")
        aligned = cons if is_long else -cons
        out.append((_ts_ms(r["ts_open"]), str(r["symbol"]), aligned, pnl / entry))
    return out


def _perf(nets: Sequence[float]) -> Dict[str, float]:
    """Aggregate stats for a series of NET per-trade returns (ordered by time)."""
    n = len(nets)
    if n == 0:
        return {"n": 0, "win": 0.0, "exp": 0.0, "total": 0.0, "peak": 0.0, "maxdd": 0.0}
    win = sum(1 for x in nets if x > 0) / n
    exp = sum(nets) / n
    cum = 0.0
    peak = 0.0
    maxdd = 0.0
    for x in nets:
        cum += x
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
    return {"n": n, "win": win, "exp": exp, "total": cum, "peak": peak, "maxdd": maxdd}


def _line(label: str, p: Dict[str, float]) -> str:
    if p["n"] == 0:
        return f"  {label:<22} (no trades)"
    return (f"  {label:<22} n={int(p['n']):<5} win {p['win']*100:>4.0f}%  "
            f"net/trade {p['exp']*100:>+7.3f}%  total {p['total']*100:>+8.2f}%  "
            f"maxDD {p['maxdd']*100:>6.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Dry-run the net-bias gate on the harvest")
    ap.add_argument("--db", default="observations.db")
    ap.add_argument("--min-consensus", type=float, default=5.0,
                    help="take a trade only when aligned consensus >= this (default 5)")
    ap.add_argument("--fee-bps", type=float, default=10.0,
                    help="round-trip cost in basis points (default 10 = BNB-ish)")
    ap.add_argument("--csv", default="", help="optional path to write the gated equity curve")
    ap.add_argument("--exclude", default="",
                    help="comma-separated symbols to drop (robustness check, e.g. XRP/USDT)")
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0

    data = load(args.db)
    if not data:
        print("no decided trades carry consensus yet — keep harvesting")
        return 1
    excluded = {s.strip() for s in args.exclude.split(",") if s.strip()}
    if excluded:
        data = [r for r in data if r[1] not in excluded]
        print(f"(excluding {', '.join(sorted(excluded))})")
    data.sort(key=lambda r: r[0])

    thr = args.min_consensus
    split = int(len(data) * (1 - HOLDOUT_FRAC))
    train, hold = data[:split], data[split:]

    def nets(rows: Sequence[Row], gated: bool) -> List[float]:
        return [ret - fee for _t, _s, a, ret in rows if (not gated or a >= thr)]

    all_perf = _perf(nets(data, gated=False))
    gate_perf = _perf(nets(data, gated=True))
    gate_rate = (gate_perf["n"] / all_perf["n"]) if all_perf["n"] else 0.0

    print(f"\n=== GATE SIM — consensus >= {thr:g}, {args.fee_bps:.0f} bps fee, "
          f"{len(data)} decided trades ===")
    print(f"gate passes {gate_perf['n']} of {all_perf['n']} setups "
          f"({gate_rate*100:.0f}%) — the rest would be skipped live\n")

    print("FULL PERIOD (net of fees):")
    print(_line("take-all", all_perf))
    print(_line(f"gated (>= {thr:g})", gate_perf))

    print("\nTIME-SPLIT (gate only):")
    print(_line("gated · in-sample", _perf(nets(train, gated=True))))
    print(_line("gated · hold-out", _perf(nets(hold, gated=True))))
    print(_line("take-all · hold-out", _perf(nets(hold, gated=False))))

    # per-symbol gated
    syms: Dict[str, List[float]] = {}
    for t, s, a, ret in data:
        if a >= thr:
            syms.setdefault(s, []).append(ret - fee)
    if syms:
        print("\nGATED BY SYMBOL (net of fees):")
        for s in sorted(syms, key=lambda k: -_perf(syms[k])["total"]):
            print(_line(s, _perf(syms[s])))

    # verdict
    g_in = _perf(nets(train, gated=True))
    g_oo = _perf(nets(hold, gated=True))
    t_oo = _perf(nets(hold, gated=False))
    print("\n=== read ===")
    if g_oo["n"] < 20:
        print("hold-out has too few gated trades to trust — keep harvesting.")
    elif g_in["exp"] > 0 and g_oo["exp"] > 0 and g_oo["exp"] > t_oo["exp"]:
        print(f"gate is net-positive in-sample ({g_in['exp']*100:+.3f}%) AND out-of-sample "
              f"({g_oo['exp']*100:+.3f}%), and beats take-all OOS ({t_oo['exp']*100:+.3f}%). "
              f"Consistent with a real edge — proceed to the live forward test.")
    elif g_oo["exp"] > 0:
        print(f"gate is net-positive OOS ({g_oo['exp']*100:+.3f}%) but the margin is thin / "
              f"not clearly above take-all ({t_oo['exp']*100:+.3f}%). Promising but fragile — "
              f"the live forward test will settle it.")
    else:
        print(f"gate is NOT net-positive out-of-sample ({g_oo['exp']*100:+.3f}%) — the edge "
              f"doesn't survive costs on held-out data. Don't go live yet.")

    if args.csv:
        cum = 0.0
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["trade_index", "ts_ms", "symbol", "aligned_consensus",
                            "net_return", "cum_net_return"])
                i = 0
                for t, s, a, ret in data:
                    if a >= thr:
                        i += 1
                        cum += ret - fee
                        w.writerow([i, t, s, f"{a:.3f}", f"{ret-fee:.6f}", f"{cum:.6f}"])
            print(f"\nequity curve ({i} gated trades) written to {args.csv}")
        except OSError as exc:
            print(f"could not write csv: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

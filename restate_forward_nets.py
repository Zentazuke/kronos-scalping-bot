"""restate_forward_nets.py — one-off migration: recompute the forward test's net
returns at the REAL round-trip cost (20 bps), so the whole track record is on one
consistent fee basis.

Why (2026-07-03 audit): tsm_forward.db settled its early rows at FEE_BPS=10, but the
live executor pays two taker fills (~20 bps). intraday_tsm_forward.py now charges 20,
which left the history MIXED — old rows flattered by 10 bps/trade vs new rows. Since
gross_ret was stored all along, the fix is exact, not an estimate.

Run ON THE SERVER (where the real DB lives). Dry-run by default:

    python restate_forward_nets.py            # shows before/after, changes nothing
    python restate_forward_nets.py --apply    # writes (backs up the DB first)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import time

DB = "tsm_forward.db"
FEE = 20.0 / 10000.0     # keep in sync with intraday_tsm_forward.FEE_BPS


def headline(rows) -> str:
    nets = [n for n in rows if n is not None]
    if not nets:
        return "no settled directional trades"
    win = sum(1 for n in nets if n > 0) / len(nets)
    return (f"n={len(nets)}  win {win*100:.0f}%  net/trade {sum(nets)/len(nets)*100:+.3f}%  "
            f"total {sum(nets)*100:+.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Restate forward-test nets at the real fee")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry-run)")
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()
    if not os.path.exists(args.db):
        print(f"{args.db} not found — run this on the machine that holds the real DB."); return 1
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT rowid, gross_ret, net_ret FROM forward_trades "
        "WHERE status='SETTLED' AND direction!='FLAT' AND gross_ret IS NOT NULL").fetchall()
    if not rows:
        print("nothing to restate."); return 0
    before = [r[2] for r in rows]
    after = [r[1] - FEE for r in rows]
    changed = sum(1 for b, a in zip(before, after) if b is None or abs(b - a) > 1e-12)
    print(f"=== restatement preview ({args.db}) — fee basis {FEE*1e4:.0f} bps ===")
    print(f"  BEFORE (mixed fees): {headline(before)}")
    print(f"  AFTER  (all 20 bps): {headline(after)}")
    print(f"  rows to update: {changed}/{len(rows)}")
    if not args.apply:
        print("\ndry-run — nothing written. Re-run with --apply to commit."); conn.close(); return 0
    bak = f"{args.db}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    conn.close(); shutil.copy(args.db, bak)
    conn = sqlite3.connect(args.db)
    conn.executemany(
        "UPDATE forward_trades SET net_ret=? WHERE rowid=?",
        [(a, r[0]) for r, a in zip(rows, after)])
    conn.commit(); conn.close()
    print(f"\napplied. backup saved to {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""compare_live_vs_trial.py — diagnose why the LIVE trader and the TRIAL (shadow
forward test) disagree, when they're supposed to be the same strategy.

They use the same gates, so they should pick the same day+coin+direction. This lines
up every live closed trade against the matching trial decision and prints, side by
side: direction, entry price, exit price, and return — so we can SEE whether the gap
is (a) different trades, (b) different directions, or (c) bad fills (entry/exit prices
far apart on the same trade -> testnet execution slippage).

Run it wherever the dashboard runs (the server), where both .db files live:

    python compare_live_vs_trial.py                  # full side-by-side table
    python compare_live_vs_trial.py --alert          # one line + exit 1 on divergence
    python compare_live_vs_trial.py --alert --days 3 # only check the last 3 days

AUDIT FIX (2026-07-03): --alert mode added so reconciliation is a daily cron, not a
thing you remember to do. The original live/trial wiring bug sat unnoticed until a
human eyeballed the divergence; this makes the guard automatic:

    10 1 * * *  cd ~/kronos-scalping-bot && .venv/bin/python compare_live_vs_trial.py \
                --alert --days 3 >> reconcile.log 2>&1
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timedelta, timezone

LIVE_DB = "tsm_live.db"
FWD_DB = "tsm_forward.db"


def live_trades():
    if not os.path.exists(LIVE_DB):
        return {}
    c = sqlite3.connect(LIVE_DB)
    rows = c.execute(
        "SELECT day,symbol,side,entry_price,exit_price,pnl,status FROM positions"
    ).fetchall()
    c.close()
    return {(d, s): dict(side=side, entry=e, exit=x, pnl=p, status=st)
            for (d, s, side, e, x, p, st) in rows}


def trial_trades():
    if not os.path.exists(FWD_DB):
        return {}
    c = sqlite3.connect(FWD_DB)
    rows = c.execute(
        "SELECT decision_day,symbol,direction,entry_price,exit_price,gross_ret,net_ret,status"
        " FROM forward_trades").fetchall()
    c.close()
    return {(d, s): dict(dir=dr, entry=e, exit=x, gross=g, net=n, status=st)
            for (d, s, dr, e, x, g, n, st) in rows}


def _f(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconcile the live book against the trial")
    ap.add_argument("--alert", action="store_true",
                    help="terse mode for cron: one line, exit 1 on any divergence")
    ap.add_argument("--days", type=int, default=0,
                    help="only check decisions from the last N days (0 = all)")
    args = ap.parse_args()
    say = (lambda *a, **k: None) if args.alert else print

    live = live_trades()
    trial = trial_trades()
    keys = sorted(set(live) | set(trial))
    if args.days > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")
        keys = [k for k in keys if k[0] >= cutoff]

    say(f"\n=== LIVE (tsm_live.db) vs TRIAL (tsm_forward.db) — {len(live)} live, "
        f"{len(trial)} trial rows ===\n")
    hdr = f"{'day':<11}{'coin':<10}{'L.side':>7}{'T.dir':>7}{'L.entry':>11}{'T.entry':>11}{'entryΔ%':>8}{'L.exit':>11}{'T.exit':>11}{'L.pnl$':>9}{'T.net%':>8}  flag"
    say(hdr)
    say("-" * len(hdr))

    n_dir_mismatch = n_only_live = n_only_trial = n_big_entry = n_short_skipped = 0
    for k in keys:
        d, s = k
        L = live.get(k)
        T = trial.get(k)
        if L and not T:
            n_only_live += 1
            say(f"{d:<11}{s:<10}{L['side']:>7}{'--':>7}{_f(L['entry']):>11}{'--':>11}{'--':>8}"
                f"{_f(L['exit']):>11}{'--':>11}{_f(L['pnl'],2):>9}{'--':>8}  ONLY-LIVE")
            continue
        if T and not L:
            # only count directional trial trades as "missing from live".
            # FIX (2026-07-07): SHORTs are EXPECTED to be trial-only while the live
            # book runs on spot (long-side validation only) — tracked separately,
            # never alerted, so the alert stays meaningful for missing LONGs.
            if T['dir'] == 'SHORT':
                n_short_skipped += 1
                tag = "short-skipped(spot)"
            elif T['dir'] != 'FLAT':
                n_only_trial += 1
                tag = "ONLY-TRIAL"
            else:
                tag = "(trial-flat)"
            say(f"{d:<11}{s:<10}{'--':>7}{T['dir']:>7}{'--':>11}{_f(T['entry']):>11}{'--':>8}"
                f"{'--':>11}{_f(T['exit']):>11}{'--':>9}{_f((T['net'] or 0)*100,3):>8}  {tag}")
            continue
        # both present — compare
        lside = L['side']
        tdir = T['dir']
        same_dir = (lside == tdir)
        entry_delta = ""
        if L['entry'] and T['entry']:
            ed = (L['entry'] / T['entry'] - 1.0) * 100
            entry_delta = f"{ed:+.2f}"
            if abs(ed) > 0.20:
                n_big_entry += 1
        flags = []
        if not same_dir:
            flags.append("DIR-MISMATCH"); n_dir_mismatch += 1
        if entry_delta and abs(float(entry_delta)) > 0.20:
            flags.append("ENTRY-GAP")
        flag = ",".join(flags) if flags else "ok"
        say(f"{d:<11}{s:<10}{lside:>7}{tdir:>7}{_f(L['entry']):>11}{_f(T['entry']):>11}"
            f"{entry_delta:>8}{_f(L['exit']):>11}{_f(T['exit']):>11}"
            f"{_f(L['pnl'],2):>9}{_f((T['net'] or 0)*100,3):>8}  {flag}")

    diverged = n_dir_mismatch + n_only_live + n_only_trial + n_big_entry
    if args.alert:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        skipped = f" (+{n_short_skipped} shorts spot-skipped, expected)" if n_short_skipped else ""
        if diverged:
            print(f"[{stamp}] RECONCILE ALERT: dir-mismatch {n_dir_mismatch}, only-live {n_only_live}, "
                  f"only-trial {n_only_trial}, entry-gaps {n_big_entry}{skipped} — run without --alert to see rows")
            return 1
        print(f"[{stamp}] reconcile ok — live book matches trial{skipped}")
        return 0
    print("\n=== summary ===")
    print(f"  direction mismatches : {n_dir_mismatch}")
    print(f"  big entry-price gaps : {n_big_entry}  (>0.20% apart — testnet fill slippage)")
    print(f"  trades only live     : {n_only_live}")
    print(f"  trades only trial    : {n_only_trial}  (directional LONGs missing = real problem)")
    print(f"  shorts spot-skipped  : {n_short_skipped}  (expected — live book is long-side only)")
    print("\nRead: many DIR-MISMATCH -> the two are taking different trades (signal-timing).")
    print("      many ENTRY-GAP    -> same trades, but testnet fills are far off the clean")
    print("                           candle price the trial assumes (execution slippage).")
    return 1 if diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())

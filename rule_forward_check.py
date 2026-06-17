"""rule_forward_check.py — live forward test of a LOCKED entry rule on NEW observations.

The entry-rule search keeps surfacing `conviction>=0.9 AND supertrend>=2` as its best
in-sample rule — but it can't be backtested (conviction is live-Kronos, not
reconstructable from candles), and the OOS *inside* the search is a coin flip
(~53% win, half the OOS column negative, a sibling rule at 14% win). The honest
verdict on a rule like this can only come FORWARD: lock it today, then judge it
ONLY on observations the search never saw.

This reads observations.db through the SAME feature definitions as strategy_search
(reusing load_setups — conviction = chosen-side prob, supertrend = direction-aware
ta_supertrend), splits at a lock date, and reports the rule's win%/expectancy on the
post-lock (forward) trades vs the historical ones, net of fees. Read-only.

    python rule_forward_check.py
    python rule_forward_check.py --lock 2026-06-17 --rule "conviction>=0.9,supertrend>=2" --fee-bps 10
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import List, Tuple

from strategy_search import load_setups  # identical feature math to the search

Condition = Tuple[str, float]


def parse_rule(text: str) -> List[Condition]:
    """'conviction>=0.9,supertrend>=2' -> [('conviction',0.9),('supertrend',2.0)]."""
    out: List[Condition] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ">=" in part:
            f, v = part.split(">=")
        elif ">" in part:
            f, v = part.split(">")
        else:
            raise SystemExit(f"condition '{part}' must use >= (e.g. supertrend>=2)")
        out.append((f.strip(), float(v)))
    return out


def passes(feats: dict, rule: List[Condition]) -> bool:
    return all(f in feats and feats[f] >= thr for f, thr in rule)


def _lock_ms(lock: str) -> int:
    return int(datetime.strptime(lock, "%Y-%m-%d")
              .replace(tzinfo=timezone.utc).timestamp() * 1000)


def _stat(rets: List[float], fee: float):
    n = len(rets)
    if n == 0:
        return 0, 0.0, 0.0
    nets = [r - fee for r in rets]
    win = sum(1 for r in rets if r > 0) / n        # win is pre-fee outcome (hit TP)
    exp = sum(nets) / n
    return n, win, exp


def main() -> int:
    ap = argparse.ArgumentParser(description="Forward test a locked entry rule on new observations")
    ap.add_argument("--db", default="observations.db")
    ap.add_argument("--rule", default="conviction>=0.9,supertrend>=2")
    ap.add_argument("--lock", default="2026-06-17",
                    help="observations on/after this UTC date are the FORWARD (out-of-sample) set; "
                         "FIXED so the forward window doesn't move each run")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--min-forward", type=int, default=30, help="forward trades needed before we judge")
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0
    rule = parse_rule(args.rule)
    lock_ms = _lock_ms(args.lock)

    setups = load_setups(args.db)
    if not setups:
        print("no decided observations yet — keep harvesting"); return 1

    hist = [s for s in setups if s.ts < lock_ms]
    fwd = [s for s in setups if s.ts >= lock_ms]
    hist_rule = [s.ret for s in hist if passes(s.feats, rule)]
    fwd_rule = [s.ret for s in fwd if passes(s.feats, rule)]
    fwd_all = [s.ret for s in fwd]

    rn_h, win_h, exp_h = _stat(hist_rule, fee)
    rn_f, win_f, exp_f = _stat(fwd_rule, fee)
    an_f, awin_f, aexp_f = _stat(fwd_all, fee)

    print(f"\n=== LOCKED-RULE FORWARD TEST — {args.rule} ===")
    print(f"lock {args.lock} (forward = on/after) · {args.fee_bps:g}bps · "
          f"2.5/2.5 geometry (50% win = breakeven pre-fee)\n")
    print(f"HISTORICAL (what the search saw):  n={rn_h:<4} win {win_h*100:>3.0f}%  "
          f"net/trade {exp_h*100:>+7.3f}%")
    print(f"FORWARD    (new, never searched):  n={rn_f:<4} win {win_f*100:>3.0f}%  "
          f"net/trade {exp_f*100:>+7.3f}%")
    print(f"  forward take-all baseline:       n={an_f:<4} win {awin_f*100:>3.0f}%  "
          f"net/trade {aexp_f*100:>+7.3f}%")

    print("\n=== read ===")
    if rn_f < args.min_forward:
        need = args.min_forward - rn_f
        print(f"only {rn_f} forward trades so far — need ~{need} more before the verdict means anything. "
              f"This rule fires rarely (high conviction bar), so give it time; check back as it accumulates.")
    elif exp_f > 0 and win_f > 0.5 and exp_f >= aexp_f:
        print(f"FORWARD-POSITIVE: +{exp_f*100:.3f}%/trade at {win_f*100:.0f}% win on {rn_f} unseen trades, "
              f"beating the take-all baseline. The rule is holding up out-of-sample — the first directional "
              f"signal to survive a live forward test. Keep watching; if it persists, it's real.")
    else:
        print(f"FORWARD-FAILS: {exp_f*100:+.3f}%/trade at {win_f*100:.0f}% win on {rn_f} unseen trades — "
              f"the in-sample shine ({win_h*100:.0f}% win historically) did NOT carry forward. Same "
              f"rule-roulette the tracker has logged: green in the search, coin-flip live. Directional "
              f"search confirmed as noise on fresh data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

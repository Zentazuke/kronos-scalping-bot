"""forward_rules.py — auto-enroll every GREEN search rule into a live forward test.

The problem this kills: the entry-rule search surfaces a "green" almost every run,
but those rules can't be backtested (conviction = live Kronos, TA scores are computed
bar-by-bar), and re-running until green then stopping is itself a bias. So instead of
vetting each one by hand, this registry LOCKS every green the instant it appears and
then judges it ONLY on observations recorded after the lock — data the search never
saw. Reality, not a backtest, decides.

Flow:
  * register(rule, lock_date)  — add a rule (idempotent; dedup by normalized text).
  * auto_from_history()        — read the latest search run; if its verdict is green,
                                 register its best rule locked as of today.
  * summary()                  — for each enrolled rule, forward win%/net-per-trade on
                                 post-lock observations, plus a verdict (accruing /
                                 holding / failing). This is what the dashboard shows.

Reuses load_setups (identical feature math to the search) + parse_rule/passes/_stat
from rule_forward_check, so the forward test measures exactly what the search flagged.

    python forward_rules.py --report
    python forward_rules.py --register "conviction>=0.9 AND supertrend>=3"
    python forward_rules.py --auto          # enroll the latest green from search_history.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

from rule_forward_check import _stat, parse_rule, passes
from strategy_search import load_setups

DB_PATH = "forward_rules.db"
OBS_DB = "observations.db"
HISTORY = "search_history.jsonl"
FEE = 0.001          # 10 bps
MIN_FORWARD = 30     # forward trades before a verdict means anything
GREEN_MARK = "BEATS the noise floor AND holds out-of-sample"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _norm(rule_text: str) -> str:
    """Normalize 'a >= 0.9 AND b >= 3' or 'a>=0.9,b>=3' to a canonical key."""
    return rule_text.replace(" AND ", ",").replace(" ", "")


def _lock_ms(lock: str) -> int:
    return int(datetime.strptime(lock, "%Y-%m-%d")
              .replace(tzinfo=timezone.utc).timestamp() * 1000)


def ensure(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS rules ("
                 "rule TEXT PRIMARY KEY, lock_date TEXT, source TEXT, first_seen TEXT)")
    conn.commit()


def register(rule_text: str, lock_date: str | None = None,
             source: str = "manual", db: str = DB_PATH) -> bool:
    """Add a rule locked as of lock_date (default today). Idempotent — returns
    True only when a genuinely new rule was enrolled."""
    norm = _norm(rule_text)
    if not norm:
        return False
    conn = sqlite3.connect(db)
    ensure(conn)
    exists = conn.execute("SELECT 1 FROM rules WHERE rule=?", (norm,)).fetchone()
    if exists:
        conn.close()
        return False
    conn.execute("INSERT INTO rules VALUES (?,?,?,?)",
                 (norm, lock_date or _today(), source, _now()))
    conn.commit()
    conn.close()
    return True


def auto_from_history(history: str = HISTORY, db: str = DB_PATH) -> str | None:
    """If the most recent search run was green, enroll its best rule. Returns the
    rule text if newly enrolled, else None."""
    if not os.path.exists(history):
        return None
    try:
        last = json.loads(open(history, encoding="utf-8").read().strip().split("\n")[-1])
    except (OSError, ValueError):
        return None
    if GREEN_MARK not in str(last.get("verdict", "")):
        return None
    rule = last.get("best_rule")
    if not rule:
        return None
    ts = str(last.get("ts", ""))[:19]
    added = register(rule, _today(), f"search {ts}", db)
    return rule if added else None


def summary(obs_db: str = OBS_DB, db: str = DB_PATH,
            fee: float = FEE, min_forward: int = MIN_FORWARD) -> Dict[str, Any]:
    """Per-rule forward performance on post-lock observations (read-only)."""
    if not os.path.exists(db):
        return {"present": False, "rules": []}
    conn = sqlite3.connect(db)
    ensure(conn)
    regs = [dict(zip(("rule", "lock_date", "source", "first_seen"), r))
            for r in conn.execute(
                "SELECT rule,lock_date,source,first_seen FROM rules ORDER BY first_seen")]
    conn.close()
    setups = load_setups(obs_db) if os.path.exists(obs_db) else []
    out: List[Dict[str, Any]] = []
    for reg in regs:
        rule = parse_rule(reg["rule"])
        lock_ms = _lock_ms(reg["lock_date"])
        fwd = [s.ret for s in setups if s.ts >= lock_ms and passes(s.feats, rule)]
        n, win, exp = _stat(fwd, fee)
        verdict = ("accruing" if n < min_forward
                   else "holding" if (exp > 0 and win > 0.5) else "failing")
        out.append({**reg, "n": n, "win": win, "exp": exp, "verdict": verdict})
    return {"present": True, "rules": out, "min_forward": min_forward}


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-enroll & forward-test green search rules")
    ap.add_argument("--register", metavar="RULE", help='e.g. "conviction>=0.9 AND supertrend>=3"')
    ap.add_argument("--lock", default=None, help="lock date YYYY-MM-DD (default today)")
    ap.add_argument("--auto", action="store_true", help="enroll the latest green from search_history.jsonl")
    ap.add_argument("--report", action="store_true", help="show all enrolled rules' forward results")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--obs", default=OBS_DB)
    args = ap.parse_args()

    if args.register:
        added = register(args.register, args.lock, "manual", args.db)
        print(("enrolled: " if added else "already enrolled: ") + _norm(args.register))
    if args.auto:
        r = auto_from_history(db=args.db)
        print(f"auto-enrolled latest green: {r}" if r else "no new green to enroll (latest run not green, or already enrolled)")

    s = summary(args.obs, args.db)
    if not s["present"] or not s["rules"]:
        print("\nno rules enrolled yet. Run a search (greens auto-enroll), or --register one.")
        return 0
    print(f"\n=== AUTO FORWARD-TEST REGISTRY — {len(s['rules'])} rule(s), "
          f"{FEE*10000:g}bps, need {s['min_forward']} fwd trades to judge ===")
    print(f"{'rule':<34}{'locked':>12}{'fwd n':>7}{'win%':>6}{'net/trade':>11}  verdict")
    for r in s["rules"]:
        print(f"{r['rule'][:33]:<34}{r['lock_date']:>12}{r['n']:>7}"
              f"{r['win']*100:>5.0f}%{r['exp']*100:>+10.3f}%  {r['verdict'].upper()}")
    print("\nHOLDING = forward-positive & >50% win past the threshold · ACCRUING = still "
          "gathering · FAILING = green in search, coin-flip live. Time decides, not the backtest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

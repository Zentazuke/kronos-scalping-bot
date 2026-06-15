"""strategy_search.py — the adaptive search engine (Phases 0-2, research run).

Tries many entry-filter combinations over the labeled observation journal,
scores each honestly, and reports *which ingredients drive an edge and why* —
the "learning algorithm" that experiments instead of repeating one fixed setup.

The honesty guard (the whole point):
  * every rule is evaluated on a time-split — searched on the earlier portion,
    confirmed on a later HOLD-OUT slice the search never optimised on;
  * the best rule's Sharpe is DEFLATED for how many rules were tried (the
    multiple-testing penalty — try enough combos and one always looks great);
  * we report the holdout result and the deflated probability, not the
    cherry-picked in-sample number.

Run it on the server, where observations.db lives (read-only; never trades):

    python strategy_search.py --db observations.db
    python strategy_search.py --db observations.db --max-conditions 2 --top 15

Pure-Python (stdlib only) so it runs anywhere the labeler does.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("bot.search")

_EULER_GAMMA = 0.5772156649015329
MIN_TRADES = 25          # a rule must select at least this many to be trusted (in-sample AND holdout)
HOLDOUT_FRAC = 0.30      # last 30% of time held out for confirmation
HISTORY_JSONL = "search_history.jsonl"  # one machine-readable record per run
HISTORY_TXT = "search_history.txt"       # human-readable append, newest at the bottom


# --------------------------------------------------------------------------- #
# Stats (same definitions as learner.py, kept standalone)                     #
# --------------------------------------------------------------------------- #
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


def _sharpe(returns: Sequence[float]) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    m = sum(returns) / n
    var = sum((r - m) ** 2 for r in returns) / (n - 1)
    sd = var ** 0.5
    return m / sd if sd > 0 else 0.0


def _moments(returns: Sequence[float]) -> Tuple[float, float]:
    n = len(returns)
    if n < 4:
        return 0.0, 3.0
    m = sum(returns) / n
    s2 = sum((r - m) ** 2 for r in returns) / n
    if s2 <= 0:
        return 0.0, 3.0
    sd = s2 ** 0.5
    g3 = sum(((r - m) / sd) ** 3 for r in returns) / n
    g4 = sum(((r - m) / sd) ** 4 for r in returns) / n
    return g3, g4


def _psr(sr: float, n_obs: int, g3: float, g4: float, sr_benchmark: float = 0.0) -> float:
    if n_obs < 2:
        return 0.0
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if denom <= 0:
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return NormalDist().cdf(z)


def _deflated_benchmark_sr(sr_variance: float, n_trials: int) -> float:
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    nd = NormalDist()
    e_max = (1.0 - _EULER_GAMMA) * nd.inv_cdf(1.0 - 1.0 / n_trials) + _EULER_GAMMA * nd.inv_cdf(
        1.0 - 1.0 / (n_trials * math.e)
    )
    return (sr_variance ** 0.5) * e_max


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Setup:
    ts: int
    ret: float                 # per-trade return, pnl / entry_price
    feats: Dict[str, float]    # direction-aware, interpretable signal values


def _f(row: sqlite3.Row, key: str) -> Optional[float]:
    try:
        v = row[key]
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError, IndexError):
        return None


def load_setups(db_path: str) -> List[Setup]:
    """Read decided observations into direction-aware, interpretable signals."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM trades WHERE status IN ('WIN','LOSS') ORDER BY ts_open"
    ).fetchall()
    conn.close()
    out: List[Setup] = []
    for r in rows:
        entry = _f(r, "entry_price")
        pnl = _f(r, "pnl")
        if not entry or pnl is None:
            continue
        is_long = str(r["direction"]).endswith("LONG")
        p_up, p_down = _f(r, "p_up"), _f(r, "p_down")
        pdi, mdi = _f(r, "plus_di"), _f(r, "minus_di")
        feats: Dict[str, float] = {}
        # direction-aware conviction: the chosen side's probability
        if p_up is not None and p_down is not None:
            feats["conviction"] = p_up if is_long else p_down
        # direction-aware trend agreement
        if pdi is not None and mdi is not None:
            feats["di_align"] = (pdi - mdi) if is_long else (mdi - pdi)
        # direction-agnostic context
        for key, name in (
            ("adx", "adx"),
            ("confluence_votes", "votes"),
            ("relative_volume", "rel_volume"),
            ("rsi", "rsi"),
            ("book_imbalance", "book_imb"),
        ):
            v = _f(r, key)
            if v is not None:
                feats[name] = v
        # sentiment / alt-data ingredients — None until the engine is wired in,
        # so they simply won't enter the search until enough rows carry them.
        ss = _f(r, "sent_score")
        if ss is not None:
            feats["sent_aligned"] = ss if is_long else -ss
        ol = _f(r, "outlook_1h")
        if ol is not None:
            feats["outlook_aligned"] = ol if is_long else -ol
        for key, name in (
            ("fear_greed", "fear_greed"),
            ("funding_rate", "funding"),
            ("long_short_ratio", "ls_ratio"),
            ("attention_spike", "attention"),
        ):
            v = _f(r, key)
            if v is not None:
                feats[name] = v
        # TA-board indicators (direction-aware: + = indicator agrees with the trade)
        for key, name in (
            ("ta_macd", "macd"), ("ta_supertrend", "supertrend"), ("ta_stoch", "stoch"),
            ("ta_cci", "cci"), ("ta_boll", "boll"), ("ta_donchian", "donchian"), ("ta_obv", "obv"),
        ):
            v = _f(r, key)
            if v is not None:
                feats[name] = v if is_long else -v
        out.append(Setup(ts=_ts_ms(r["ts_open"]), ret=pnl / entry, feats=feats))
    return out


# --------------------------------------------------------------------------- #
# Search                                                                       #
# --------------------------------------------------------------------------- #
Condition = Tuple[str, float]  # (feature, threshold) -> pass if feats[feat] >= threshold


def _thresholds(setups: List[Setup], feat: str) -> List[float]:
    """A few data-driven cut points (percentiles) so thresholds fit each scale."""
    vals = sorted(s.feats[feat] for s in setups if feat in s.feats)
    if len(vals) < MIN_TRADES:
        return []
    cuts = []
    for q in (0.4, 0.55, 0.7, 0.85):
        idx = min(len(vals) - 1, int(q * len(vals)))
        cuts.append(round(vals[idx], 6))
    return sorted(set(cuts))


def _passes(s: Setup, rule: Sequence[Condition]) -> bool:
    for feat, thr in rule:
        v = s.feats.get(feat)
        if v is None or v < thr:
            return False
    return True


@dataclass
class RuleResult:
    rule: Tuple[Condition, ...]
    n_search: int
    sharpe_search: float
    win_search: float
    n_hold: int
    sharpe_hold: float
    win_hold: float


def _metrics(rets: List[float]) -> Tuple[int, float, float]:
    n = len(rets)
    if n == 0:
        return 0, 0.0, 0.0
    return n, _sharpe(rets), sum(1 for r in rets if r > 0) / n


def search(setups: List[Setup], max_conditions: int) -> Tuple[List[RuleResult], int, float]:
    setups.sort(key=lambda s: s.ts)
    split = int(len(setups) * (1 - HOLDOUT_FRAC))
    train, hold = setups[:split], setups[split:]

    feats = sorted({f for s in setups for f in s.feats})
    conditions: List[Condition] = []
    for feat in feats:
        for thr in _thresholds(setups, feat):
            conditions.append((feat, thr))

    # candidate rules: every combination of 1..max_conditions conditions, but
    # never two conditions on the same feature (a single >= is enough per signal)
    candidates: List[Tuple[Condition, ...]] = []
    for k in range(1, max_conditions + 1):
        for combo in itertools.combinations(conditions, k):
            if len({c[0] for c in combo}) == len(combo):
                candidates.append(combo)

    results: List[RuleResult] = []
    sharpes: List[float] = []
    for rule in candidates:
        tr = [s.ret for s in train if _passes(s, rule)]
        if len(tr) < MIN_TRADES:
            continue
        ns, ss, ws = _metrics(tr)
        hr = [s.ret for s in hold if _passes(s, rule)]
        nh, sh, wh = _metrics(hr)
        results.append(RuleResult(rule, ns, ss, ws, nh, sh, wh))
        sharpes.append(ss)

    var_sr = 0.0
    if len(sharpes) >= 2:
        m = sum(sharpes) / len(sharpes)
        var_sr = sum((x - m) ** 2 for x in sharpes) / (len(sharpes) - 1)
    results.sort(key=lambda r: r.sharpe_search, reverse=True)
    return results, len(candidates), var_sr


def _fmt_rule(rule: Tuple[Condition, ...]) -> str:
    return " AND ".join(f"{f}>={thr:g}" for f, thr in rule)


def _format_history(r: Dict[str, Any]) -> str:
    """Human-readable block for search_history.txt, so you can eyeball the trend
    across runs without parsing JSON."""
    lines = ["=" * 78]
    lines.append(f"{r['ts']}  db={r['db']}  n={r['n_decided']}  "
                 f"max_cond={r['max_conditions']}  trials={r['n_trials']}")
    lines.append(f"take-all: Sharpe {r['take_all_sharpe']:+.3f}, win {r['take_all_win']*100:.0f}%"
                 f"   |   noise floor {r['noise_floor']:.3f}")
    lines.append(f"VERDICT: {r['verdict']}")
    lines.append(f"best: {r['best_rule']}")
    lines.append(f"      in {r['best_in_sharpe']:+.3f}/{r['best_in_n']}   "
                 f"oos {r['best_oos_sharpe']:+.3f}/{r['best_oos_n']}   psr {r['best_psr']*100:.0f}%")
    if r["pattern"]:
        lines.append("pattern: " + ", ".join(f"{k}x{v}" for k, v in r["pattern"].items()))
    lines.append("top rules (in-sample | out-of-sample):")
    for t in r["top"][:8]:
        lines.append(f"  {t['rule'][:48]:<48}  in {t['in_sharpe']:+.2f}/{t['in_n']:<3} "
                     f"{t['in_win']*100:>3.0f}%  |  oos {t['oos_sharpe']:+.2f}/{t['oos_n']:<3} "
                     f"{t['oos_win']*100:>3.0f}%")
    return "\n".join(lines) + "\n"


def _append_history(record: Dict[str, Any]) -> None:
    """Append one run to the persistent history (JSONL + readable text) so every
    test is kept and comparable over time. Best-effort: a logging failure must
    never sink a successful search."""
    try:
        with open(HISTORY_JSONL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        with open(HISTORY_TXT, "a", encoding="utf-8") as fh:
            fh.write(_format_history(record))
        logger.info("\nrun saved to %s (and %s)", HISTORY_JSONL, HISTORY_TXT)
    except OSError as exc:  # pragma: no cover — disk/permission edge
        logger.warning("could not write search history: %s", exc)


def main() -> int:
    p = argparse.ArgumentParser(description="Search entry-filter combinations honestly")
    p.add_argument("--db", default="observations.db")
    p.add_argument("--max-conditions", type=int, default=2, help="max ANDed conditions per rule")
    p.add_argument("--top", type=int, default=15, help="leaderboard size")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    setups = load_setups(args.db)
    if len(setups) < MIN_TRADES * 3:
        logger.info("only %d decided observations — too few to search; keep collecting", len(setups))
        return 1

    base_n, base_sr, base_win = _metrics([s.ret for s in setups])
    logger.info("loaded %d decided observations | take-all: Sharpe %.3f, win %.0f%%",
                len(setups), base_sr, base_win * 100)

    results, n_trials, var_sr = search(setups, args.max_conditions)
    if not results:
        logger.info("no rule selected >=%d trades; lower --max-conditions or collect more data", MIN_TRADES)
        return 1

    logger.info("\n=== searched %d rule combinations ===", n_trials)
    logger.info("%-46s %6s %7s %5s | %6s %7s %5s", "rule", "n", "Sharpe", "win%", "OOS n", "Sharpe", "win%")
    for r in results[: args.top]:
        logger.info("%-46s %6d %7.3f %4.0f%% | %6d %7.3f %4.0f%%",
                    _fmt_rule(r.rule)[:46], r.n_search, r.sharpe_search, r.win_search * 100,
                    r.n_hold, r.sharpe_hold, r.win_hold * 100)

    # honest verdict: did the BEST rule beat the noise floor of trying this many
    # combos, AND hold up on a holdout big enough to trust (>= MIN_TRADES)?
    sr_star = _deflated_benchmark_sr(var_sr, n_trials)
    best = results[0]  # sorted by in-sample Sharpe
    hold_start = int(len(setups) * (1 - HOLDOUT_FRAC))
    hold_rets = [s.ret for s in setups[hold_start:] if _passes(s, best.rule)]
    g3, g4 = _moments(hold_rets)
    psr = _psr(best.sharpe_hold, best.n_hold, g3, g4, 0.0)
    if best.sharpe_search <= sr_star:
        verdict = "even the best rule sits within the noise floor — no real edge yet"
    elif best.n_hold < MIN_TRADES:
        verdict = (f"best rule beats the noise floor in-sample, but its holdout is too small "
                   f"({best.n_hold} trades, need {MIN_TRADES}) to trust — needs more data")
    elif best.sharpe_hold > 0:
        verdict = (f"best rule BEATS the noise floor AND holds out-of-sample "
                   f"(OOS Sharpe {best.sharpe_hold:.3f} over {best.n_hold}, "
                   f"edge-is-real {psr * 100:.0f}%) — worth a closer look")
    else:
        verdict = (f"best rule beats the noise floor in-sample but FAILS out-of-sample "
                   f"(OOS Sharpe {best.sharpe_hold:.3f}) — overfit, not real")

    logger.info("\n=== honesty check ===")
    logger.info("noise floor — expected best Sharpe from %d random tries: %.3f", n_trials, sr_star)
    logger.info("best rule in-sample: Sharpe %.3f over %d trades  (%s)",
                best.sharpe_search, best.n_search, _fmt_rule(best.rule))
    logger.info("VERDICT: %s", verdict)

    survivors: List[RuleResult] = [
        r for r in results
        if r.sharpe_search > sr_star and r.n_hold >= MIN_TRADES and r.sharpe_hold > 0
    ]

    # patterns: ingredients of rules that beat the floor AND held out-of-sample
    logger.info("\n=== patterns (ingredients of rules that beat the floor + held OOS) ===")
    tally: Dict[str, int] = {}
    for r in survivors:
        for feat, _thr in r.rule:
            tally[feat] = tally.get(feat, 0) + 1
    if tally:
        for feat, cnt in sorted(tally.items(), key=lambda kv: kv[1], reverse=True):
            logger.info("  %-12s in %d of %d surviving rules", feat, cnt, len(survivors))
    else:
        logger.info("  (none survived — no consistent winning ingredient on this data yet)")

    # persist this run so every test is kept and comparable over time
    record: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db": args.db,
        "n_decided": len(setups),
        "take_all_sharpe": round(base_sr, 3),
        "take_all_win": round(base_win, 3),
        "max_conditions": args.max_conditions,
        "n_trials": n_trials,
        "noise_floor": round(sr_star, 3),
        "verdict": verdict,
        "best_rule": _fmt_rule(best.rule),
        "best_in_sharpe": round(best.sharpe_search, 3),
        "best_in_n": best.n_search,
        "best_oos_sharpe": round(best.sharpe_hold, 3),
        "best_oos_n": best.n_hold,
        "best_psr": round(psr, 3),
        "survivors": len(survivors),
        "pattern": dict(sorted(tally.items(), key=lambda kv: kv[1], reverse=True)),
        "top": [
            {"rule": _fmt_rule(r.rule), "in_sharpe": round(r.sharpe_search, 3),
             "in_n": r.n_search, "in_win": round(r.win_search, 3),
             "oos_sharpe": round(r.sharpe_hold, 3), "oos_n": r.n_hold,
             "oos_win": round(r.win_hold, 3)}
            for r in results[: args.top]
        ],
    }
    _append_history(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""geometry_search.py — honest take-profit / stop-loss geometry sweep.

The entry-filter search (``strategy_search.py``) asks *which* setups to take.
This asks the other half of the question: given that we take a setup, *where do
the brackets belong*? It re-replays every recorded observation against real
mainnet candles at a grid of take-profit / stop-loss ATR multiples, scores each
geometry on a time hold-out the sweep never optimised on, and **deflates** the
winner for how many geometries were tried — so a lucky ratio cannot masquerade
as an edge.

Why a re-replay (and not the stored WIN/LOSS): every row in ``observations.db``
was labeled at *one* fixed geometry, so its pnl only tells us what that bracket
did. To judge a *different* bracket we must replay the raw price path — exactly
what ``label_observations.simulate`` already does, here parametrised by the
candidate (TP, SL). The pessimistic straddle rule (a bar that touches both legs
counts as the STOP) and the time-stop SCRATCH exit are kept, so the numbers are
honest, not flattering.

Needs network access to fetch mainnet candles, same as the labeler. Runs on the
server where ``observations.db`` lives; never trades.

    python geometry_search.py --db observations.db
    python geometry_search.py --db observations.db --window 48 --json

Pure-Python apart from the candle fetch (ccxt, imported lazily by the labeler).
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Reuse the labeler's candle plumbing and the search's honesty maths verbatim so
# the two tools can never drift apart in how they replay or score.
from label_observations import _fetch_mainnet_candles, _pnl, _to_ms
from strategy_search import (
    _deflated_benchmark_sr,
    _psr,
    _sharpe,
)

logger = logging.getLogger("bot.geometry")

MIN_TRADES = 25          # a geometry must resolve at least this many (in-sample AND holdout)
HOLDOUT_FRAC = 0.30      # last 30% of time held out for confirmation
DEFAULT_WINDOW = 48      # max 5m bars to let a bracket resolve (4h), matches the labeler
HISTORY_JSONL = "geometry_history.jsonl"
HISTORY_TXT = "geometry_history.txt"

# Coarse-to-fine sweep. Stage 1 casts a WIDE net with BIG steps (few combos -> a
# low multiple-testing penalty); only if the coarse winner looks promising do we
# spend a zoom budget on FINE steps around it. This is how we "try lots of
# values" without the combinatorial explosion that would inflate the noise floor.
#
# All values are ATR multiples. The live bracket (TP 1.5 / SL 2.5, R:R 0.6) is
# always scored too, so the report shows how the status quo ranks regardless of
# whether it lands on a grid point.
TP_COARSE: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)   # step 1.0, range 1-4
SL_COARSE: Tuple[float, ...] = (0.5, 1.5, 2.5)        # step 1.0, range 0.5-2.5
ZOOM_STEP: float = 0.5                                 # fine step for stage 2
ZOOM_SPAN: float = 1.0                                 # +/- this far around the coarse winner
MIN_MULT: float = 0.25                                 # never test a bracket tighter than this
LIVE_TP: float = 1.5
LIVE_SL: float = 2.5

# Back-compat single grid (used by the older flat sweep() wrapper and tests).
TP_GRID: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
SL_GRID: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Obs:
    ts: int            # epoch ms of the bar that produced the setup
    symbol: str
    direction: str
    entry: float
    atr: float


def load_obs(db_path: str) -> List[Obs]:
    """Every observation that carries the fields needed to replay a bracket.

    Status is ignored on purpose — we re-derive the outcome from candles, so a
    row labeled WIN at 2.5/2.5 can be re-judged at any other geometry.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ts_open, symbol, direction, entry_price, atr FROM trades "
        "ORDER BY ts_open"
    ).fetchall()
    conn.close()
    out: List[Obs] = []
    for r in rows:
        ts = _to_ms(r["ts_open"])
        entry, atr = r["entry_price"], r["atr"]
        if ts is None or entry in (None, "") or atr in (None, ""):
            continue
        try:
            out.append(Obs(ts, r["symbol"], str(r["direction"]),
                           float(entry), float(atr)))
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# Replay (parametrised by the candidate geometry)                             #
# --------------------------------------------------------------------------- #
def replay(
    future: Sequence[Sequence[float]],
    is_long: bool,
    entry: float,
    atr: float,
    tp_mult: float,
    sl_mult: float,
    window: int,
) -> Optional[Tuple[str, float]]:
    """Replay one (TP, SL) bracket over a setup's future candles.

    ``future`` is the ascending ``[[ms, open, high, low, close], ...]`` slice at
    and after the setup bar. Returns ``(status, exit_price)`` once a leg is
    touched, SCRATCH at the last close if the window elapses untouched, or
    ``None`` if there aren't yet ``window`` candles to decide (too recent).

    A bar that straddles both legs is read as the STOP — the pessimistic call,
    matching the live monitor and the labeler.
    """
    tp = entry + tp_mult * atr if is_long else entry - tp_mult * atr
    sl = entry - sl_mult * atr if is_long else entry + sl_mult * atr
    seen = 0
    for c in future[:window]:
        seen += 1
        high, low = c[2], c[3]
        hit_tp = (high >= tp) if is_long else (low <= tp)
        hit_sl = (low <= sl) if is_long else (high >= sl)
        if hit_sl and hit_tp:
            return ("LOSS", sl)
        if hit_tp:
            return ("WIN", tp)
        if hit_sl:
            return ("LOSS", sl)
    if seen >= window:
        return ("SCRATCH", future[window - 1][4])
    return None


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class GeoResult:
    tp: float
    sl: float
    n_search: int
    sharpe_search: float
    win_search: float
    exp_search: float       # mean per-trade return (expectancy)
    n_hold: int
    sharpe_hold: float
    win_hold: float
    exp_hold: float


def _metrics(rets: Sequence[float]) -> Tuple[int, float, float, float]:
    n = len(rets)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    return n, _sharpe(rets), sum(1 for r in rets if r > 0) / n, sum(rets) / n


Future = Tuple[int, bool, float, float, List[List[float]]]  # ts, is_long, entry, atr, candles


def build_futures(
    db_path: str,
    window: int = DEFAULT_WINDOW,
    candle_provider: Callable[[str, int], List[List[float]]] = _fetch_mainnet_candles,
) -> List[Future]:
    """Fetch each symbol's mainnet candles ONCE and pre-slice every observation's
    future window. Done a single time, then reused across every geometry of every
    sweep stage — the candle fetch is the only slow part, so we never repeat it."""
    obs = load_obs(db_path)
    by_symbol: Dict[str, List[Obs]] = {}
    for o in obs:
        by_symbol.setdefault(o.symbol, []).append(o)
    futures: List[Future] = []
    for symbol, group in by_symbol.items():
        starts = [o.ts for o in group]
        candles = candle_provider(symbol, min(starts) - 5 * 60_000)
        for o in group:
            fut = [c for c in candles if c[0] >= o.ts]
            if len(fut) < window:
                continue  # too recent to resolve any geometry
            futures.append((o.ts, o.direction.endswith("LONG"), o.entry, o.atr, fut))
    return futures


def score_grid(
    futures: Sequence[Future],
    tp_grid: Sequence[float],
    sl_grid: Sequence[float],
    window: int,
) -> List[GeoResult]:
    """Replay every pre-sliced setup at every (TP, SL) in the grid and score each
    geometry on a time hold-out. Returns results sorted by in-sample Sharpe."""
    results: List[GeoResult] = []
    for tp_mult in tp_grid:
        for sl_mult in sl_grid:
            timed: List[Tuple[int, float]] = []
            for ts, is_long, entry, atr, fut in futures:
                res = replay(fut, is_long, entry, atr, tp_mult, sl_mult, window)
                if res is None:
                    continue
                _status, exit_price = res
                direction = "LONG" if is_long else "SHORT"
                timed.append((ts, _pnl(direction, entry, exit_price) / entry))
            timed.sort(key=lambda t: t[0])
            split = int(len(timed) * (1 - HOLDOUT_FRAC))
            train = [r for _t, r in timed[:split]]
            hold = [r for _t, r in timed[split:]]
            if len(train) < MIN_TRADES:
                continue
            ns, ss, ws, es = _metrics(train)
            nh, sh, wh, eh = _metrics(hold)
            results.append(GeoResult(tp_mult, sl_mult, ns, ss, ws, es, nh, sh, wh, eh))
    results.sort(key=lambda r: r.sharpe_search, reverse=True)
    return results


def _var_sr(results: Sequence[GeoResult]) -> float:
    sh = [r.sharpe_search for r in results]
    if len(sh) < 2:
        return 0.0
    m = sum(sh) / len(sh)
    return sum((x - m) ** 2 for x in sh) / (len(sh) - 1)


def _floor(results: Sequence[GeoResult]) -> float:
    """The deflated noise floor for a set of geometries — the Sharpe you'd expect
    the *best* of this many random tries to hit by luck alone."""
    return _deflated_benchmark_sr(_var_sr(results), len(results))


def _zoom_axis(center: float, step: float = ZOOM_STEP, span: float = ZOOM_SPAN,
               lo: float = MIN_MULT) -> Tuple[float, ...]:
    """Fine grid of values around a coarse winner: center +/- span in `step`s,
    clamped so we never test an absurdly tight bracket."""
    n = int(round(span / step))
    vals = sorted({round(center + i * step, 4) for i in range(-n, n + 1)
                   if center + i * step >= lo})
    return tuple(vals)


def sweep(
    db_path: str,
    window: int = DEFAULT_WINDOW,
    tp_grid: Sequence[float] = TP_GRID,
    sl_grid: Sequence[float] = SL_GRID,
    candle_provider: Callable[[str, int], List[List[float]]] = _fetch_mainnet_candles,
) -> Tuple[List[GeoResult], int, float]:
    """Back-compat flat single-grid sweep. Returns (results, n_trials, var)."""
    futures = build_futures(db_path, window, candle_provider)
    results = score_grid(futures, tp_grid, sl_grid, window)
    return results, len(results), _var_sr(results)


@dataclass
class CtfResult:
    coarse: List[GeoResult]
    zoom: List[GeoResult]
    combined: List[GeoResult]      # distinct geometries, best in-sample Sharpe first
    interesting: bool              # did the coarse winner clear its own noise floor?
    zoom_center: Optional[Tuple[float, float]]


def coarse_to_fine(
    db_path: str,
    window: int = DEFAULT_WINDOW,
    candle_provider: Callable[[str, int], List[List[float]]] = _fetch_mainnet_candles,
) -> CtfResult:
    """Two-stage sweep: a broad coarse grid, then — only if the coarse winner
    beats its own deflated noise floor — a fine zoom around it. The live bracket
    is always scored for comparison. Honest by construction: the final winner is
    deflated over EVERY geometry tried (coarse + zoom + live), so the zoom cannot
    smuggle in a multiple-testing edge."""
    futures = build_futures(db_path, window, candle_provider)
    coarse = score_grid(futures, TP_COARSE, SL_COARSE, window)

    interesting = bool(
        coarse and coarse[0].sharpe_search > _floor(coarse)
        and coarse[0].n_search >= MIN_TRADES
    )
    zoom: List[GeoResult] = []
    zoom_center: Optional[Tuple[float, float]] = None
    if interesting:
        best_c = coarse[0]
        zoom_center = (best_c.tp, best_c.sl)
        zoom = score_grid(futures, _zoom_axis(best_c.tp), _zoom_axis(best_c.sl), window)

    # Always score the live bracket so it appears in the comparison even off-grid.
    live = score_grid(futures, (LIVE_TP,), (LIVE_SL,), window)

    distinct: Dict[Tuple[float, float], GeoResult] = {}
    for r in coarse + zoom + live:
        distinct.setdefault((r.tp, r.sl), r)
    combined = sorted(distinct.values(), key=lambda r: r.sharpe_search, reverse=True)
    return CtfResult(coarse, zoom, combined, interesting, zoom_center)


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _fmt_geo(tp: float, sl: float) -> str:
    rr = tp / sl if sl else 0.0
    return f"TP{tp:g}/SL{sl:g} (R:R {rr:.2f})"


def _geo_dict(r: GeoResult, stage: str = "", psr: Optional[float] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "tp": r.tp, "sl": r.sl, "rr": round(r.tp / r.sl, 3) if r.sl else 0.0,
        "in_sharpe": round(r.sharpe_search, 3), "in_n": r.n_search,
        "in_win": round(r.win_search, 3), "in_exp": round(r.exp_search, 6),
        "oos_sharpe": round(r.sharpe_hold, 3), "oos_n": r.n_hold,
        "oos_win": round(r.win_hold, 3), "oos_exp": round(r.exp_hold, 6),
    }
    if stage:
        d["stage"] = stage
    if psr is not None:
        d["psr"] = round(psr, 3)
    return d


def build_record(db_path: str, window: int, ctf: CtfResult) -> Dict[str, Any]:
    """Assemble the machine-readable run record + the honest verdict from a
    coarse-to-fine result. The noise floor deflates over EVERY distinct geometry
    tried across both stages, so the zoom earns no free pass."""
    combined = ctf.combined
    n_trials = len(combined)
    sr_star = _deflated_benchmark_sr(_var_sr(combined), n_trials)
    best = combined[0] if combined else None
    live = next((r for r in combined if r.tp == LIVE_TP and r.sl == LIVE_SL), None)
    coarse_best = ctf.coarse[0] if ctf.coarse else None
    zoom_set = {(r.tp, r.sl) for r in ctf.zoom}

    if best is None:
        verdict = "no geometry resolved enough trades to score — keep collecting"
        psr = 0.0
    else:
        psr = _psr(best.sharpe_hold, best.n_hold, 0.0, 3.0, 0.0)
        if best.sharpe_search <= sr_star:
            tail = ("" if ctf.interesting
                    else " (the coarse pass found nothing promising enough to zoom into)")
            verdict = ("even the best geometry sits within the noise floor of trying "
                       f"{n_trials} — no ratio clearly beats another yet" + tail)
        elif best.n_hold < MIN_TRADES:
            verdict = (f"best geometry beats the noise floor in-sample, but its hold-out is "
                       f"too small ({best.n_hold} trades, need {MIN_TRADES}) — needs more data")
        elif best.sharpe_hold > 0:
            verdict = (f"best geometry {_fmt_geo(best.tp, best.sl)} BEATS the noise floor AND "
                       f"holds out-of-sample (OOS Sharpe {best.sharpe_hold:.3f} over "
                       f"{best.n_hold}, edge-is-real {psr * 100:.0f}%) — worth adopting")
        else:
            verdict = (f"best geometry beats the noise floor in-sample but FAILS out-of-sample "
                       f"(OOS Sharpe {best.sharpe_hold:.3f}) — overfit, keep current brackets")

    record: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db": db_path,
        "window": window,
        "mode": "coarse-to-fine",
        "n_trials": n_trials,
        "coarse_n": len(ctf.coarse),
        "zoom_n": len(ctf.zoom),
        "zoomed": ctf.interesting,
        "zoom_center": (None if ctf.zoom_center is None
                        else {"tp": ctf.zoom_center[0], "sl": ctf.zoom_center[1]}),
        "coarse_best": None if coarse_best is None else _geo_dict(coarse_best),
        "noise_floor": round(sr_star, 3),
        "verdict": verdict,
        "best": None if best is None else _geo_dict(best, psr=psr),
        "live": None if live is None else _geo_dict(live),
        "grid": [
            _geo_dict(r, stage="zoom" if (r.tp, r.sl) in zoom_set else "coarse")
            for r in combined
        ],
    }
    return record


def _format_history(r: Dict[str, Any]) -> str:
    lines = ["=" * 78]
    lines.append(f"{r['ts']}  db={r['db']}  window={r['window']}  geometries={r['n_trials']}")
    stages = f"coarse {r.get('coarse_n', 0)}"
    if r.get("zoomed") and r.get("zoom_center"):
        zc = r["zoom_center"]
        stages += f" -> zoomed {r.get('zoom_n', 0)} around TP{zc['tp']:g}/SL{zc['sl']:g}"
    else:
        stages += " -> no zoom (nothing promising)"
    lines.append(f"search: {stages}")
    lines.append(f"noise floor: {r['noise_floor']:.3f}")
    lines.append(f"VERDICT: {r['verdict']}")
    if r["best"]:
        b = r["best"]
        lines.append(f"best: TP{b['tp']:g}/SL{b['sl']:g} (R:R {b['rr']:.2f})  "
                     f"in {b['in_sharpe']:+.3f}/{b['in_n']} win {b['in_win']*100:.0f}% "
                     f"exp {b['in_exp']*100:+.3f}%  |  oos {b['oos_sharpe']:+.3f}/{b['oos_n']} "
                     f"win {b['oos_win']*100:.0f}% exp {b['oos_exp']*100:+.3f}%")
    if r["live"]:
        lv = r["live"]
        lines.append(f"live: TP{lv['tp']:g}/SL{lv['sl']:g} (R:R {lv['rr']:.2f})  "
                     f"in {lv['in_sharpe']:+.3f}/{lv['in_n']} win {lv['in_win']*100:.0f}% "
                     f"exp {lv['in_exp']*100:+.3f}%  |  oos {lv['oos_sharpe']:+.3f}/{lv['oos_n']} "
                     f"win {lv['oos_win']*100:.0f}% exp {lv['oos_exp']*100:+.3f}%")
    lines.append("grid (in-sample | out-of-sample), best Sharpe first:")
    for g in r["grid"]:
        lines.append(f"  TP{g['tp']:g}/SL{g['sl']:g} R:R{g['rr']:>4.2f}  "
                     f"in {g['in_sharpe']:+.2f}/{g['in_n']:<3} {g['in_win']*100:>3.0f}% "
                     f"exp{g['in_exp']*100:+.3f}%  |  oos {g['oos_sharpe']:+.2f}/{g['oos_n']:<3} "
                     f"{g['oos_win']*100:>3.0f}% exp{g['oos_exp']*100:+.3f}%")
    return "\n".join(lines) + "\n"


def append_history(record: Dict[str, Any]) -> None:
    """Persist one run (JSONL + readable text). Best-effort — a logging failure
    must never sink a successful sweep."""
    try:
        with open(HISTORY_JSONL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        with open(HISTORY_TXT, "a", encoding="utf-8") as fh:
            fh.write(_format_history(record))
        logger.info("\nrun saved to %s (and %s)", HISTORY_JSONL, HISTORY_TXT)
    except OSError as exc:  # pragma: no cover — disk/permission edge
        logger.warning("could not write geometry history: %s", exc)


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep TP/SL bracket geometry honestly")
    p.add_argument("--db", default="observations.db")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"max 5m bars to resolve a bracket (default {DEFAULT_WINDOW} = 4h)")
    p.add_argument("--json", action="store_true", help="print the run record as JSON")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ctf = coarse_to_fine(args.db, args.window)
    if not ctf.combined:
        logger.info("no geometry resolved >=%d trades — keep collecting observations", MIN_TRADES)
        return 1

    record = build_record(args.db, args.window, ctf)

    if args.json:
        print(json.dumps(record))
    else:
        zc = record["zoom_center"]
        stage = (f"coarse {record['coarse_n']} → zoomed {record['zoom_n']} around "
                 f"TP{zc['tp']:g}/SL{zc['sl']:g}" if record["zoomed"] and zc
                 else f"coarse {record['coarse_n']} → no zoom (nothing promising yet)")
        logger.info("=== TP/SL coarse-to-fine sweep — %s, %d geometries, window %d bars ===",
                    stage, record["n_trials"], args.window)
        logger.info("%-22s %6s %6s %7s %5s %9s | %6s %7s %5s %9s",
                    "geometry", "stage", "n", "Sharpe", "win%", "exp", "OOS n", "Sharpe", "win%", "exp")
        for g in record["grid"]:
            logger.info("%-22s %6s %6d %7.3f %4.0f%% %+8.3f%% | %6d %7.3f %4.0f%% %+8.3f%%",
                        _fmt_geo(g["tp"], g["sl"]), g.get("stage", ""), g["in_n"], g["in_sharpe"],
                        g["in_win"] * 100, g["in_exp"] * 100, g["oos_n"], g["oos_sharpe"],
                        g["oos_win"] * 100, g["oos_exp"] * 100)
        logger.info("\n=== honesty check ===")
        logger.info("noise floor — expected best Sharpe from %d geometries: %.3f",
                    record["n_trials"], record["noise_floor"])
        logger.info("VERDICT: %s", record["verdict"])

    append_history(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

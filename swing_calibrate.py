"""swing_calibrate.py — Phase 1 of the medium/position-horizon experiment.

The 5-minute scalping calibration (``calibrate.py``) asked: *when Kronos says
UP for the next bar, how often is it right?* — and the answer was "barely better
than a coin flip." This asks the **swing/position** version of the same
question: *when Kronos says price will be UP in ~1 month, how often is it right?*

It reuses the production inference engine and the calibration maths verbatim,
changing only two things:

  1. the model's forecast **horizon** is set to the holding period (days/weeks/
     month) instead of 1 bar, so its p(up) is a genuine swing forecast;
  2. each forecast is labelled against the realized return over that **same**
     horizon (close[i + hold] vs close[i]), no lookahead — the window for bar i
     ends at bar i, the label looks `hold` bars into the future.

Because daily/4h bars are sparse, this is a *backtest over years of history* —
real statistical power immediately, no waiting weeks. One timeframe per run so a
CPU box can chew a daily pass overnight; run it twice (1d, 4h) to compare.

Offline like calibrate.py: public candles, no API keys, never touches the live
bot or the journals.

    python swing_calibrate.py --timeframe 1d --hold-days 30
    python swing_calibrate.py --timeframe 4h --hold-days 30 --stride 4 --samples 20

Cost warning: the model now generates `hold` bars per path (e.g. 30 daily bars)
instead of 1, so each evaluation is far heavier than the scalping calibration.
Start with --timeframe 1d; for 4h use a bigger --stride / fewer --samples, or
shorten the model's forecast with --model-horizon-bars while still labelling at
the full hold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Final, List, Optional, Sequence, Tuple

import pandas as pd

from backtest import fetch_history
from calibrate import CalibrationReport, analyze_samples
from predictor import MAX_CONTEXT_BARS, MIN_CONTEXT_BARS, KronosInferenceEngine

logger: Final[logging.Logger] = logging.getLogger("bot.swing_calibrate")

# Bars per calendar day for each supported timeframe.
BARS_PER_DAY: Final[Dict[str, int]] = {"1d": 1, "12h": 2, "8h": 3, "4h": 6, "1h": 24}

DEFAULT_SYMBOLS: Final[Tuple[str, ...]] = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT",
)

HISTORY_JSONL: Final[str] = "swing_calibration_history.jsonl"
HISTORY_TXT: Final[str] = "swing_calibration_history.txt"
PROGRESS_EVERY: Final[int] = 25


def horizon_bars(timeframe: str, hold_days: int) -> int:
    """Holding period in *bars* for a timeframe (e.g. 1d/30d -> 30; 4h/30d -> 180)."""
    if timeframe not in BARS_PER_DAY:
        raise ValueError(f"unsupported timeframe {timeframe!r}; use one of {list(BARS_PER_DAY)}")
    return hold_days * BARS_PER_DAY[timeframe]


# --------------------------------------------------------------------------- #
# Replay (swing-horizon: label `hold_bars` into the future, no lookahead)      #
# --------------------------------------------------------------------------- #
async def swing_replay(
    frame: pd.DataFrame,
    engine: KronosInferenceEngine,
    *,
    symbol: str,
    hold_bars: int,
    stride: int = 1,
) -> List[Tuple[float, int]]:
    """Replay the engine over ``frame`` and pair each p(up) with the realized
    direction ``hold_bars`` ahead. Returns ``(p_up, realized_up)`` samples.

    No lookahead: the window for bar i ends AT bar i; the label compares
    close[i + hold_bars] to the anchor close[i]. Bars whose horizon close equals
    the anchor (flat) carry no direction and are skipped.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if hold_bars < 1:
        raise ValueError("hold_bars must be >= 1")
    samples: List[Tuple[float, int]] = []
    first_index: int = MIN_CONTEXT_BARS - 1
    last_index: int = len(frame) - hold_bars - 1  # need hold_bars of future to label
    if last_index < first_index:
        return samples
    evaluations: int = 0
    started: float = time.monotonic()

    for i in range(first_index, last_index + 1, stride):
        window: pd.DataFrame = frame.iloc[
            max(0, i - MAX_CONTEXT_BARS + 1) : i + 1
        ].reset_index(drop=True)
        report = await engine.evaluate(symbol, window)
        evaluations += 1

        anchor: float = float(report.anchor_close)
        future_close: float = float(frame["close"].iloc[i + hold_bars])
        if future_close != anchor:
            samples.append((float(report.p_up), 1 if future_close > anchor else 0))

        if evaluations % PROGRESS_EVERY == 0:
            elapsed: float = time.monotonic() - started
            remaining: int = (last_index - i) // stride
            eta_min: float = remaining * (elapsed / evaluations) / 60.0
            logger.info("%s: %d evaluations (%d labelled) — ~%.0f min left",
                        symbol, evaluations, len(samples), eta_min)
    return samples


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _verdict(rep: CalibrationReport) -> str:
    if rep.skill > 0.0 and rep.best_shrink_lambda >= 0.85:
        return ("Kronos's swing confidence carries real, usable information — "
                "promising; worth a full Phase-2 swing backtest.")
    if rep.skill > 0.0 and rep.best_shrink_lambda >= 0.40:
        return ("directional signal but overconfident — usable WITH shrinkage "
                f"(p' = 0.5 + {rep.best_shrink_lambda:.2f}*(p-0.5)); worth Phase 2.")
    return ("no real swing edge in Kronos's confidence at this horizon — "
            "same story as the 5-minute test.")


def _print_report(label: str, tf: str, hold_days: int, rep: CalibrationReport) -> None:
    print(f"\n=== SWING CALIBRATION {label} | {tf} | ~{hold_days}d hold — "
          f"{rep.samples} labelled forecasts ===")
    print(f"base up-rate          {rep.base_up_rate:.3f}")
    print(f"Brier score           {rep.brier:.4f}  (climatology {rep.climatology_brier:.4f})")
    print(f"skill vs climatology  {rep.skill:+.3f}  (positive = real information)")
    if rep.confident_hit_rate is not None:
        print(f"confident calls       {rep.confident_calls} (p>=0.70 or <=0.30) — "
              f"hit rate {rep.confident_hit_rate:.3f}")
    else:
        print("confident calls       none")
    print(f"best shrink lambda    {rep.best_shrink_lambda:.2f} "
          f"(Brier after shrink {rep.brier_after_shrink:.4f})")
    print("\npredicted -> realized (count)")
    for b in rep.buckets:
        bar = "#" * max(1, round(b.realized_up_rate * 40))
        print(f"  {b.lo:.1f}-{b.hi:.1f}: predicted {b.mean_predicted:.3f} "
              f"realized {b.realized_up_rate:.3f} ({b.count:4d}) {bar}")


def _save(record: Dict) -> None:
    try:
        with open(HISTORY_JSONL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        with open(HISTORY_TXT, "a", encoding="utf-8") as fh:
            fh.write("=" * 78 + "\n")
            fh.write(f"{record['ts']}  {record['timeframe']}  ~{record['hold_days']}d hold  "
                     f"model_horizon={record['model_horizon_bars']} bars\n")
            fh.write(f"pooled: n={record['pooled']['samples']}  skill={record['pooled']['skill']:+.3f}  "
                     f"lambda={record['pooled']['shrink_lambda']:.2f}  "
                     f"confident_hit={record['pooled']['confident_hit_rate']}\n")
            fh.write(f"VERDICT: {record['verdict']}\n")
            for sym, s in record["per_symbol"].items():
                fh.write(f"  {sym:<10} n={s['samples']:<5} skill={s['skill']:+.3f} "
                         f"lambda={s['shrink_lambda']:.2f}\n")
        logger.info("saved to %s (and %s)", HISTORY_JSONL, HISTORY_TXT)
    except OSError as exc:  # pragma: no cover
        logger.warning("could not write swing calibration history: %s", exc)


def _report_dict(rep: CalibrationReport) -> Dict:
    return {
        "samples": rep.samples,
        "base_up_rate": round(rep.base_up_rate, 4),
        "brier": round(rep.brier, 4),
        "skill": round(rep.skill, 4),
        "shrink_lambda": rep.best_shrink_lambda,
        "confident_calls": rep.confident_calls,
        "confident_hit_rate": (None if rep.confident_hit_rate is None
                               else round(rep.confident_hit_rate, 4)),
    }


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
async def run_all(args: argparse.Namespace) -> int:
    tf: str = args.timeframe
    hold_bars: int = horizon_bars(tf, args.hold_days)
    model_horizon: int = args.model_horizon_bars or hold_bars
    symbols: List[str] = [s.strip() for s in args.symbols.split(",") if s.strip()]

    logger.info("swing calibration: %s | ~%dd hold (%d bars) | model horizon %d bars | "
                "%d symbols | stride %d | %d samples | %dd history",
                tf, args.hold_days, hold_bars, model_horizon, len(symbols),
                args.stride, args.samples, args.history_days)

    engine = KronosInferenceEngine(sample_count=args.samples, horizon=model_horizon)

    pooled: List[Tuple[float, int]] = []
    per_symbol: Dict[str, CalibrationReport] = {}
    need: int = MIN_CONTEXT_BARS + hold_bars + 1

    for sym in symbols:
        try:
            frame = await fetch_history(sym, days=args.history_days,
                                        timeframe=tf, exchange_id=args.exchange)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: history fetch failed (%s) — skipped", sym, exc)
            continue
        if len(frame) < need:
            logger.warning("%s: only %d bars, need >= %d — skipped", sym, len(frame), need)
            continue
        samples = await swing_replay(frame, engine, symbol=sym,
                                     hold_bars=hold_bars, stride=args.stride)
        if not samples:
            logger.warning("%s: no labelled forecasts — skipped", sym)
            continue
        rep = analyze_samples(samples)
        per_symbol[sym] = rep
        pooled.extend(samples)
        _print_report(sym, tf, args.hold_days, rep)

    if not pooled:
        logger.error("no labelled forecasts across any symbol — nothing to report")
        return 1

    pooled_rep = analyze_samples(pooled)
    _print_report("ALL SYMBOLS (pooled)", tf, args.hold_days, pooled_rep)
    verdict = _verdict(pooled_rep)
    print(f"\nVERDICT: {verdict}\n")

    _save({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timeframe": tf,
        "hold_days": args.hold_days,
        "hold_bars": hold_bars,
        "model_horizon_bars": model_horizon,
        "stride": args.stride,
        "samples": args.samples,
        "history_days": args.history_days,
        "verdict": verdict,
        "pooled": _report_dict(pooled_rep),
        "per_symbol": {s: _report_dict(r) for s, r in per_symbol.items()},
    })
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Swing/position-horizon Kronos calibration")
    p.add_argument("--timeframe", default="1d", choices=sorted(BARS_PER_DAY),
                   help="bar size (default 1d). Run once per timeframe to compare.")
    p.add_argument("--hold-days", type=int, default=30,
                   help="holding/forecast horizon in calendar days (default 30 ~ 1 month)")
    p.add_argument("--history-days", type=int, default=1460,
                   help="how many days of history to backtest over (default 1460 ~ 4y)")
    p.add_argument("--stride", type=int, default=3,
                   help="evaluate every Nth bar (raise to cut CPU cost)")
    p.add_argument("--samples", type=int, default=20,
                   help="Monte Carlo paths per evaluation (default 20)")
    p.add_argument("--model-horizon-bars", type=int, default=0,
                   help="override the model's forecast length in bars; 0 = use the full hold")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                   help="comma-separated symbols (default = the live bot's 8)")
    p.add_argument("--exchange", default="binance")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    return asyncio.run(run_all(args))


if __name__ == "__main__":
    raise SystemExit(main())

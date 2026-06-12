"""calibrate.py — Phase 9½: offline Kronos reliability replay.

The journal showed Kronos emitting 28–30/30 paths in one direction while
price went the other way (2026-06-11). This tool answers the only question
that matters about that: WHEN KRONOS SAYS 90%, HOW OFTEN IS IT RIGHT?

It replays the production inference engine over historical bars (public
data, no API keys), records predicted p(up) against the realized next-bar
direction, and reports:

  * a reliability table — predicted-probability buckets vs realized up-rate;
  * the Brier score (mean squared probability error; lower is better, and a
    forecaster who always says the base rate scores the "climatology" line);
  * the confident-call hit rate (p >= 0.7 or <= 0.3);
  * the best probability SHRINKAGE lambda: p' = 0.5 + lambda * (p - 0.5),
    fitted by grid search on the Brier score. lambda = 1 means the raw
    probabilities are already calibrated; lambda near 0 means Kronos's
    confidence is noise and the dead band should do the talking.

Offline like backtest.py: never touches the journal, the live bot, or keys.
CPU cost: (bars / stride) full Monte Carlo evaluations — a 14-day, stride-3
run is ~1.3k evaluations; expect hours on CPU. Run it overnight:

    python calibrate.py BTC/USDT --days 14 --stride 3
    python calibrate.py ADA/USDT --days 30 --stride 5 --samples 30

Strict typing: annotated for ``mypy --strict``. Embedded unittest suite uses
an injected fake backend and synthetic frames (zero network):
``python -m unittest calibrate``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import unittest
from dataclasses import dataclass
from typing import Final, List, Optional, Sequence, Tuple

import pandas as pd

from backtest import fetch_history
from predictor import (
    MAX_CONTEXT_BARS,
    MIN_CONTEXT_BARS,
    KronosInferenceEngine,
    PredictionReport,
)

__all__ = ["CalibrationReport", "calibrate", "analyze_samples"]

logger: Final[logging.Logger] = logging.getLogger("bot.calibrate")

BUCKET_COUNT: Final[int] = 10
CONFIDENT_HIGH: Final[float] = 0.70
CONFIDENT_LOW: Final[float] = 0.30
PROGRESS_EVERY: Final[int] = 50


# --------------------------------------------------------------------------- #
# Report containers                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    """One predicted-probability band and what reality did inside it."""

    lo: float
    hi: float
    count: int
    mean_predicted: float
    realized_up_rate: float

    @property
    def gap(self) -> float:
        """Predicted minus realized — positive means overconfident upward."""
        return self.mean_predicted - self.realized_up_rate


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Aggregate reliability statistics for one replay."""

    samples: int
    base_up_rate: float
    brier: float
    climatology_brier: float  # always-predict-base-rate reference
    buckets: Tuple[CalibrationBucket, ...]
    confident_calls: int
    confident_hit_rate: Optional[float]
    best_shrink_lambda: float
    brier_after_shrink: float

    @property
    def skill(self) -> float:
        """Brier skill score vs climatology (positive = real information)."""
        if self.climatology_brier == 0.0:
            return 0.0
        return 1.0 - self.brier / self.climatology_brier


# --------------------------------------------------------------------------- #
# Pure analysis (unit-testable without any model)                              #
# --------------------------------------------------------------------------- #


def _brier(samples: Sequence[Tuple[float, int]], lam: float) -> float:
    total: float = 0.0
    for p, y in samples:
        shrunk: float = 0.5 + lam * (p - 0.5)
        total += (shrunk - y) ** 2
    return total / len(samples)


def analyze_samples(samples: Sequence[Tuple[float, int]]) -> CalibrationReport:
    """Reliability statistics for (predicted_p_up, realized_up) pairs."""
    if not samples:
        raise ValueError("no calibration samples — nothing to analyze")

    n: int = len(samples)
    base: float = sum(y for _, y in samples) / n
    brier: float = _brier(samples, 1.0)
    climatology: float = sum((base - y) ** 2 for _, y in samples) / n

    buckets: List[CalibrationBucket] = []
    width: float = 1.0 / BUCKET_COUNT
    # Index per sample (not range tests) — float bucket edges drop samples
    # that land exactly on a boundary like p = 0.6.
    grouped: List[List[Tuple[float, int]]] = [[] for _ in range(BUCKET_COUNT)]
    for p, y in samples:
        index: int = min(int(p * BUCKET_COUNT), BUCKET_COUNT - 1)
        grouped[index].append((p, y))
    for b in range(BUCKET_COUNT):
        lo: float = b * width
        hi: float = lo + width
        members = grouped[b]
        if not members:
            continue
        buckets.append(
            CalibrationBucket(
                lo=lo,
                hi=hi,
                count=len(members),
                mean_predicted=sum(p for p, _ in members) / len(members),
                realized_up_rate=sum(y for _, y in members) / len(members),
            )
        )

    confident = [
        (p, y)
        for p, y in samples
        if p >= CONFIDENT_HIGH or p <= CONFIDENT_LOW
    ]
    confident_hits: int = sum(
        1
        for p, y in confident
        if (p >= CONFIDENT_HIGH and y == 1) or (p <= CONFIDENT_LOW and y == 0)
    )
    confident_rate: Optional[float] = (
        confident_hits / len(confident) if confident else None
    )

    best_lambda: float = 1.0
    best_score: float = brier
    lam: float = 0.0
    while lam <= 1.0001:
        score: float = _brier(samples, lam)
        if score < best_score - 1e-12:
            best_score, best_lambda = score, round(lam, 2)
        lam += 0.05

    return CalibrationReport(
        samples=n,
        base_up_rate=base,
        brier=brier,
        climatology_brier=climatology,
        buckets=tuple(buckets),
        confident_calls=len(confident),
        confident_hit_rate=confident_rate,
        best_shrink_lambda=best_lambda,
        brier_after_shrink=best_score,
    )


# --------------------------------------------------------------------------- #
# Replay                                                                       #
# --------------------------------------------------------------------------- #


async def calibrate(
    frame: pd.DataFrame,
    engine: KronosInferenceEngine,
    *,
    symbol: str = "BTC/USDT",
    stride: int = 3,
) -> CalibrationReport:
    """Replay the engine over ``frame`` and score its probabilities.

    No lookahead: the window for bar i ends AT bar i; the label is bar i+1.
    Bars whose next close equals the anchor (flat) carry no direction label
    and are skipped.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    samples: List[Tuple[float, int]] = []
    last_index: int = len(frame) - 2
    first_index: int = MIN_CONTEXT_BARS - 1
    evaluations: int = 0
    started: float = time.monotonic()

    for i in range(first_index, last_index + 1, stride):
        window: pd.DataFrame = frame.iloc[
            max(0, i - MAX_CONTEXT_BARS + 1) : i + 1
        ].reset_index(drop=True)
        report: PredictionReport = await engine.evaluate(symbol, window)
        evaluations += 1

        anchor: float = float(report.anchor_close)
        next_close: float = float(frame["close"].iloc[i + 1])
        if next_close != anchor:
            samples.append((float(report.p_up), 1 if next_close > anchor else 0))

        if evaluations % PROGRESS_EVERY == 0:
            elapsed: float = time.monotonic() - started
            remaining: int = (last_index - i) // stride
            eta_min: float = remaining * (elapsed / evaluations) / 60.0
            logger.info(
                "%s: %d evaluations (%d labelled) — ~%.0f min remaining",
                symbol,
                evaluations,
                len(samples),
                eta_min,
            )

    return analyze_samples(samples)


def _print_report(symbol: str, report: CalibrationReport) -> None:
    print(f"\n=== KRONOS CALIBRATION {symbol} — {report.samples} labelled bars ===")
    print(f"base up-rate          {report.base_up_rate:.3f}")
    print(f"Brier score           {report.brier:.4f}  (climatology {report.climatology_brier:.4f})")
    print(f"skill vs climatology  {report.skill:+.3f}  (positive = real information)")
    if report.confident_hit_rate is not None:
        print(
            f"confident calls       {report.confident_calls} "
            f"(p>={CONFIDENT_HIGH} or <={CONFIDENT_LOW}) — hit rate "
            f"{report.confident_hit_rate:.3f}"
        )
    else:
        print("confident calls       none")
    print(f"best shrink lambda    {report.best_shrink_lambda:.2f} "
          f"(Brier after shrink {report.brier_after_shrink:.4f})")
    print("\npredicted -> realized (count)")
    for bucket in report.buckets:
        bar: str = "#" * max(1, round(bucket.realized_up_rate * 40))
        print(
            f"  {bucket.lo:.1f}-{bucket.hi:.1f}: predicted {bucket.mean_predicted:.3f} "
            f"realized {bucket.realized_up_rate:.3f} ({bucket.count:4d}) {bar}"
        )
    print()
    if report.best_shrink_lambda >= 0.85 and report.skill > 0.0:
        print("VERDICT: probabilities usable roughly as-is.")
    elif report.best_shrink_lambda >= 0.4:
        print(
            "VERDICT: overconfident — apply shrinkage "
            f"p' = 0.5 + {report.best_shrink_lambda:.2f}*(p-0.5) before the edge gate."
        )
    else:
        print(
            "VERDICT: confidence is mostly noise — widen the dead band / raise "
            "the edge threshold; consider Phase 10 fine-tuning only after that."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Kronos reliability replay")
    parser.add_argument("symbol", help="e.g. BTC/USDT")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--stride", type=int, default=3,
                        help="evaluate every Nth bar (3 cuts CPU to a third)")
    parser.add_argument("--samples", type=int, default=30,
                        help="Monte Carlo paths per evaluation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    frame: pd.DataFrame = asyncio.run(
        fetch_history(
            args.symbol,
            days=args.days,
            timeframe=args.timeframe,
            exchange_id=args.exchange,
        )
    )
    engine = KronosInferenceEngine(sample_count=args.samples)
    report: CalibrationReport = asyncio.run(
        calibrate(frame, engine, symbol=args.symbol, stride=args.stride)
    )
    _print_report(args.symbol, report)
    return 0


# --------------------------------------------------------------------------- #
# Embedded tests — fake backend, synthetic frames, zero network                #
# --------------------------------------------------------------------------- #


class _BiasedBackend:
    """Always predicts up with a fixed path share — calibration test dummy."""

    def __init__(self, up_share: float) -> None:
        self._up_share: float = up_share

    def sample_terminal_closes(
        self,
        window: pd.DataFrame,
        horizon: int,
        temperature: float,
        sample_count: int,
    ) -> Sequence[float]:
        anchor: float = float(window["close"].iloc[-1])
        ups: int = round(sample_count * self._up_share)
        return [anchor * 1.01] * ups + [anchor * 0.99] * (sample_count - ups)


def _synthetic_frame(num_bars: int, up_every: int) -> pd.DataFrame:
    """Closes rise on multiples of ``up_every``, fall otherwise."""
    closes: List[float] = [100.0]
    for i in range(1, num_bars):
        closes.append(closes[-1] * (1.001 if i % up_every == 0 else 0.9995))
    stamps = pd.date_range("2026-01-01", periods=num_bars, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamps": stamps,
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [100.0] * num_bars,
            "amount": [c * 100.0 for c in closes],
        }
    )


class AnalyzeSamplesTests(unittest.TestCase):
    def test_perfectly_calibrated_keeps_lambda_one(self) -> None:
        # p = 0.8 comes true 80% of the time, p = 0.3 comes true 30%.
        samples: List[Tuple[float, int]] = []
        samples += [(0.8, 1)] * 80 + [(0.8, 0)] * 20
        samples += [(0.3, 1)] * 30 + [(0.3, 0)] * 70
        report = analyze_samples(samples)
        self.assertEqual(report.best_shrink_lambda, 1.0)
        self.assertGreater(report.skill, 0.0)

    def test_pure_noise_shrinks_toward_half(self) -> None:
        # Confident calls that are right only half the time.
        samples = [(0.9, 1)] * 50 + [(0.9, 0)] * 50 + [(0.1, 1)] * 50 + [(0.1, 0)] * 50
        report = analyze_samples(samples)
        self.assertLess(report.best_shrink_lambda, 0.10)
        self.assertEqual(report.confident_calls, 200)
        assert report.confident_hit_rate is not None
        self.assertAlmostEqual(report.confident_hit_rate, 0.5)
        self.assertLessEqual(report.skill, 0.0)

    def test_overconfident_lands_in_the_middle(self) -> None:
        # Says 0.9 / 0.1 but reality is 65% / 35% — directional signal,
        # inflated confidence.
        samples = [(0.9, 1)] * 65 + [(0.9, 0)] * 35 + [(0.1, 1)] * 35 + [(0.1, 0)] * 65
        report = analyze_samples(samples)
        self.assertGreater(report.best_shrink_lambda, 0.15)
        self.assertLess(report.best_shrink_lambda, 0.85)
        self.assertLess(report.brier_after_shrink, report.brier)

    def test_buckets_partition_all_samples(self) -> None:
        samples = [(i / 100.0, i % 2) for i in range(101)]
        report = analyze_samples(samples)
        self.assertEqual(sum(b.count for b in report.buckets), 101)

    def test_empty_refuses(self) -> None:
        with self.assertRaises(ValueError):
            analyze_samples([])


class CalibrateReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_scores_biased_backend_as_overconfident(self) -> None:
        # stride=1 so sampled bars see both up and down successors —
        # stride 2 on an alternating pattern would phase-lock to one label.
        frame = _synthetic_frame(140, up_every=2)  # ~50% up bars
        engine = KronosInferenceEngine(backend=_BiasedBackend(0.9))
        report = await calibrate(frame, engine, symbol="X/Y", stride=1)
        self.assertGreater(report.samples, 20)
        # Backend always says 90% up; reality is a coin flip — heavy shrink.
        self.assertLess(report.best_shrink_lambda, 0.3)

    async def test_replay_never_uses_lookahead(self) -> None:
        # A window ending at bar i must contain bar i as its last close.
        frame = _synthetic_frame(80, up_every=3)
        captured: List[float] = []

        class _Spy(_BiasedBackend):
            def sample_terminal_closes(self, window, horizon, temperature, sample_count):  # type: ignore[no-untyped-def]
                captured.append(float(window["close"].iloc[-1]))
                return super().sample_terminal_closes(
                    window, horizon, temperature, sample_count
                )

        engine = KronosInferenceEngine(backend=_Spy(0.6))
        await calibrate(frame, engine, symbol="X/Y", stride=1)
        expected = [float(frame["close"].iloc[i]) for i in range(MIN_CONTEXT_BARS - 1, len(frame) - 1)]
        self.assertEqual(captured, expected)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and "/" in sys.argv[1]:
        raise SystemExit(main())
    logging.basicConfig(level=logging.INFO)
    unittest.main()

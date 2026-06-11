"""learner.py — Phase 9: meta-labeling filter over the trade journal.

The journal (Phase 7) records what the bot knew at every entry and how the
trade ended. This module learns from that record: a small logistic-regression
classifier estimates, from the decision-time features alone, the probability
that a proposed trade will end as a WIN. It never generates signals — it
learns which of the bot's *own* signals tend to fail.

Deployment contract (set via META_FILTER_MODE in the environment):
  * ``off``     — the filter is never consulted.
  * ``shadow``  — DEFAULT. The filter scores every proposed trade and logs
                  what it *would* have done; it cannot block anything. Its
                  scores are journaled alongside each trade so its shadow
                  record can be audited against reality.
  * ``veto``    — scores below META_MIN_PWIN block the trade. Promote to
                  this mode only after the shadow record has beaten the
                  unfiltered baseline over a meaningful sample.

The model refuses to train (and ``MetaFilter.ready`` stays False) below
``META_MIN_SAMPLES`` decided trades — fitting six weights to a handful of
results is noise, not learning.

Feature vector (direction-signed, so one model serves both sides — each
feature is positive when the evidence favors the proposed trade):
  edge_p     Monte Carlo path share for the proposed side
  di_align   (favoring DI − opposing DI) / 100
  rsi_room   distance from the exhaustion boundary, / 100
  book_align book-imbalance lean toward the proposed side
  atr_ratio  ATR expansion vs baseline (ATR/SMA − 1)
  adx_norm   trend strength / 100

Model math is plain float/numpy — this is the model domain, the same side
of the Decimal boundary as torch and pandas. Training is deterministic:
zero-init weights, fixed-epoch full-batch gradient descent, no RNG.

Train/refresh from the shell (offline, like backtest.py):
    python learner.py train
    python learner.py train --db journal.db --out meta_model.json

Strict typing: annotated for ``mypy --strict``. Embedded unittest suite is
synthetic and file-local (``python -m unittest learner``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, Final, List, Optional, Sequence, Tuple

import numpy as np

from journal import STATUS_LOSS, STATUS_WIN, TradeJournal, TradeRecord

__all__ = [
    "FEATURE_NAMES",
    "MetaModel",
    "MetaFilter",
    "features_from_context",
    "features_from_record",
    "train_model",
]

logger: Final[logging.Logger] = logging.getLogger("bot.learner")

FEATURE_NAMES: Final[Tuple[str, ...]] = (
    "edge_p",
    "di_align",
    "rsi_room",
    "book_align",
    "atr_ratio",
    "adx_norm",
)

META_MIN_SAMPLES: Final[int] = 100  # decided trades before the model is real
TRAIN_EPOCHS: Final[int] = 4_000
TRAIN_LR: Final[float] = 0.05
TRAIN_L2: Final[float] = 0.01
HOLDOUT_FRACTION: Final[float] = 0.2


# --------------------------------------------------------------------------- #
# Features                                                                     #
# --------------------------------------------------------------------------- #


def _f(value: Optional[Decimal], default: float = 0.0) -> float:
    return default if value is None else float(value)


def features_from_context(
    *,
    long_side: bool,
    p_up: Optional[Decimal],
    p_down: Optional[Decimal],
    adx: Optional[Decimal],
    rsi: Optional[Decimal],
    plus_di: Optional[Decimal],
    minus_di: Optional[Decimal],
    book_imbalance: Optional[Decimal],
    atr: Optional[Decimal],
    atr_sma: Optional[Decimal],
) -> List[float]:
    """Direction-signed feature vector from live decision-time values."""
    edge_p: float = _f(p_up if long_side else p_down, 0.5)
    plus: float = _f(plus_di)
    minus: float = _f(minus_di)
    di_align: float = ((plus - minus) if long_side else (minus - plus)) / 100.0
    rsi_value: float = _f(rsi, 50.0)
    rsi_room: float = (
        (70.0 - rsi_value) if long_side else (rsi_value - 30.0)
    ) / 100.0
    imbalance: float = _f(book_imbalance, 0.5)
    book_align: float = (imbalance - 0.5) if long_side else (0.5 - imbalance)
    atr_value: float = _f(atr)
    atr_baseline: float = _f(atr_sma)
    atr_ratio: float = (atr_value / atr_baseline - 1.0) if atr_baseline > 0 else 0.0
    adx_norm: float = _f(adx) / 100.0
    return [edge_p, di_align, rsi_room, book_align, atr_ratio, adx_norm]


def features_from_record(trade: TradeRecord) -> List[float]:
    return features_from_context(
        long_side=trade.is_long,
        p_up=trade.p_up,
        p_down=trade.p_down,
        adx=trade.adx,
        rsi=trade.rsi,
        plus_di=trade.plus_di,
        minus_di=trade.minus_di,
        book_imbalance=trade.book_imbalance,
        atr=trade.atr,
        atr_sma=trade.atr_sma,
    )


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MetaModel:
    """Standardized logistic regression, serialized as plain JSON."""

    weights: Tuple[float, ...]
    bias: float
    feature_mean: Tuple[float, ...]
    feature_std: Tuple[float, ...]
    n_samples: int
    trained_at: str

    def score(self, features: Sequence[float]) -> float:
        """P(win) for one feature vector, clamped to (0, 1)."""
        z: float = self.bias
        for w, x, mu, sigma in zip(
            self.weights, features, self.feature_mean, self.feature_std
        ):
            z += w * ((x - mu) / sigma)
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))

    def save(self, path: Path) -> None:
        payload: Dict[str, object] = {
            "feature_names": list(FEATURE_NAMES),
            "weights": list(self.weights),
            "bias": self.bias,
            "feature_mean": list(self.feature_mean),
            "feature_std": list(self.feature_std),
            "n_samples": self.n_samples,
            "trained_at": self.trained_at,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "MetaModel":
        payload: Dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        names = tuple(payload.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise ValueError(
                f"model file features {names} do not match the running "
                f"feature set {FEATURE_NAMES} — retrain with learner.py"
            )
        return MetaModel(
            weights=tuple(float(w) for w in payload["weights"]),  # type: ignore[union-attr]
            bias=float(payload["bias"]),  # type: ignore[arg-type]
            feature_mean=tuple(float(m) for m in payload["feature_mean"]),  # type: ignore[union-attr]
            feature_std=tuple(float(s) for s in payload["feature_std"]),  # type: ignore[union-attr]
            n_samples=int(payload["n_samples"]),  # type: ignore[arg-type]
            trained_at=str(payload["trained_at"]),
        )


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    n_train: int
    n_holdout: int
    holdout_accuracy: float
    holdout_base_rate: float


def train_model(
    features: Sequence[Sequence[float]],
    labels: Sequence[float],
    *,
    epochs: int = TRAIN_EPOCHS,
    learning_rate: float = TRAIN_LR,
    l2: float = TRAIN_L2,
) -> Tuple[MetaModel, TrainingMetrics]:
    """Deterministic full-batch logistic regression with a chronological
    holdout split (the last 20% — never shuffle time-series trades)."""
    if len(features) != len(labels) or not features:
        raise ValueError("features and labels must be equal-length and non-empty")

    matrix: "np.ndarray" = np.asarray(features, dtype=np.float64)
    target: "np.ndarray" = np.asarray(labels, dtype=np.float64)

    split: int = max(1, int(len(matrix) * (1.0 - HOLDOUT_FRACTION)))
    train_x, hold_x = matrix[:split], matrix[split:]
    train_y, hold_y = target[:split], target[split:]

    mean: "np.ndarray" = train_x.mean(axis=0)
    std: "np.ndarray" = train_x.std(axis=0)
    std[std < 1e-9] = 1.0  # constant features carry no gradient, not a crash
    normalized: "np.ndarray" = (train_x - mean) / std

    weights: "np.ndarray" = np.zeros(matrix.shape[1], dtype=np.float64)
    bias: float = 0.0
    n: float = float(len(normalized))
    for _ in range(epochs):
        z: "np.ndarray" = normalized @ weights + bias
        predictions: "np.ndarray" = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        error: "np.ndarray" = predictions - train_y
        gradient: "np.ndarray" = (normalized.T @ error) / n + l2 * weights
        bias_gradient: float = float(error.mean())
        weights -= learning_rate * gradient
        bias -= learning_rate * bias_gradient

    model: MetaModel = MetaModel(
        weights=tuple(float(w) for w in weights),
        bias=bias,
        feature_mean=tuple(float(m) for m in mean),
        feature_std=tuple(float(s) for s in std),
        n_samples=len(matrix),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )

    if len(hold_x) > 0:
        scores: List[float] = [model.score(row) for row in hold_x.tolist()]
        correct: int = sum(
            1 for s, y in zip(scores, hold_y.tolist()) if (s >= 0.5) == (y >= 0.5)
        )
        accuracy: float = correct / len(scores)
        base_rate: float = max(float(hold_y.mean()), 1.0 - float(hold_y.mean()))
    else:
        accuracy, base_rate = 0.0, 0.0

    return model, TrainingMetrics(
        n_train=int(split),
        n_holdout=int(len(hold_x)),
        holdout_accuracy=accuracy,
        holdout_base_rate=base_rate,
    )


def train_from_journal(db_path: Path, model_path: Path) -> Optional[TrainingMetrics]:
    """CLI worker: decided trades -> features/labels -> trained model file."""
    journal: TradeJournal = TradeJournal(db_path)
    try:
        decided: List[TradeRecord] = [
            t
            for t in journal.closed_trades()
            if t.status in (STATUS_WIN, STATUS_LOSS)
        ]
    finally:
        journal.close()

    if len(decided) < META_MIN_SAMPLES:
        logger.warning(
            "refusing to train: %d decided trades < %d required — "
            "keep the bot journaling",
            len(decided),
            META_MIN_SAMPLES,
        )
        return None

    features: List[List[float]] = [features_from_record(t) for t in decided]
    labels: List[float] = [1.0 if t.status == STATUS_WIN else 0.0 for t in decided]
    model, metrics = train_model(features, labels)
    model.save(model_path)
    logger.info(
        "meta model trained on %d trades -> %s | holdout accuracy %.1f%% "
        "(predict-majority baseline %.1f%%)",
        model.n_samples,
        model_path,
        metrics.holdout_accuracy * 100,
        metrics.holdout_base_rate * 100,
    )
    return metrics


# --------------------------------------------------------------------------- #
# Live-side filter                                                             #
# --------------------------------------------------------------------------- #


class MetaFilter:
    """Thin live wrapper: loads the model file once, scores proposals.

    ``ready`` is False (and ``score`` returns None) until a trained model
    with enough samples exists — absence of evidence never blocks a trade.
    """

    def __init__(
        self, model_path: Path, *, min_samples: int = META_MIN_SAMPLES
    ) -> None:
        self._model_path: Path = model_path
        self._min_samples: int = min_samples
        self._model: Optional[MetaModel] = None
        self._load_attempted: bool = False

    def _load(self) -> None:
        self._load_attempted = True
        if not self._model_path.exists():
            return
        try:
            model: MetaModel = MetaModel.load(self._model_path)
        except (ValueError, KeyError, json.JSONDecodeError):
            logger.error(
                "meta model at %s is unreadable — ignoring it",
                self._model_path,
                exc_info=True,
            )
            return
        if model.n_samples < self._min_samples:
            logger.info(
                "meta model trained on %d < %d samples — staying dormant",
                model.n_samples,
                self._min_samples,
            )
            return
        self._model = model
        logger.info(
            "meta filter armed: %d-sample model from %s",
            model.n_samples,
            model.trained_at,
        )

    @property
    def ready(self) -> bool:
        if not self._load_attempted:
            self._load()
        return self._model is not None

    def score(self, features: Sequence[float]) -> Optional[float]:
        if not self.ready or self._model is None:
            return None
        return self._model.score(features)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the meta-label filter")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--db", default="journal.db")
    parser.add_argument("--out", default="meta_model.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    metrics: Optional[TrainingMetrics] = train_from_journal(
        Path(args.db), Path(args.out)
    )
    return 0 if metrics is not None else 1


# --------------------------------------------------------------------------- #
# Embedded tests                                                               #
# --------------------------------------------------------------------------- #


def _synthetic_dataset(n: int = 240) -> Tuple[List[List[float]], List[float]]:
    """Separable-but-noisy data: wins cluster at favorable feature values.

    Deterministic construction (index arithmetic, no RNG) so every test run
    trains the identical model.
    """
    features: List[List[float]] = []
    labels: List[float] = []
    for i in range(n):
        winning: bool = i % 2 == 0
        wobble: float = ((i * 7) % 11 - 5.0) / 50.0  # deterministic noise
        if winning:
            row: List[float] = [0.62 + wobble / 4, 0.15, 0.10, 0.06, 0.05, 0.32]
        else:
            row = [0.54 + wobble / 4, -0.05, -0.02, -0.04, 0.30, 0.27]
        row = [value + wobble for value in row]
        features.append(row)
        labels.append(1.0 if winning else 0.0)
    return features, labels


class LearnerTests(unittest.TestCase):
    def test_learns_separable_outcomes(self) -> None:
        features, labels = _synthetic_dataset()
        model, metrics = train_model(features, labels)
        self.assertGreaterEqual(metrics.holdout_accuracy, 0.9)
        self.assertEqual(model.n_samples, len(features))

    def test_training_is_deterministic(self) -> None:
        features, labels = _synthetic_dataset()
        first, _ = train_model(features, labels)
        second, _ = train_model(features, labels)
        self.assertEqual(first.weights, second.weights)
        self.assertEqual(first.bias, second.bias)

    def test_scores_are_probabilities(self) -> None:
        features, labels = _synthetic_dataset()
        model, _ = train_model(features, labels)
        for row in features:
            score: float = model.score(row)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_save_load_round_trip(self) -> None:
        import tempfile

        features, labels = _synthetic_dataset()
        model, _ = train_model(features, labels)
        with tempfile.TemporaryDirectory() as tmp:
            path: Path = Path(tmp) / "model.json"
            model.save(path)
            loaded: MetaModel = MetaModel.load(path)
            self.assertEqual(loaded.weights, model.weights)
            self.assertEqual(loaded.score(features[0]), model.score(features[0]))

    def test_filter_dormant_without_model_or_samples(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            missing: MetaFilter = MetaFilter(Path(tmp) / "absent.json")
            self.assertFalse(missing.ready)
            self.assertIsNone(missing.score([0.5] * len(FEATURE_NAMES)))

            features, labels = _synthetic_dataset(40)
            small, _ = train_model(features, labels)
            small_path: Path = Path(tmp) / "small.json"
            small.save(small_path)
            dormant: MetaFilter = MetaFilter(small_path, min_samples=100)
            self.assertFalse(dormant.ready)  # 40 samples < 100 floor

    def test_features_are_direction_signed(self) -> None:
        shared: Dict[str, Optional[Decimal]] = {
            "p_up": Decimal("0.6"),
            "p_down": Decimal("0.4"),
            "adx": Decimal("30"),
            "rsi": Decimal("60"),
            "plus_di": Decimal("28"),
            "minus_di": Decimal("12"),
            "book_imbalance": Decimal("0.6"),
            "atr": Decimal("2"),
            "atr_sma": Decimal("2"),
        }
        long_row: List[float] = features_from_context(long_side=True, **shared)
        short_row: List[float] = features_from_context(long_side=False, **shared)
        # Bullish evidence favors the long and opposes the short.
        self.assertGreater(long_row[1], 0)  # di_align
        self.assertLess(short_row[1], 0)
        self.assertGreater(long_row[3], 0)  # book_align
        self.assertLess(short_row[3], 0)
        # edge_p picks the side's own probability.
        self.assertAlmostEqual(long_row[0], 0.6)
        self.assertAlmostEqual(short_row[0], 0.4)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        raise SystemExit(main())
    logging.basicConfig(level=logging.INFO)
    unittest.main()

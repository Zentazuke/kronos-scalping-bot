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
    "walk_forward",
    "WalkForwardFold",
]

logger: Final[logging.Logger] = logging.getLogger("bot.learner")

FEATURE_NAMES: Final[Tuple[str, ...]] = (
    "edge_p",
    "di_align",
    "rsi_room",
    "book_align",
    "atr_ratio",
    "adx_norm",
    # Phase A microstructure features. Safe to extend while no trained
    # model exists; a saved model records its own feature_names, and the
    # filter stays dormant until retrained with the new vector.
    "spread_bps",
    "rel_volume",
    "depth_align",
    # Phase B v2 features. Extending invalidates any saved v1 model on
    # purpose: MetaModel.load() refuses a feature_names mismatch, so the
    # filter goes dormant until retrained — never scores with stale weights.
    "flow_align",
    "ofi_align",
    "mvwap_align",
    "micro_gap_align",
    "htf_1h_align",
    "htf_4h_align",
    "rsi_1h_room",
    "day_range_pos",
    # Daily macro context. Near-constant within any short harvest window —
    # informative only across regime changes; journaled now so the dataset
    # is regime-aware when that day comes.
    "trend_1d_align",
    "macro_align",
    "dist_30d_high",
    "vol_pct_1d",
)

META_MIN_SAMPLES: Final[int] = 100  # decided trades before the model is real
TRAIN_EPOCHS: Final[int] = 4_000
TRAIN_LR: Final[float] = 0.05
TRAIN_L2: Final[float] = 0.01
HOLDOUT_FRACTION: Final[float] = 0.2
WALK_FORWARD_FOLDS: Final[int] = 5  # expanding-window folds for walk-forward


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
    spread_bps: Optional[Decimal] = None,
    relative_volume: Optional[Decimal] = None,
    depth_imbalance: Optional[Decimal] = None,
    trade_imbalance: Optional[Decimal] = None,
    ofi_rel: Optional[Decimal] = None,
    mvwap_gap_bps: Optional[Decimal] = None,
    microprice_gap_bps: Optional[Decimal] = None,
    trend_1h: Optional[Decimal] = None,
    trend_4h: Optional[Decimal] = None,
    rsi_1h: Optional[Decimal] = None,
    day_range_pos: Optional[Decimal] = None,
    trend_1d: Optional[Decimal] = None,
    macro_trend: Optional[Decimal] = None,
    dist_30d_high: Optional[Decimal] = None,
    vol_pct_1d: Optional[Decimal] = None,
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
    spread: float = _f(spread_bps, 0.0)
    rel_volume: float = _f(relative_volume, 1.0)
    raw_depth: float = _f(depth_imbalance, 0.0)
    depth_align: float = raw_depth if long_side else -raw_depth
    sign: float = 1.0 if long_side else -1.0
    flow_align: float = sign * _f(trade_imbalance, 0.0)
    ofi_align: float = sign * _f(ofi_rel, 0.0)
    mvwap_align: float = sign * _f(mvwap_gap_bps, 0.0) / 100.0
    micro_gap_align: float = sign * _f(microprice_gap_bps, 0.0)
    htf_1h_align: float = sign * _f(trend_1h, 0.0)
    htf_4h_align: float = sign * _f(trend_4h, 0.0)
    rsi_1h_value: float = _f(rsi_1h, 50.0)
    rsi_1h_room: float = (
        (70.0 - rsi_1h_value) if long_side else (rsi_1h_value - 30.0)
    ) / 100.0
    day_pos: float = _f(day_range_pos, 0.5)
    trend_1d_align: float = sign * _f(trend_1d, 0.0)
    macro_align: float = sign * _f(macro_trend, 0.0)
    dist_high: float = _f(dist_30d_high, 0.0)
    vol_regime: float = _f(vol_pct_1d, 0.5)
    return [
        edge_p, di_align, rsi_room, book_align, atr_ratio, adx_norm,
        spread, rel_volume, depth_align,
        flow_align, ofi_align, mvwap_align, micro_gap_align,
        htf_1h_align, htf_4h_align, rsi_1h_room, day_pos,
        trend_1d_align, macro_align, dist_high, vol_regime,
    ]


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
        spread_bps=trade.spread_bps,
        relative_volume=trade.relative_volume,
        depth_imbalance=trade.depth_imbalance,
        trade_imbalance=trade.trade_imbalance,
        ofi_rel=trade.ofi_rel,
        mvwap_gap_bps=trade.mvwap_gap_bps,
        microprice_gap_bps=trade.microprice_gap_bps,
        trend_1h=trade.trend_1h,
        trend_4h=trade.trend_4h,
        rsi_1h=trade.rsi_1h,
        day_range_pos=trade.day_range_pos,
        trend_1d=trade.trend_1d,
        macro_trend=trade.macro_trend,
        dist_30d_high=trade.dist_30d_high,
        vol_pct_1d=trade.vol_pct_1d,
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


def train_from_journal(
    db_paths: "Path | Sequence[Path]", model_path: Path
) -> Optional[TrainingMetrics]:
    """CLI worker: decided trades -> features/labels -> trained model file.

    Accepts one journal or several. Pooling across data-farm variants is
    deliberate and CORRECT here: the meta-labeler estimates
    P(win | setup features), and the harvester's unfiltered setups remove
    exactly the selection bias that a single enforced journal suffers from.
    (Kelly replay, by contrast, must never pool — that stays variant-scoped
    in journal.py.)
    """
    paths: List[Path] = (
        [db_paths] if isinstance(db_paths, Path) else [Path(p) for p in db_paths]
    )
    decided: List[TradeRecord] = []
    for db_path in paths:
        if not db_path.exists():
            logger.warning("journal %s does not exist — skipped", db_path)
            continue
        journal: TradeJournal = TradeJournal(db_path)
        try:
            found: List[TradeRecord] = [
                t
                for t in journal.closed_trades()
                if t.status in (STATUS_WIN, STATUS_LOSS)
            ]
        finally:
            journal.close()
        logger.info("%s: %d decided trades", db_path, len(found))
        decided.extend(found)

    if len(decided) < META_MIN_SAMPLES:
        logger.warning(
            "refusing to train: %d decided trades < %d required — "
            "keep the bot(s) journaling",
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
# Walk-forward validation                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One expanding-window fold: trained on all trades before the test block,
    evaluated on the block that immediately follows (never shuffled — order is
    time)."""

    fold: int
    n_train: int
    n_test: int
    accuracy: float
    base_rate: float

    @property
    def edge(self) -> float:
        """Accuracy above the predict-majority baseline (in [-1, 1])."""
        return self.accuracy - self.base_rate


def _fit_logistic(
    train_x: "np.ndarray",
    train_y: "np.ndarray",
    *,
    epochs: int = TRAIN_EPOCHS,
    learning_rate: float = TRAIN_LR,
    l2: float = TRAIN_L2,
) -> Tuple["np.ndarray", float, "np.ndarray", "np.ndarray"]:
    """Train logistic-regression weights on ALL given rows (no internal
    holdout). Byte-for-byte the same optimizer as ``train_model`` so a fold's
    model matches what production would have fit on the same window."""
    mean: "np.ndarray" = train_x.mean(axis=0)
    std: "np.ndarray" = train_x.std(axis=0)
    std[std < 1e-9] = 1.0
    normalized: "np.ndarray" = (train_x - mean) / std
    weights: "np.ndarray" = np.zeros(train_x.shape[1], dtype=np.float64)
    bias: float = 0.0
    n: float = float(len(normalized))
    for _ in range(epochs):
        z: "np.ndarray" = normalized @ weights + bias
        predictions: "np.ndarray" = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        error: "np.ndarray" = predictions - train_y
        gradient: "np.ndarray" = (normalized.T @ error) / n + l2 * weights
        weights -= learning_rate * gradient
        bias -= learning_rate * float(error.mean())
    return weights, bias, mean, std


def _predict_proba(
    weights: "np.ndarray",
    bias: float,
    mean: "np.ndarray",
    std: "np.ndarray",
    rows: "np.ndarray",
) -> "np.ndarray":
    normalized: "np.ndarray" = (rows - mean) / std
    z: "np.ndarray" = normalized @ weights + bias
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def walk_forward(
    features: Sequence[Sequence[float]],
    labels: Sequence[float],
    *,
    n_folds: int = WALK_FORWARD_FOLDS,
) -> List[WalkForwardFold]:
    """Expanding-window time-series validation.

    Splits the chronological trades into ``n_folds + 1`` contiguous blocks.
    Fold k (1..n_folds) trains on every block before k and tests on block k,
    so each successive fold sees more history and is always tested on unseen,
    later trades. A single holdout (as in ``train_model``) can flatter or
    damn a model on the luck of one split; agreement across folds is the
    real evidence that an edge is stable rather than noise.
    """
    if len(features) != len(labels) or not features:
        raise ValueError("features and labels must be equal-length and non-empty")

    matrix: "np.ndarray" = np.asarray(features, dtype=np.float64)
    target: "np.ndarray" = np.asarray(labels, dtype=np.float64)
    n: int = len(matrix)

    # Keep at least ~2 rows per block; shrink the fold count for small samples.
    if n < (n_folds + 1) * 2:
        n_folds = max(1, n // 2 - 1)

    edges: List[int] = [round(n * i / (n_folds + 1)) for i in range(n_folds + 2)]
    results: List[WalkForwardFold] = []
    for k in range(1, n_folds + 1):
        tr_end: int = edges[k]
        te_start, te_end = edges[k], edges[k + 1]
        if tr_end < 1 or te_end <= te_start:
            continue
        tx, ty = matrix[:tr_end], target[:tr_end]
        ex, ey = matrix[te_start:te_end], target[te_start:te_end]
        weights, bias, mean, std = _fit_logistic(tx, ty)
        probs: "np.ndarray" = _predict_proba(weights, bias, mean, std, ex)
        correct: int = int(np.sum((probs >= 0.5) == (ey >= 0.5)))
        accuracy: float = correct / len(ey)
        base_rate: float = max(float(ey.mean()), 1.0 - float(ey.mean()))
        results.append(
            WalkForwardFold(
                fold=k,
                n_train=int(tr_end),
                n_test=int(len(ey)),
                accuracy=accuracy,
                base_rate=base_rate,
            )
        )
    return results


def walk_forward_from_journal(
    db_paths: "Path | Sequence[Path]", *, n_folds: int = WALK_FORWARD_FOLDS
) -> Optional[List[WalkForwardFold]]:
    """CLI worker: decided trades -> expanding-window folds, logged as a table.

    Pools journals exactly like ``train_from_journal`` (the meta-labeler
    estimates P(win | features); the harvester's unfiltered setups remove the
    selection bias a single enforced journal suffers). Reports nothing to disk
    — walk-forward is a verdict, not a model.
    """
    paths: List[Path] = (
        [db_paths] if isinstance(db_paths, Path) else [Path(p) for p in db_paths]
    )
    decided: List[TradeRecord] = []
    for db_path in paths:
        if not db_path.exists():
            logger.warning("journal %s does not exist — skipped", db_path)
            continue
        journal: TradeJournal = TradeJournal(db_path)
        try:
            found: List[TradeRecord] = [
                t
                for t in journal.closed_trades()
                if t.status in (STATUS_WIN, STATUS_LOSS)
            ]
        finally:
            journal.close()
        logger.info("%s: %d decided trades", db_path, len(found))
        decided.extend(found)

    if len(decided) < META_MIN_SAMPLES:
        logger.warning(
            "refusing to validate: %d decided trades < %d required — "
            "keep the bot(s) journaling",
            len(decided),
            META_MIN_SAMPLES,
        )
        return None

    features: List[List[float]] = [features_from_record(t) for t in decided]
    labels: List[float] = [1.0 if t.status == STATUS_WIN else 0.0 for t in decided]
    folds: List[WalkForwardFold] = walk_forward(features, labels, n_folds=n_folds)

    logger.info(
        "walk-forward: %d expanding-window fold(s) over %d decided trades",
        len(folds),
        len(decided),
    )
    beat: int = 0
    acc_sum: float = 0.0
    edge_sum: float = 0.0
    for f in folds:
        verdict: str = "beats baseline" if f.edge > 0 else "at/below baseline"
        logger.info(
            "  fold %d: train %4d -> test %4d | accuracy %5.1f%% vs baseline "
            "%5.1f%% (edge %+.1f pp) %s",
            f.fold,
            f.n_train,
            f.n_test,
            f.accuracy * 100,
            f.base_rate * 100,
            f.edge * 100,
            verdict,
        )
        beat += 1 if f.edge > 0 else 0
        acc_sum += f.accuracy
        edge_sum += f.edge
    if folds:
        logger.info(
            "walk-forward summary: mean accuracy %.1f%% | mean edge %+.1f pp | "
            "%d/%d folds beat baseline",
            acc_sum / len(folds) * 100,
            edge_sum / len(folds) * 100,
            beat,
            len(folds),
        )
    return folds


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
    parser.add_argument("command", choices=["train", "walkforward"])
    parser.add_argument(
        "--db",
        action="append",
        default=None,
        help="journal database; repeat to pool across data-farm variants "
        "(e.g. --db prod/journal.db --db harvester/journal.db)",
    )
    parser.add_argument("--out", default="meta_model.json")
    parser.add_argument(
        "--folds",
        type=int,
        default=WALK_FORWARD_FOLDS,
        help="walk-forward: number of expanding-window folds",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    db_args: List[str] = args.db if args.db else ["journal.db"]
    paths: List[Path] = [Path(p) for p in db_args]
    if args.command == "walkforward":
        folds = walk_forward_from_journal(paths, n_folds=args.folds)
        return 0 if folds else 1
    metrics: Optional[TrainingMetrics] = train_from_journal(paths, Path(args.out))
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


def _seed_journal(db: Path, variant: str, n_decided: int) -> None:
    """Populate a journal with n_decided WIN/LOSS trades carrying features."""
    from execution import ExecutionResult, ExecutionStatus
    from predictor import SignalDirection

    journal = TradeJournal(db, variant=variant)
    try:
        for i in range(n_decided):
            winning: bool = i % 2 == 0
            result = ExecutionResult(
                status=ExecutionStatus.EXECUTED,
                symbol="BTC/USDT",
                direction=SignalDirection.LONG,
                reason="seed",
                executed_amount=Decimal("0.01"),
                entry_fill_price=Decimal("64000"),
                take_profit_price=Decimal("64100"),
                stop_loss_price=Decimal("63800"),
                take_profit_order_id=f"tp-{variant}-{i}",
                stop_loss_order_id=f"sl-{variant}-{i}",
            )
            trade_id: int = journal.open_trade(
                result,
                adx=Decimal("32" if winning else "26"),
                atr=Decimal("2"),
                atr_sma=Decimal("2"),
                rsi=Decimal("55" if winning else "67"),
                plus_di=Decimal("30" if winning else "22"),
                minus_di=Decimal("12" if winning else "20"),
                book_imbalance=Decimal("0.62" if winning else "0.48"),
                p_up=Decimal("0.62" if winning else "0.54"),
                p_down=Decimal("0.30"),
                confluence_votes=3 if winning else 1,
            )
            journal.close_trade(
                trade_id,
                status=STATUS_WIN if winning else STATUS_LOSS,
                exit_price=Decimal("64100" if winning else "63800"),
                pnl=Decimal("1" if winning else "-2"),
            )
    finally:
        journal.close()


class LearnerTests(unittest.TestCase):
    def test_pooled_training_across_variant_journals(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Each journal alone is below META_MIN_SAMPLES; pooled they pass.
            half: int = META_MIN_SAMPLES // 2 + 10
            _seed_journal(base / "prod.db", "prod", half)
            _seed_journal(base / "harvester.db", "harvester", half)
            out: Path = base / "model.json"
            # Single journal: refused.
            self.assertIsNone(train_from_journal(base / "prod.db", out))
            self.assertFalse(out.exists())
            # Pooled: trains.
            metrics = train_from_journal(
                [base / "prod.db", base / "harvester.db"], out
            )
            self.assertIsNotNone(metrics)
            assert metrics is not None
            self.assertEqual(
                metrics.n_train + metrics.n_holdout, half * 2
            )
            self.assertTrue(out.exists())

    def test_missing_journal_is_skipped_not_fatal(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _seed_journal(base / "prod.db", "prod", META_MIN_SAMPLES + 20)
            out: Path = base / "model.json"
            metrics = train_from_journal(
                [base / "prod.db", base / "missing.db"], out
            )
            self.assertIsNotNone(metrics)

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

    def test_walk_forward_reports_folds_and_beats_baseline(self) -> None:
        features, labels = _synthetic_dataset()
        folds = walk_forward(features, labels, n_folds=4)
        self.assertEqual(len(folds), 4)
        # expanding window: each fold trains on strictly more history.
        self.assertTrue(
            all(folds[i].n_train < folds[i + 1].n_train for i in range(len(folds) - 1))
        )
        # every fold is tested on later, unseen trades.
        self.assertTrue(all(f.n_test > 0 for f in folds))
        # separable data -> mean accuracy clears the majority baseline.
        mean_acc = sum(f.accuracy for f in folds) / len(folds)
        mean_base = sum(f.base_rate for f in folds) / len(folds)
        self.assertGreater(mean_acc, mean_base)

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

    if len(sys.argv) > 1 and sys.argv[1] in ("train", "walkforward"):
        raise SystemExit(main())
    logging.basicConfig(level=logging.INFO)
    unittest.main()

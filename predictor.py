"""predictor.py — Phase 3: machine learning prediction layer.

Kronos foundation-model inference engine for the 5-minute scalping system.
Wraps the Tsinghua 'NeoQuasar/Kronos-small' (24.7M parameter) time-series
model behind a strictly typed asynchronous interface.

Responsibilities (and *only* these — strict modular pipeline):
  * One-time weight/tokenizer (BSQ codebook) initialization at instantiation;
    weights are never reloaded inside the 5-minute execution loops.
  * Hardware-agnostic device mapping: CUDA -> MPS (Apple Silicon) -> CPU.
  * Context slicing: the latest <= 512 confirmed bars from the float64 feed
    DataFrame are handed to the Kronos predictor, which performs the BSQ
    tokenization into the model's expected token tensor structure.
  * Monte Carlo path generation: exactly ``sample_count=30`` independent
    autoregressive trajectories, 1-bar forward horizon, temperature 1.0.
  * Predictive edge sieve: directional signal only on a >= 3-point edge over
    the coin-flip baseline (p >= 0.53, i.e. >= 16/30 paths), with a hard
    dead band (0.48 < p < 0.52) returning STRAT_NEUTRAL to block overtrading
    on statistical noise.
  * Inference failure safe-state: any hardware fault, timeout, OOM, or tensor
    shape mismatch is caught, logged CRITICAL, and resolved to STRAT_NEUTRAL —
    a failed model can never create an accidental live entry.

Upstream packaging note: the Kronos classes (``Kronos``, ``KronosTokenizer``,
``KronosPredictor``) ship from the NeoQuasar/Kronos research repository (not
as ``transformers`` AutoModel classes); weights resolve from the Hugging Face
hub. ``KronosPredictor.predict`` averages across its internal samples, so the
backend draws ``sample_count`` independent single-path samples to obtain the
true terminal-outcome distribution required for divergence scoring.

Domain boundary contract: model I/O stays float-native (torch/pandas domain);
every *decision* value — path counts, probabilities, thresholds, anchors —
is converted to ``decimal.Decimal`` before comparison.

Strict typing: annotated for ``mypy --strict``. Heavy imports (torch, Kronos)
are deferred to real-backend construction so the embedded unittest suite runs
against a mocked backend with no GPU stack present
(``python -m unittest predictor``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum, unique
from typing import Any, Final, List, Optional, Protocol, Sequence, Tuple

import pandas as pd

__all__ = [
    "KronosInferenceEngine",
    "KronosBackend",
    "InferenceBackend",
    "SignalDirection",
    "PredictionReport",
    "detect_device",
]

logger: Final[logging.Logger] = logging.getLogger("bot.predictor")

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

DEFAULT_MODEL_ID: Final[str] = "NeoQuasar/Kronos-small"
DEFAULT_TOKENIZER_ID: Final[str] = "NeoQuasar/Kronos-Tokenizer-base"

SAMPLE_COUNT: Final[int] = 30
HORIZON_BARS: Final[int] = 1
TEMPERATURE: Final[float] = 1.0
MAX_CONTEXT_BARS: Final[int] = 512
MIN_CONTEXT_BARS: Final[int] = 64

#: 3-point statistical edge over the 50% coin-flip baseline.
EDGE_THRESHOLD: Final[Decimal] = Decimal("0.53")
DEAD_BAND_LOW: Final[Decimal] = Decimal("0.48")
DEAD_BAND_HIGH: Final[Decimal] = Decimal("0.52")

INFERENCE_TIMEOUT_S: Final[float] = 120.0

PRICE_QUANTUM: Final[Decimal] = Decimal("0.00000001")

_REQUIRED_COLUMNS: Final[Tuple[str, ...]] = (
    "timestamps",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

# --------------------------------------------------------------------------- #
# Types                                                                        #
# --------------------------------------------------------------------------- #


@unique
class SignalDirection(str, Enum):
    """Definitive directional output of the predictive edge sieve."""

    LONG = "STRAT_LONG"
    SHORT = "STRAT_SHORT"
    NEUTRAL = "STRAT_NEUTRAL"


@dataclass(frozen=True, slots=True)
class PredictionReport:
    """Full observability record of one Monte Carlo inference round."""

    symbol: str
    signal: SignalDirection
    sample_count: int
    paths_up: int
    paths_down: int
    paths_flat: int
    p_up: Decimal
    p_down: Decimal
    anchor_close: Decimal


class InferenceBackend(Protocol):
    """Seam between the edge sieve and the heavyweight model stack.

    Implementations return the terminal close of each independently sampled
    trajectory. The production implementation is ``KronosBackend``; tests
    inject deterministic mocks.
    """

    def sample_terminal_closes(
        self,
        window: pd.DataFrame,
        horizon: int,
        temperature: float,
        sample_count: int,
    ) -> Sequence[float]: ...


# --------------------------------------------------------------------------- #
# Device mapping                                                               #
# --------------------------------------------------------------------------- #


def detect_device() -> str:
    """Fastest available accelerator: CUDA -> MPS -> CPU (graceful fallback)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps: Any = getattr(torch.backends, "mps", None)
    if mps is not None and bool(mps.is_available()):
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# Production backend                                                           #
# --------------------------------------------------------------------------- #


class KronosBackend:
    """Adapter over the upstream KronosPredictor (BSQ tokenizer + 24.7M model).

    Weights and the tokenizer codebook are resolved from the Hugging Face hub
    exactly once, here, at construction. The predictor object then performs
    the OHLCVA-matrix -> token-tensor formatting internally on every call.
    """

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        device: Optional[str] = None,
        max_context: int = MAX_CONTEXT_BARS,
    ) -> None:
        resolved_model: str = model_id or os.getenv("KRONOS_MODEL_ID", DEFAULT_MODEL_ID)
        resolved_tokenizer: str = tokenizer_id or os.getenv(
            "KRONOS_TOKENIZER_ID", DEFAULT_TOKENIZER_ID
        )
        self._device: str = device or detect_device()

        kronos_cls, tokenizer_cls, predictor_cls = self._import_kronos()
        tokenizer: Any = tokenizer_cls.from_pretrained(resolved_tokenizer)
        model: Any = kronos_cls.from_pretrained(resolved_model)
        self._predictor: Any = predictor_cls(
            model, tokenizer, device=self._device, max_context=max_context
        )
        logger.info(
            "Kronos backend ready: model=%s tokenizer=%s device=%s max_context=%d",
            resolved_model,
            resolved_tokenizer,
            self._device,
            max_context,
        )

    @property
    def device(self) -> str:
        return self._device

    @staticmethod
    def _import_kronos() -> Tuple[Any, Any, Any]:
        """Resolve the upstream Kronos classes across known package layouts."""
        repo_path: Optional[str] = os.getenv("KRONOS_REPO_PATH")
        if repo_path:
            resolved_path: str = str(Path(repo_path).expanduser().resolve())
            if resolved_path not in sys.path:
                sys.path.insert(0, resolved_path)
        try:
            from model import (  # type: ignore[import-not-found]
                Kronos,
                KronosPredictor,
                KronosTokenizer,
            )

            return Kronos, KronosTokenizer, KronosPredictor
        except ImportError:
            pass
        try:
            from kronos import (  # type: ignore[import-not-found]
                Kronos,
                KronosPredictor,
                KronosTokenizer,
            )

            return Kronos, KronosTokenizer, KronosPredictor
        except ImportError as exc:
            raise RuntimeError(
                "Kronos package not importable. Install the upstream model "
                "package (github.com/NeoQuasar/Kronos: clone and add to "
                "PYTHONPATH, or pip-install its packaged distribution) so "
                "that `Kronos`, `KronosTokenizer` and `KronosPredictor` "
                "resolve; weights then download from the Hugging Face hub."
            ) from exc

    def sample_terminal_closes(
        self,
        window: pd.DataFrame,
        horizon: int,
        temperature: float,
        sample_count: int,
    ) -> List[float]:
        """Draw ``sample_count`` independent autoregressive trajectories.

        Upstream ``predict`` averages over its internal ``sample_count``, so
        each Monte Carlo path here is one stochastic ``sample_count=1`` call —
        preserving the per-path terminal distribution the edge sieve needs.
        Synchronous and blocking by design: the engine offloads this to a
        worker thread so the event loop never stalls.
        """
        x_df: pd.DataFrame = window[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp: pd.Series = pd.Series(window["timestamps"].to_numpy())

        last_ts: pd.Timestamp = pd.Timestamp(window["timestamps"].iloc[-1])
        bar_delta: pd.Timedelta = pd.Timestamp(
            window["timestamps"].iloc[-1]
        ) - pd.Timestamp(window["timestamps"].iloc[-2])
        y_timestamp: pd.Series = pd.Series(
            [last_ts + bar_delta * (step + 1) for step in range(horizon)]
        )

        terminals: List[float] = []
        for _ in range(sample_count):
            prediction: pd.DataFrame = self._predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=horizon,
                T=temperature,
                top_p=1.0,  # pure temperature sampling — no nucleus truncation
                sample_count=1,
                verbose=False,
            )
            terminals.append(float(prediction["close"].iloc[-1]))
        return terminals


# --------------------------------------------------------------------------- #
# Inference engine                                                             #
# --------------------------------------------------------------------------- #


class KronosInferenceEngine:
    """Asynchronous Monte Carlo edge sieve over the Kronos backend.

    Instantiate once at boot (weights load here); call ``generate_signal``
    from the 5-minute loop. Inference is serialized through an internal lock
    (one model instance, no concurrent device contention) and executed on a
    worker thread under a hard timeout.
    """

    def __init__(
        self,
        *,
        backend: Optional[InferenceBackend] = None,
        sample_count: int = SAMPLE_COUNT,
        horizon: int = HORIZON_BARS,
        temperature: float = TEMPERATURE,
        lookback: int = MAX_CONTEXT_BARS,
        min_context: int = MIN_CONTEXT_BARS,
        edge_threshold: Decimal = EDGE_THRESHOLD,
        dead_band_low: Decimal = DEAD_BAND_LOW,
        dead_band_high: Decimal = DEAD_BAND_HIGH,
        inference_timeout_s: float = INFERENCE_TIMEOUT_S,
    ) -> None:
        if sample_count < 1 or horizon < 1:
            raise ValueError("sample_count and horizon must be >= 1")
        if min_context < 2 or lookback < min_context:
            raise ValueError("require lookback >= min_context >= 2")
        if not dead_band_low < dead_band_high <= edge_threshold:
            raise ValueError("require dead_band_low < dead_band_high <= edge_threshold")

        self._backend: InferenceBackend = (
            backend if backend is not None else KronosBackend()
        )
        self._sample_count: int = sample_count
        self._horizon: int = horizon
        self._temperature: float = temperature
        self._lookback: int = lookback
        self._min_context: int = min_context
        self._edge_threshold: Decimal = edge_threshold
        self._dead_band_low: Decimal = dead_band_low
        self._dead_band_high: Decimal = dead_band_high
        self._inference_timeout_s: float = inference_timeout_s
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Public interface                                                    #
    # ------------------------------------------------------------------ #

    async def generate_signal(
        self, symbol: str, dataframe: pd.DataFrame
    ) -> SignalDirection:
        """Edge-sieved directional signal with an absolute safe-state.

        Any fault on the token execution path — hardware timeout, OOM, tensor
        shape mismatch, malformed context — resolves to STRAT_NEUTRAL.
        """
        try:
            report: PredictionReport = await self.evaluate(symbol, dataframe)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — safe-state boundary, by mandate
            logger.critical(
                "%s: inference failure — safe-state %s emitted",
                symbol,
                SignalDirection.NEUTRAL.value,
                exc_info=True,
            )
            return SignalDirection.NEUTRAL

        logger.info(
            "%s: %s | paths up/down/flat=%d/%d/%d p_up=%.4f p_down=%.4f anchor=%s",
            symbol,
            report.signal.value,
            report.paths_up,
            report.paths_down,
            report.paths_flat,
            report.p_up,
            report.p_down,
            report.anchor_close,
        )
        return report.signal

    async def evaluate(self, symbol: str, dataframe: pd.DataFrame) -> PredictionReport:
        """Run one Monte Carlo round and score it. Raises on any fault."""
        window: pd.DataFrame = self._slice_window(dataframe)
        anchor_close: Decimal = self._dec(float(window["close"].iloc[-1]))

        async with self._lock:
            closes: Sequence[float] = await asyncio.wait_for(
                asyncio.to_thread(
                    self._backend.sample_terminal_closes,
                    window,
                    self._horizon,
                    self._temperature,
                    self._sample_count,
                ),
                timeout=self._inference_timeout_s,
            )

        return self._score_paths(symbol, closes, anchor_close)

    # ------------------------------------------------------------------ #
    # Context slicing                                                     #
    # ------------------------------------------------------------------ #

    def _slice_window(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Validate the feed frame and slice the latest model context."""
        missing: List[str] = [
            col for col in _REQUIRED_COLUMNS if col not in dataframe.columns
        ]
        if missing:
            raise ValueError(f"feed frame missing required columns: {missing}")
        if len(dataframe) < self._min_context:
            raise ValueError(
                f"insufficient context: {len(dataframe)} bars < "
                f"{self._min_context} required"
            )
        return dataframe.tail(self._lookback).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Divergence scoring + edge gate (Decimal domain)                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dec(value: float) -> Decimal:
        return Decimal(str(value)).quantize(PRICE_QUANTUM, rounding=ROUND_HALF_EVEN)

    def _score_paths(
        self, symbol: str, closes: Sequence[float], anchor_close: Decimal
    ) -> PredictionReport:
        if len(closes) != self._sample_count:
            # Tensor shape mismatch analog: the generator did not return the
            # configured trajectory count — never score a partial ensemble.
            raise ValueError(
                f"path-count mismatch: expected {self._sample_count} "
                f"trajectories, received {len(closes)}"
            )

        paths_up: int = 0
        paths_down: int = 0
        paths_flat: int = 0
        for terminal in closes:
            terminal_dec: Decimal = self._dec(terminal)
            if terminal_dec > anchor_close:
                paths_up += 1
            elif terminal_dec < anchor_close:
                paths_down += 1
            else:
                paths_flat += 1

        total: Decimal = Decimal(self._sample_count)
        p_up: Decimal = Decimal(paths_up) / total
        p_down: Decimal = Decimal(paths_down) / total

        signal: SignalDirection = SignalDirection.NEUTRAL
        if self._in_dead_band(p_up) or self._in_dead_band(p_down):
            # Statistical noise zone — block overtrading immediately.
            signal = SignalDirection.NEUTRAL
        elif p_up >= self._edge_threshold:
            signal = SignalDirection.LONG
        elif p_down >= self._edge_threshold:
            signal = SignalDirection.SHORT

        return PredictionReport(
            symbol=symbol,
            signal=signal,
            sample_count=self._sample_count,
            paths_up=paths_up,
            paths_down=paths_down,
            paths_flat=paths_flat,
            p_up=p_up,
            p_down=p_down,
            anchor_close=anchor_close,
        )

    def _in_dead_band(self, probability: Decimal) -> bool:
        return self._dead_band_low < probability < self._dead_band_high


# --------------------------------------------------------------------------- #
# Mock test infrastructure                                                     #
# --------------------------------------------------------------------------- #

_ANCHOR_CLOSE: Final[float] = 100.0
_UP_CLOSE: Final[float] = 101.0
_DOWN_CLOSE: Final[float] = 99.0
_BAR_MS: Final[int] = 5 * 60 * 1_000
_T0_MS: Final[int] = 1_750_000_500_000 - (1_750_000_500_000 % _BAR_MS)


def _test_frame(num_bars: int = 80) -> pd.DataFrame:
    """Feed-layout frame whose final close is exactly the anchor price."""
    ts_ms: List[int] = [_T0_MS + i * _BAR_MS for i in range(num_bars)]
    closes: List[float] = [
        _ANCHOR_CLOSE - 0.1 * (num_bars - 1 - i) for i in range(num_bars)
    ]
    frame: pd.DataFrame = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(ts_ms, unit="ms", utc=True),
            "open": [c - 0.05 for c in closes],
            "high": [c + 0.10 for c in closes],
            "low": [c - 0.15 for c in closes],
            "close": closes,
            "volume": [100.0] * num_bars,
        }
    )
    frame["amount"] = frame["close"] * frame["volume"]
    float_cols: List[str] = ["open", "high", "low", "close", "volume", "amount"]
    frame[float_cols] = frame[float_cols].astype("float64")
    return frame


class _StaticBackend:
    """Mocked tensor output: returns a preset terminal-close distribution."""

    def __init__(self, closes: Sequence[float]) -> None:
        self._closes: Tuple[float, ...] = tuple(closes)
        self.calls: int = 0

    def sample_terminal_closes(
        self,
        window: pd.DataFrame,
        horizon: int,
        temperature: float,
        sample_count: int,
    ) -> Sequence[float]:
        self.calls += 1
        return list(self._closes)


class _ExplodingBackend:
    """Simulates a hardware fault on the token execution path."""

    def sample_terminal_closes(
        self,
        window: pd.DataFrame,
        horizon: int,
        temperature: float,
        sample_count: int,
    ) -> Sequence[float]:
        raise RuntimeError("CUDA error: out of memory")


class _SlowBackend:
    """Simulates a hardware stall longer than the inference timeout."""

    def sample_terminal_closes(
        self,
        window: pd.DataFrame,
        horizon: int,
        temperature: float,
        sample_count: int,
    ) -> Sequence[float]:
        time.sleep(0.5)
        return [_UP_CLOSE] * sample_count


def _engine(backend: InferenceBackend, **overrides: Any) -> KronosInferenceEngine:
    return KronosInferenceEngine(backend=backend, **overrides)


class KronosInferenceEngineTests(unittest.IsolatedAsyncioTestCase):
    """Edge gate, dead band, and safe-state verification on mocked tensors."""

    async def test_15_15_split_returns_neutral(self) -> None:
        backend = _StaticBackend([_UP_CLOSE] * 15 + [_DOWN_CLOSE] * 15)
        engine = _engine(backend)
        report = await engine.evaluate("BTC/USDT", _test_frame())
        self.assertEqual(report.paths_up, 15)
        self.assertEqual(report.paths_down, 15)
        self.assertEqual(report.p_up, Decimal("0.5"))
        self.assertEqual(report.signal, SignalDirection.NEUTRAL)
        self.assertEqual(
            await engine.generate_signal("BTC/USDT", _test_frame()),
            SignalDirection.NEUTRAL,
        )

    async def test_16_14_split_triggers_long(self) -> None:
        backend = _StaticBackend([_UP_CLOSE] * 16 + [_DOWN_CLOSE] * 14)
        engine = _engine(backend)
        report = await engine.evaluate("BTC/USDT", _test_frame())
        self.assertEqual(report.paths_up, 16)
        self.assertGreaterEqual(report.p_up, Decimal("0.53"))
        self.assertEqual(report.signal, SignalDirection.LONG)
        self.assertEqual(
            await engine.generate_signal("BTC/USDT", _test_frame()),
            SignalDirection.LONG,
        )

    async def test_14_16_split_triggers_short(self) -> None:
        backend = _StaticBackend([_UP_CLOSE] * 14 + [_DOWN_CLOSE] * 16)
        engine = _engine(backend)
        report = await engine.evaluate("ADA/USDT", _test_frame())
        self.assertEqual(report.paths_down, 16)
        self.assertGreaterEqual(report.p_down, Decimal("0.53"))
        self.assertEqual(report.signal, SignalDirection.SHORT)

    async def test_flat_paths_never_count_as_direction(self) -> None:
        # 15 up + 15 flat: p_up == 0.5 sits in the dead band -> NEUTRAL.
        backend = _StaticBackend([_UP_CLOSE] * 15 + [_ANCHOR_CLOSE] * 15)
        engine = _engine(backend)
        report = await engine.evaluate("BTC/USDT", _test_frame())
        self.assertEqual(report.paths_flat, 15)
        self.assertEqual(report.signal, SignalDirection.NEUTRAL)

    async def test_dead_band_blocks_immediately(self) -> None:
        # 15/30 = 0.50 -> strictly inside (0.48, 0.52).
        backend = _StaticBackend([_UP_CLOSE] * 15 + [_DOWN_CLOSE] * 15)
        engine = _engine(backend)
        report = await engine.evaluate("BTC/USDT", _test_frame())
        self.assertTrue(Decimal("0.48") < report.p_up < Decimal("0.52"))
        self.assertEqual(report.signal, SignalDirection.NEUTRAL)

    async def test_backend_fault_resolves_to_safe_state(self) -> None:
        engine = _engine(_ExplodingBackend())
        self.assertEqual(
            await engine.generate_signal("BTC/USDT", _test_frame()),
            SignalDirection.NEUTRAL,
        )

    async def test_inference_timeout_resolves_to_safe_state(self) -> None:
        engine = _engine(_SlowBackend(), inference_timeout_s=0.05)
        self.assertEqual(
            await engine.generate_signal("BTC/USDT", _test_frame()),
            SignalDirection.NEUTRAL,
        )

    async def test_path_count_mismatch_resolves_to_safe_state(self) -> None:
        # Backend returns 7 trajectories instead of 30: shape mismatch analog.
        engine = _engine(_StaticBackend([_UP_CLOSE] * 7))
        self.assertEqual(
            await engine.generate_signal("BTC/USDT", _test_frame()),
            SignalDirection.NEUTRAL,
        )
        with self.assertRaises(ValueError):
            await engine.evaluate("BTC/USDT", _test_frame())

    async def test_insufficient_context_resolves_to_safe_state(self) -> None:
        engine = _engine(_StaticBackend([_UP_CLOSE] * 30))
        self.assertEqual(
            await engine.generate_signal("BTC/USDT", _test_frame(num_bars=10)),
            SignalDirection.NEUTRAL,
        )

    async def test_window_sliced_to_lookback(self) -> None:
        captured: List[int] = []

        class _CapturingBackend:
            def sample_terminal_closes(
                self,
                window: pd.DataFrame,
                horizon: int,
                temperature: float,
                sample_count: int,
            ) -> Sequence[float]:
                captured.append(len(window))
                return [_UP_CLOSE] * sample_count

        engine = _engine(_CapturingBackend(), lookback=72)
        await engine.evaluate("BTC/USDT", _test_frame(num_bars=80))
        self.assertEqual(captured, [72])

    async def test_signal_string_contract(self) -> None:
        self.assertEqual(SignalDirection.LONG.value, "STRAT_LONG")
        self.assertEqual(SignalDirection.SHORT.value, "STRAT_SHORT")
        self.assertEqual(SignalDirection.NEUTRAL.value, "STRAT_NEUTRAL")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

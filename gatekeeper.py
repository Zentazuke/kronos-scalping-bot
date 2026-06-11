"""gatekeeper.py — Phase 2: quantitative confluence engine.

Structural gatekeeper deciding whether the current market regime is favorable
enough to permit a Kronos inference at all. Sits between the float-native
ingestion domain (feed.py) and the model/execution pipeline.

Domain boundary contract (per architecture read-back): the incoming pandas
DataFrame and L2OrderBook are float-native. Every value used for indicator
criteria or execution decisions is converted to ``decimal.Decimal`` — quantized
to standard exchange precision — at this boundary. All indicator arithmetic
(ATR, ADX, SMA, volume averages) below runs in pure Decimal: zero binary
floating-point drift in any number that gates capital.

Confluence barriers enforced by ``verify_regime`` (ALL must pass):
  * Trend check       — 14-period ADX (Wilder)  > 25
  * Volatility check  — 14-period ATR (Wilder)  > 20-period SMA of that ATR
  * Volume expansion  — current candle volume   > mean of previous 5 candles
  * L2 book freshness — book ``timestamp_ms`` within 1000 ms of candle close;
                        otherwise the liquidity is flagged stale/phantom.

Directional confirmation (``confluence``, applied AFTER the model proposes a
side — votes, not hard gates; CONFLUENCE_MIN_VOTES of 3 must agree):
  * DI direction      — Wilder +DI > -DI for longs (inverse for shorts)
  * RSI exhaustion    — 14-period Wilder RSI below 70 for longs / above 30
                        for shorts (never buy into overbought, never sell
                        into oversold)
  * Book imbalance    — top-5 L2 resting depth leaning toward the trade side

Strict typing: annotated for ``mypy --strict``. Embedded ``unittest`` suite
at the bottom verifies each barrier blocks independently (run the module
directly, or ``python -m unittest gatekeeper``).
"""

from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Final, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd

from feed import TIMEFRAME_MS, L2OrderBook

__all__ = ["MarketGatekeeper", "RegimeReport", "ConfluenceReport", "Bar"]

logger: Final[logging.Logger] = logging.getLogger("bot.gatekeeper")

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

#: Standard exchange precision scales used when quantizing at the boundary.
PRICE_QUANTUM: Final[Decimal] = Decimal("0.00000001")  # 8 dp — crypto standard
VOLUME_QUANTUM: Final[Decimal] = Decimal("0.00000001")

ATR_PERIOD: Final[int] = 14
ATR_SMA_PERIOD: Final[int] = 20
ADX_PERIOD: Final[int] = 14
VOLUME_LOOKBACK: Final[int] = 5
ADX_THRESHOLD: Final[Decimal] = Decimal("25")
BOOK_MAX_AGE_MS: Final[int] = 1_000

RSI_PERIOD: Final[int] = 14
RSI_OVERBOUGHT: Final[Decimal] = Decimal("70")
RSI_OVERSOLD: Final[Decimal] = Decimal("30")
IMBALANCE_TIERS: Final[int] = 5
CONFLUENCE_MIN_VOTES: Final[int] = 2  # of the 3 directional confirmations

_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")
_HALF: Final[Decimal] = Decimal("0.5")
_HUNDRED: Final[Decimal] = Decimal("100")

# --------------------------------------------------------------------------- #
# Types                                                                        #
# --------------------------------------------------------------------------- #


class Bar(NamedTuple):
    """One confirmed OHLCV bar, fully inside the Decimal domain."""

    ts_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class RegimeReport:
    """Full observability record of a single regime evaluation.

    ``verify_regime`` reduces this to a Boolean; the report itself is kept
    so logs, tests, and post-trade analysis can see exactly which barrier
    blocked (or admitted) a signal.
    """

    sufficient_data: bool
    trend_ok: bool
    volatility_ok: bool
    volume_ok: bool
    book_fresh: bool
    adx: Optional[Decimal] = None
    atr: Optional[Decimal] = None
    atr_sma: Optional[Decimal] = None
    candle_volume: Optional[Decimal] = None
    average_volume: Optional[Decimal] = None
    book_age_ms: Optional[int] = None
    # Directional context for the confluence vote (informational here —
    # these never gate regime admission, only confirm a proposed side).
    plus_di: Optional[Decimal] = None
    minus_di: Optional[Decimal] = None
    rsi: Optional[Decimal] = None
    book_imbalance: Optional[Decimal] = None  # top-5 bid share, 0..1

    @property
    def passed(self) -> bool:
        return (
            self.sufficient_data
            and self.trend_ok
            and self.volatility_ok
            and self.volume_ok
            and self.book_fresh
        )


@dataclass(frozen=True, slots=True)
class ConfluenceReport:
    """Directional confirmation votes for one proposed trade side.

    The model (Kronos) proposes a direction; this report records how many of
    the three independent confirmations agree. Votes, not vetoes: the trade
    proceeds when ``votes >= required`` so a single disagreeing signal can
    never strangle a 5-minute scalper on its own.
    """

    long_side: bool
    di_vote: bool
    rsi_vote: bool
    book_vote: bool
    required: int
    plus_di: Optional[Decimal] = None
    minus_di: Optional[Decimal] = None
    rsi: Optional[Decimal] = None
    book_imbalance: Optional[Decimal] = None

    @property
    def votes(self) -> int:
        return int(self.di_vote) + int(self.rsi_vote) + int(self.book_vote)

    @property
    def passed(self) -> bool:
        return self.votes >= self.required


# --------------------------------------------------------------------------- #
# Gatekeeper                                                                   #
# --------------------------------------------------------------------------- #


class MarketGatekeeper:
    """Quantitative confluence engine — regime admission control.

    Stateless across calls by design: every evaluation derives exclusively
    from the confirmed-bar DataFrame and the L2 snapshot handed in, so the
    gatekeeper can never act on data the feed has not validated.
    """

    def __init__(
        self,
        *,
        atr_period: int = ATR_PERIOD,
        atr_sma_period: int = ATR_SMA_PERIOD,
        adx_period: int = ADX_PERIOD,
        volume_lookback: int = VOLUME_LOOKBACK,
        adx_threshold: Decimal = ADX_THRESHOLD,
        book_max_age_ms: int = BOOK_MAX_AGE_MS,
        timeframe_ms: int = TIMEFRAME_MS,
        rsi_period: int = RSI_PERIOD,
        confluence_min_votes: int = CONFLUENCE_MIN_VOTES,
    ) -> None:
        if min(atr_period, atr_sma_period, adx_period, volume_lookback, rsi_period) < 1:
            raise ValueError("all indicator periods must be >= 1")
        if not 0 <= confluence_min_votes <= 3:
            raise ValueError("confluence_min_votes must be within 0..3")
        self._atr_period: int = atr_period
        self._atr_sma_period: int = atr_sma_period
        self._adx_period: int = adx_period
        self._volume_lookback: int = volume_lookback
        self._adx_threshold: Decimal = adx_threshold
        self._book_max_age_ms: int = book_max_age_ms
        self._timeframe_ms: int = timeframe_ms
        self._rsi_period: int = rsi_period
        self._confluence_min_votes: int = confluence_min_votes

        # Minimum confirmed history for every indicator to be fully formed:
        #   ATR + its SMA window, double-smoothed ADX, RSI, and volume window.
        self._min_bars: int = max(
            atr_period + atr_sma_period + 1,
            2 * adx_period + 1,
            rsi_period + 1,
            volume_lookback + 1,
        )

    # ------------------------------------------------------------------ #
    # Public interface                                                    #
    # ------------------------------------------------------------------ #

    def verify_regime(self, dataframe: pd.DataFrame, l2_order_book: L2OrderBook) -> bool:
        """Definitive admission decision: True only if ALL barriers pass."""
        report: RegimeReport = self.evaluate(dataframe, l2_order_book)

        if not report.sufficient_data:
            logger.warning(
                "regime blocked: insufficient history (%d bars < %d required)",
                len(dataframe),
                self._min_bars,
            )
            return False

        if not report.book_fresh:
            logger.warning(
                "regime blocked: L2 liquidity stale/phantom "
                "(book age=%s ms, limit=%d ms) — depth untrusted",
                report.book_age_ms,
                self._book_max_age_ms,
            )
        if not report.trend_ok:
            logger.info(
                "regime blocked: ADX %s <= %s (range chop)",
                report.adx,
                self._adx_threshold,
            )
        if not report.volatility_ok:
            logger.info(
                "regime blocked: ATR %s <= ATR-SMA%d %s (stagnant market)",
                report.atr,
                self._atr_sma_period,
                report.atr_sma,
            )
        if not report.volume_ok:
            logger.info(
                "regime blocked: volume %s <= %d-bar average %s (no participation)",
                report.candle_volume,
                self._volume_lookback,
                report.average_volume,
            )

        return report.passed

    def evaluate(self, dataframe: pd.DataFrame, l2_order_book: L2OrderBook) -> RegimeReport:
        """Compute all barriers and return the full observability report."""
        bars: List[Bar] = self._to_bars(dataframe)
        if len(bars) < self._min_bars:
            return RegimeReport(
                sufficient_data=False,
                trend_ok=False,
                volatility_ok=False,
                volume_ok=False,
                book_fresh=False,
            )

        atr_series: List[Decimal] = self._atr_series(bars)
        atr: Decimal = atr_series[-1]
        atr_sma: Decimal = self._sma(atr_series, self._atr_sma_period)
        adx, plus_di, minus_di = self._adx(bars)
        rsi: Decimal = self._rsi(bars)
        book_imbalance: Optional[Decimal] = self._book_imbalance(l2_order_book)

        candle_volume: Decimal = bars[-1].volume
        prior: Sequence[Bar] = bars[-(self._volume_lookback + 1) : -1]
        average_volume: Decimal = sum(
            (b.volume for b in prior), _ZERO
        ) / Decimal(len(prior))

        book_age_ms, book_fresh = self._book_freshness(bars[-1].ts_ms, l2_order_book)

        return RegimeReport(
            sufficient_data=True,
            trend_ok=adx > self._adx_threshold,
            volatility_ok=atr > atr_sma,
            volume_ok=candle_volume > average_volume,
            book_fresh=book_fresh,
            adx=adx,
            atr=atr,
            atr_sma=atr_sma,
            candle_volume=candle_volume,
            average_volume=average_volume,
            book_age_ms=book_age_ms,
            plus_di=plus_di,
            minus_di=minus_di,
            rsi=rsi,
            book_imbalance=book_imbalance,
        )

    def confluence(self, report: RegimeReport, *, long_side: bool) -> ConfluenceReport:
        """Count how many directional confirmations agree with the model.

        Each missing input (short history, empty book) counts as a failed
        vote — absent evidence never confirms a trade.
        """
        di_vote: bool = False
        if report.plus_di is not None and report.minus_di is not None:
            di_vote = (
                report.plus_di > report.minus_di
                if long_side
                else report.minus_di > report.plus_di
            )
        rsi_vote: bool = False
        if report.rsi is not None:
            rsi_vote = (
                report.rsi < RSI_OVERBOUGHT if long_side else report.rsi > RSI_OVERSOLD
            )
        book_vote: bool = False
        if report.book_imbalance is not None:
            book_vote = (
                report.book_imbalance > _HALF
                if long_side
                else report.book_imbalance < _HALF
            )
        return ConfluenceReport(
            long_side=long_side,
            di_vote=di_vote,
            rsi_vote=rsi_vote,
            book_vote=book_vote,
            required=self._confluence_min_votes,
            plus_di=report.plus_di,
            minus_di=report.minus_di,
            rsi=report.rsi,
            book_imbalance=report.book_imbalance,
        )

    # ------------------------------------------------------------------ #
    # Domain boundary conversion                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dec(value: float, quantum: Decimal) -> Decimal:
        """Float -> Decimal via str round-trip, quantized to exchange scale.

        ``Decimal(str(x))`` takes the shortest decimal representation of the
        float (what the exchange actually printed), not its raw binary
        expansion — the drift-free way across this boundary.
        """
        return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_EVEN)

    def _to_bars(self, dataframe: pd.DataFrame) -> List[Bar]:
        bars: List[Bar] = []
        for ts, o, h, l, c, v in zip(
            dataframe["timestamps"],
            dataframe["open"],
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            dataframe["volume"],
        ):
            bars.append(
                Bar(
                    ts_ms=int(pd.Timestamp(ts).value // 1_000_000),
                    open=self._dec(float(o), PRICE_QUANTUM),
                    high=self._dec(float(h), PRICE_QUANTUM),
                    low=self._dec(float(l), PRICE_QUANTUM),
                    close=self._dec(float(c), PRICE_QUANTUM),
                    volume=self._dec(float(v), VOLUME_QUANTUM),
                )
            )
        return bars

    # ------------------------------------------------------------------ #
    # Indicators (pure Decimal, Wilder formulations)                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _true_ranges(bars: Sequence[Bar]) -> List[Decimal]:
        return [
            max(
                cur.high - cur.low,
                abs(cur.high - prev.close),
                abs(cur.low - prev.close),
            )
            for prev, cur in zip(bars, bars[1:])
        ]

    def _atr_series(self, bars: Sequence[Bar]) -> List[Decimal]:
        """Wilder ATR: SMA seed over the first period, then recursive smooth."""
        p: int = self._atr_period
        trs: List[Decimal] = self._true_ranges(bars)
        dp: Decimal = Decimal(p)
        series: List[Decimal] = [sum(trs[:p], _ZERO) / dp]
        pm1: Decimal = dp - 1
        for tr in trs[p:]:
            series.append((series[-1] * pm1 + tr) / dp)
        return series

    @staticmethod
    def _sma(values: Sequence[Decimal], period: int) -> Decimal:
        window: Sequence[Decimal] = values[-period:]
        return sum(window, _ZERO) / Decimal(len(window))

    @staticmethod
    def _dx(sm_plus: Decimal, sm_minus: Decimal, sm_tr: Decimal) -> Decimal:
        if sm_tr == _ZERO:
            return _ZERO
        plus_di: Decimal = _HUNDRED * sm_plus / sm_tr
        minus_di: Decimal = _HUNDRED * sm_minus / sm_tr
        total: Decimal = plus_di + minus_di
        if total == _ZERO:
            return _ZERO
        return _HUNDRED * abs(plus_di - minus_di) / total

    def _adx(self, bars: Sequence[Bar]) -> Tuple[Decimal, Decimal, Decimal]:
        """Wilder ADX: smoothed DM/TR -> DX series -> double-smoothed ADX.

        Returns ``(adx, plus_di, minus_di)`` — the directional index lines
        are byproducts of the same smoothing pass and feed the confluence
        vote, so one computation serves both trend strength and direction.
        """
        p: int = self._adx_period
        dp: Decimal = Decimal(p)

        trs: List[Decimal] = []
        plus_dm: List[Decimal] = []
        minus_dm: List[Decimal] = []
        for prev, cur in zip(bars, bars[1:]):
            up: Decimal = cur.high - prev.high
            down: Decimal = prev.low - cur.low
            plus_dm.append(up if (up > down and up > _ZERO) else _ZERO)
            minus_dm.append(down if (down > up and down > _ZERO) else _ZERO)
            trs.append(
                max(
                    cur.high - cur.low,
                    abs(cur.high - prev.close),
                    abs(cur.low - prev.close),
                )
            )

        sm_tr: Decimal = sum(trs[:p], _ZERO)
        sm_plus: Decimal = sum(plus_dm[:p], _ZERO)
        sm_minus: Decimal = sum(minus_dm[:p], _ZERO)
        dxs: List[Decimal] = [self._dx(sm_plus, sm_minus, sm_tr)]

        for i in range(p, len(trs)):
            sm_tr = sm_tr - (sm_tr / dp) + trs[i]
            sm_plus = sm_plus - (sm_plus / dp) + plus_dm[i]
            sm_minus = sm_minus - (sm_minus / dp) + minus_dm[i]
            dxs.append(self._dx(sm_plus, sm_minus, sm_tr))

        seed_len: int = min(p, len(dxs))
        adx: Decimal = sum(dxs[:seed_len], _ZERO) / Decimal(seed_len)
        pm1: Decimal = dp - 1
        for dx in dxs[seed_len:]:
            adx = (adx * pm1 + dx) / dp

        if sm_tr == _ZERO:
            return adx, _ZERO, _ZERO
        return adx, _HUNDRED * sm_plus / sm_tr, _HUNDRED * sm_minus / sm_tr

    def _rsi(self, bars: Sequence[Bar]) -> Decimal:
        """Wilder RSI: SMA-seeded average gain/loss, then recursive smooth."""
        p: int = self._rsi_period
        dp: Decimal = Decimal(p)
        gains: List[Decimal] = []
        losses: List[Decimal] = []
        for prev, cur in zip(bars, bars[1:]):
            delta: Decimal = cur.close - prev.close
            gains.append(delta if delta > _ZERO else _ZERO)
            losses.append(-delta if delta < _ZERO else _ZERO)

        avg_gain: Decimal = sum(gains[:p], _ZERO) / dp
        avg_loss: Decimal = sum(losses[:p], _ZERO) / dp
        pm1: Decimal = dp - 1
        for gain, loss in zip(gains[p:], losses[p:]):
            avg_gain = (avg_gain * pm1 + gain) / dp
            avg_loss = (avg_loss * pm1 + loss) / dp

        if avg_loss == _ZERO:
            return _HUNDRED if avg_gain > _ZERO else Decimal("50")
        rs: Decimal = avg_gain / avg_loss
        return _HUNDRED - _HUNDRED / (_ONE + rs)

    def _book_imbalance(self, book: L2OrderBook) -> Optional[Decimal]:
        """Bid share of the top-tier resting depth: 0..1, 0.5 is balanced.

        Above 0.5 the near book leans bid (buy-side pressure); below 0.5 it
        leans ask. ``None`` when the book is empty — never inferred.
        """
        if not book.is_populated:
            return None
        bid_depth: Decimal = sum(
            (self._dec(float(level[1]), VOLUME_QUANTUM)
             for level in book.bids[:IMBALANCE_TIERS]),
            _ZERO,
        )
        ask_depth: Decimal = sum(
            (self._dec(float(level[1]), VOLUME_QUANTUM)
             for level in book.asks[:IMBALANCE_TIERS]),
            _ZERO,
        )
        total: Decimal = bid_depth + ask_depth
        if total <= _ZERO:
            return None
        return bid_depth / total

    # ------------------------------------------------------------------ #
    # L2 freshness                                                        #
    # ------------------------------------------------------------------ #

    def _book_freshness(
        self, last_open_ms: int, book: L2OrderBook
    ) -> Tuple[Optional[int], bool]:
        """Age of the book snapshot relative to the candle-close timestamp.

        An unpopulated book or a missing exchange timestamp is treated as
        phantom liquidity outright — never trusted, never traded against.
        """
        if not book.is_populated or book.timestamp_ms is None:
            return None, False
        candle_close_ms: int = last_open_ms + self._timeframe_ms
        age_ms: int = candle_close_ms - book.timestamp_ms
        return age_ms, age_ms <= self._book_max_age_ms


# --------------------------------------------------------------------------- #
# Embedded test architecture                                                   #
# --------------------------------------------------------------------------- #

_T0_MS: Final[int] = 1_750_000_500_000 - (1_750_000_500_000 % TIMEFRAME_MS)


def _build_frame(
    rows: Sequence[Tuple[float, float, float, float, float]],
) -> pd.DataFrame:
    """Assemble a feed-layout DataFrame from (o, h, l, c, v) tuples."""
    ts_ms: List[int] = [_T0_MS + i * TIMEFRAME_MS for i in range(len(rows))]
    frame: pd.DataFrame = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(ts_ms, unit="ms", utc=True),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [r[4] for r in rows],
        }
    )
    frame["amount"] = frame["close"] * frame["volume"]
    float_cols: List[str] = ["open", "high", "low", "close", "volume", "amount"]
    frame[float_cols] = frame[float_cols].astype("float64")
    return frame


def _trending_frame(
    num_bars: int = 80, *, expanding: bool = True, volume_spike: bool = True
) -> pd.DataFrame:
    """Pure uptrend (all +DM -> ADX ~ 100) with controllable range profile."""
    rows: List[Tuple[float, float, float, float, float]] = []
    price: float = 100.0
    for i in range(num_bars):
        step: float = 0.5 + 0.05 * i if expanding else 3.0 - 0.03 * i
        o: float = price
        c: float = o + step
        h: float = c + 0.10 * step
        lo: float = o - 0.05 * step
        v: float = 100.0
        rows.append((o, h, lo, c, v))
        price = c
    if volume_spike:
        o, h, lo, c, _ = rows[-1]
        rows[-1] = (o, h, lo, c, 500.0)
    return _build_frame(rows)


def _choppy_frame(num_bars: int = 80) -> pd.DataFrame:
    """Balanced expanding zigzag: up bars print higher highs, down bars print
    symmetric lower lows, so smoothed +DM ~ -DM -> DX ~ 0 -> ADX far below 25,
    while ranges still expand (volatility gate passes) so only trend fails."""
    rows: List[Tuple[float, float, float, float, float]] = []
    center: float = 100.0
    prev_close: float = center
    for i in range(num_bars):
        amp: float = 2.0 + 0.05 * i  # widening swing around the center
        o: float = prev_close
        if i % 2 == 0:
            c: float = center + amp
            h: float = c + 0.10
            lo: float = o - 0.10
        else:
            c = center - amp
            h = o + 0.10
            lo = c - 0.10
        rows.append((o, h, lo, c, 100.0))
        prev_close = c
    o, h, lo, c, _ = rows[-1]
    rows[-1] = (o, h, lo, c, 500.0)
    return _build_frame(rows)


def _book_for(
    frame: pd.DataFrame,
    *,
    age_ms: int = 0,
    populated: bool = True,
    bid_sizes: Tuple[float, ...] = (5.0, 4.0, 3.0),
    ask_sizes: Tuple[float, ...] = (5.0, 4.0, 3.0),
) -> L2OrderBook:
    """L2 snapshot timestamped ``age_ms`` before the last candle's close."""
    last_open_ms: int = int(pd.Timestamp(frame["timestamps"].iloc[-1]).value // 1_000_000)
    close_ms: int = last_open_ms + TIMEFRAME_MS
    levels: Tuple[Tuple[float, float], ...] = (
        tuple((100.0 - 0.1 * i, size) for i, size in enumerate(bid_sizes))
        if populated
        else ()
    )
    ask_levels: Tuple[Tuple[float, float], ...] = (
        tuple((100.1 + 0.1 * i, size) for i, size in enumerate(ask_sizes))
        if populated
        else ()
    )
    return L2OrderBook(
        symbol="BTC/USDT",
        bids=levels,
        asks=ask_levels,
        timestamp_ms=close_ms - age_ms,
    )


class MarketGatekeeperTests(unittest.TestCase):
    """Each barrier must block independently; all together must admit."""

    def setUp(self) -> None:
        self.gatekeeper: MarketGatekeeper = MarketGatekeeper()

    # -- admission ----------------------------------------------------- #

    def test_admits_when_all_barriers_pass(self) -> None:
        frame: pd.DataFrame = _trending_frame()
        report: RegimeReport = self.gatekeeper.evaluate(frame, _book_for(frame))
        self.assertTrue(report.sufficient_data)
        self.assertTrue(report.trend_ok, f"ADX={report.adx}")
        self.assertTrue(report.volatility_ok, f"ATR={report.atr} SMA={report.atr_sma}")
        self.assertTrue(report.volume_ok)
        self.assertTrue(report.book_fresh)
        self.assertTrue(self.gatekeeper.verify_regime(frame, _book_for(frame)))

    # -- single-barrier blocking --------------------------------------- #

    def test_blocks_weak_trend_only(self) -> None:
        frame: pd.DataFrame = _choppy_frame()
        report: RegimeReport = self.gatekeeper.evaluate(frame, _book_for(frame))
        self.assertFalse(report.trend_ok, f"ADX={report.adx} should be <= 25")
        self.assertTrue(report.volatility_ok, "isolation: volatility must pass")
        self.assertTrue(report.volume_ok, "isolation: volume must pass")
        self.assertTrue(report.book_fresh, "isolation: freshness must pass")
        self.assertFalse(self.gatekeeper.verify_regime(frame, _book_for(frame)))

    def test_blocks_contracting_volatility_only(self) -> None:
        frame: pd.DataFrame = _trending_frame(expanding=False)
        report: RegimeReport = self.gatekeeper.evaluate(frame, _book_for(frame))
        self.assertFalse(
            report.volatility_ok, f"ATR={report.atr} SMA={report.atr_sma}"
        )
        self.assertTrue(report.trend_ok, "isolation: trend must pass")
        self.assertTrue(report.volume_ok, "isolation: volume must pass")
        self.assertTrue(report.book_fresh, "isolation: freshness must pass")
        self.assertFalse(self.gatekeeper.verify_regime(frame, _book_for(frame)))

    def test_blocks_volume_dryup_only(self) -> None:
        frame: pd.DataFrame = _trending_frame(volume_spike=False)
        report: RegimeReport = self.gatekeeper.evaluate(frame, _book_for(frame))
        self.assertFalse(report.volume_ok)
        self.assertTrue(report.trend_ok, "isolation: trend must pass")
        self.assertTrue(report.volatility_ok, "isolation: volatility must pass")
        self.assertTrue(report.book_fresh, "isolation: freshness must pass")
        self.assertFalse(self.gatekeeper.verify_regime(frame, _book_for(frame)))

    def test_blocks_stale_book_only(self) -> None:
        frame: pd.DataFrame = _trending_frame()
        stale: L2OrderBook = _book_for(frame, age_ms=1_500)
        report: RegimeReport = self.gatekeeper.evaluate(frame, stale)
        self.assertFalse(report.book_fresh)
        self.assertEqual(report.book_age_ms, 1_500)
        self.assertTrue(report.trend_ok, "isolation: trend must pass")
        self.assertTrue(report.volatility_ok, "isolation: volatility must pass")
        self.assertTrue(report.volume_ok, "isolation: volume must pass")
        self.assertFalse(self.gatekeeper.verify_regime(frame, stale))

    def test_book_age_boundary_inclusive(self) -> None:
        frame: pd.DataFrame = _trending_frame()
        self.assertTrue(
            self.gatekeeper.evaluate(frame, _book_for(frame, age_ms=1_000)).book_fresh
        )
        self.assertFalse(
            self.gatekeeper.evaluate(frame, _book_for(frame, age_ms=1_001)).book_fresh
        )

    def test_blocks_unpopulated_book(self) -> None:
        frame: pd.DataFrame = _trending_frame()
        empty: L2OrderBook = _book_for(frame, populated=False)
        report: RegimeReport = self.gatekeeper.evaluate(frame, empty)
        self.assertFalse(report.book_fresh)
        self.assertIsNone(report.book_age_ms)
        self.assertFalse(self.gatekeeper.verify_regime(frame, empty))

    def test_blocks_insufficient_history(self) -> None:
        frame: pd.DataFrame = _trending_frame(num_bars=20)
        report: RegimeReport = self.gatekeeper.evaluate(frame, _book_for(frame))
        self.assertFalse(report.sufficient_data)
        self.assertFalse(report.passed)
        self.assertFalse(self.gatekeeper.verify_regime(frame, _book_for(frame)))

    # -- domain contract ------------------------------------------------ #

    def test_evaluation_values_are_decimal(self) -> None:
        frame: pd.DataFrame = _trending_frame()
        report: RegimeReport = self.gatekeeper.evaluate(frame, _book_for(frame))
        for value in (
            report.adx,
            report.atr,
            report.atr_sma,
            report.candle_volume,
            report.average_volume,
        ):
            self.assertIsInstance(value, Decimal)

    def test_pure_uptrend_adx_saturates(self) -> None:
        # All directional movement positive -> -DI == 0 -> DX == 100 -> ADX ~ 100.
        frame: pd.DataFrame = _trending_frame()
        report: RegimeReport = self.gatekeeper.evaluate(frame, _book_for(frame))
        assert report.adx is not None
        self.assertGreater(report.adx, Decimal("90"))

    # -- directional confluence ------------------------------------------ #

    def test_uptrend_directional_metrics(self) -> None:
        frame: pd.DataFrame = _trending_frame()
        report: RegimeReport = self.gatekeeper.evaluate(frame, _book_for(frame))
        assert report.plus_di is not None and report.minus_di is not None
        self.assertGreater(report.plus_di, report.minus_di)
        assert report.rsi is not None
        self.assertGreater(report.rsi, Decimal("70"))  # pure uptrend saturates
        self.assertEqual(report.book_imbalance, Decimal("0.5"))  # symmetric book
        for value in (report.plus_di, report.minus_di, report.rsi,
                      report.book_imbalance):
            self.assertIsInstance(value, Decimal)

    def test_confluence_two_of_three_admits_long(self) -> None:
        # Uptrend: DI votes long, RSI is saturated (fails), so the bid-heavy
        # book must supply the second vote — exactly the 2-of-3 contract.
        frame: pd.DataFrame = _trending_frame()
        bid_heavy: L2OrderBook = _book_for(
            frame, bid_sizes=(10.0, 9.0, 8.0), ask_sizes=(2.0, 2.0, 2.0)
        )
        report: RegimeReport = self.gatekeeper.evaluate(frame, bid_heavy)
        confluence: ConfluenceReport = self.gatekeeper.confluence(
            report, long_side=True
        )
        self.assertTrue(confluence.di_vote)
        self.assertFalse(confluence.rsi_vote, f"RSI={confluence.rsi}")
        self.assertTrue(confluence.book_vote)
        self.assertEqual(confluence.votes, 2)
        self.assertTrue(confluence.passed)

    def test_confluence_vetoes_short_against_uptrend(self) -> None:
        # Short proposal in a pure uptrend with a bid-heavy book: only the
        # RSI exhaustion check agrees -> 1 of 3 -> vetoed.
        frame: pd.DataFrame = _trending_frame()
        bid_heavy: L2OrderBook = _book_for(
            frame, bid_sizes=(10.0, 9.0, 8.0), ask_sizes=(2.0, 2.0, 2.0)
        )
        report: RegimeReport = self.gatekeeper.evaluate(frame, bid_heavy)
        confluence: ConfluenceReport = self.gatekeeper.confluence(
            report, long_side=False
        )
        self.assertFalse(confluence.di_vote)
        self.assertTrue(confluence.rsi_vote)  # RSI ~100 > 30: not oversold
        self.assertFalse(confluence.book_vote)
        self.assertEqual(confluence.votes, 1)
        self.assertFalse(confluence.passed)

    def test_confluence_min_votes_is_configurable(self) -> None:
        strict: MarketGatekeeper = MarketGatekeeper(confluence_min_votes=3)
        frame: pd.DataFrame = _trending_frame()
        bid_heavy: L2OrderBook = _book_for(
            frame, bid_sizes=(10.0, 9.0, 8.0), ask_sizes=(2.0, 2.0, 2.0)
        )
        report: RegimeReport = strict.evaluate(frame, bid_heavy)
        # Same 2-of-3 long scenario fails when all three votes are demanded.
        self.assertFalse(strict.confluence(report, long_side=True).passed)

    def test_confluence_missing_inputs_never_confirm(self) -> None:
        frame: pd.DataFrame = _trending_frame()
        empty: L2OrderBook = _book_for(frame, populated=False)
        report: RegimeReport = self.gatekeeper.evaluate(frame, empty)
        self.assertIsNone(report.book_imbalance)
        confluence: ConfluenceReport = self.gatekeeper.confluence(
            report, long_side=True
        )
        self.assertFalse(confluence.book_vote)  # absent evidence = failed vote


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

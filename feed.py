"""feed.py — Phase 1: multi-asset live data ingestion pipeline.

Institutional-grade asynchronous market-data layer for the BTC/USDT + ADA/USDT
5-minute scalping system.

Responsibilities (and *only* these — strict modular pipeline):
  * CCXT Pro websocket stream workers: ``watch_ohlcv`` (candles) and
    ``watch_order_book`` (Level 2 depth) per asset.
  * Isolated per-asset ``collections.deque(maxlen=512)`` ring buffers holding
    confirmed OHLCV rows only.
  * Candle-close validation: a bar is dispatched downstream exclusively when
    the exchange opens a *newer* bar (``barstate.isconfirmed`` equivalent).
    Live, in-progress candle updates never cross the inference boundary.
  * Resilience engine: exponential backoff with jitter on websocket dropouts;
    workers recover in place without leaking tasks or killing the process.
  * Reconciliation hooks: downstream modules (execution.py) register async
    callbacks that are fired after every reconnection so local order/position
    caches can be re-verified against exchange REST state.
  * Anti-bot guard: randomized 150–450 ms connection jitter and desktop
    User-Agent rotation on every (re)connect.

Domain boundary note (per architecture read-back): everything inside this
module — buffers and DataFrames — is deliberately **float-native** for
structural compatibility with pandas and the Kronos PyTorch inference engine.
The ``Decimal`` conversion happens downstream at the gatekeeper/execution
boundary, never here.

Strict typing: annotated for ``mypy --strict`` (ccxt ships without complete
stubs, hence the targeted ``Any`` at the exchange seam; ``pandas-stubs``
recommended in the dev environment). Requires Python >= 3.10.
"""

from __future__ import annotations

import asyncio
import time
import unittest
import logging
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

import pandas as pd

import ccxt.pro as ccxtpro  # type: ignore[import-untyped]
from ccxt.base.errors import (  # type: ignore[import-untyped]
    ExchangeError,
    NetworkError,
)

__all__ = ["MultiAssetFeed", "L2OrderBook", "FeedEvent", "OHLCVRow"]

logger: Final[logging.Logger] = logging.getLogger("bot.feed")

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

SYMBOLS: Final[Tuple[str, ...]] = tuple(
    s.strip()
    for s in os.getenv(
        "SYMBOLS",
        "BTC/USDT,ADA/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,LINK/USDT",
    ).split(",")
    if s.strip()
)
TIMEFRAME: Final[str] = "5m"
TIMEFRAME_MS: Final[int] = 5 * 60 * 1_000
#: Rolling trade-window capacity; ~20k prints comfortably covers one 5m bar
#: on BTC/USDT spot while bounding memory on busy days.
TRADE_WINDOW_MAXLEN: Final[int] = 20_000
LOOKBACK_BARS: Final[int] = 512

ORDER_BOOK_DEPTH: Final[int] = 10  # gatekeeper consumes top 3; margin kept
QUEUE_MAXSIZE: Final[int] = 64

BACKOFF_BASE_S: Final[float] = 1.0
BACKOFF_FACTOR: Final[float] = 2.0
BACKOFF_MAX_S: Final[float] = 60.0
SEED_MAX_ATTEMPTS: Final[int] = 5

# Anti-bot guard: randomized connection jitter window (150 ms – 450 ms).
JITTER_MIN_S: Final[float] = 0.150
JITTER_MAX_S: Final[float] = 0.450

DESKTOP_USER_AGENTS: Final[Tuple[str, ...]] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) "
    "Gecko/20100101 Firefox/139.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.5 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
)

# --------------------------------------------------------------------------- #
# Types                                                                        #
# --------------------------------------------------------------------------- #

#: (open_timestamp_ms, open, high, low, close, volume) — confirmed bars only.
OHLCVRow = Tuple[int, float, float, float, float, float]

#: (price, size) — one resting level of the L2 book.
PriceLevel = Tuple[float, float]

#: Async callback fired with the affected symbol after a stream reconnects.
#: Registered by execution.py to reconcile cached orders against REST truth.
ReconciliationHook = Callable[[str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class L2OrderBook:
    """Immutable snapshot of the top-of-book Level 2 depth."""

    symbol: str
    bids: Tuple[PriceLevel, ...]
    asks: Tuple[PriceLevel, ...]
    timestamp_ms: Optional[int]

    @property
    def is_populated(self) -> bool:
        """True when both sides carry at least one resting level.

        The gatekeeper's liquidity sieve must treat an unpopulated book as
        zero available liquidity (i.e. trade size resolves to zero — safe).
        """
        return bool(self.bids) and bool(self.asks)


#: Event yielded to subscribers on every confirmed 5-minute bar close.
FeedEvent = Tuple[str, pd.DataFrame, L2OrderBook]

#: One raw trade in the rolling window: (timestamp_ms, price, amount, is_buy).
TradeTick = Tuple[int, float, float, bool]


@dataclass(frozen=True, slots=True)
class DailyContext:
    """Macro context from daily candles (float domain, REST-fetched).

    ``macro_trend`` is the canonical bull/bear line: +1 above the daily
    SMA200, −1 below. ``trend_1d`` uses the faster SMA50. ``dist_30d_high``
    is the drawdown from the 30-day high (≤ 0). ``vol_pct`` ranks the
    current 30-day realized volatility against the full fetched history
    (0 = calmest regime on record, 1 = stormiest). None = not enough data.
    """

    trend_1d: Optional[float]
    macro_trend: Optional[float]
    dist_30d_high: Optional[float]
    vol_pct: Optional[float]


@dataclass(frozen=True, slots=True)
class TradeFlowSnapshot:
    """Aggressive trade-flow metrics over the last bar window.

    Float domain by design — this sits at the CCXT wire seam. The
    supervisor converts to Decimal at the journal boundary.

    ``ofi`` is the accumulated level-1 order-flow imbalance (Cont,
    Kukanov & Stoikov 2014) since the previous read: queue growth at the
    best bid and queue depletion at the best ask count as buying
    pressure, and vice versa. ``trade_imbalance`` is the CVD normalized
    to −1..+1 (None when no trades printed in the window).
    """

    window_ms: int
    trade_count: int
    buy_volume: float
    sell_volume: float
    cvd: float
    trade_imbalance: Optional[float]
    micro_vwap: Optional[float]
    ofi: float

#: Transient exceptions the resilience engine recovers from in place.
_RECOVERABLE: Final[Tuple[Type[BaseException], ...]] = (NetworkError, ExchangeError)


# --------------------------------------------------------------------------- #
# Feed                                                                         #
# --------------------------------------------------------------------------- #


class MultiAssetFeed:
    """Multi-asset 5m OHLCV + L2 depth feed with confirmed-bar dispatch.

    Usage::

        async with MultiAssetFeed() as feed:
            async for symbol, df, book in feed.stream():
                ...  # df holds confirmed bars only; book is the latest L2 snapshot
    """

    def __init__(
        self,
        symbols: Sequence[str] = SYMBOLS,
        timeframe: str = TIMEFRAME,
        lookback: int = LOOKBACK_BARS,
        exchange_id: Optional[str] = None,
    ) -> None:
        if not symbols:
            raise ValueError("MultiAssetFeed requires at least one symbol")
        self._symbols: Tuple[str, ...] = tuple(symbols)
        self._timeframe: str = timeframe
        self._timeframe_ms: int = self._parse_timeframe_ms(timeframe)
        self._lookback: int = lookback

        self._exchange: Any = self._build_exchange(exchange_id)
        # Read-only MAINNET client for daily macro candles: the testnet has
        # almost no daily history (~11 candles), so SMA50/SMA200 and the vol
        # percentile can never compute there. Keyless, never sandboxed, never
        # used to place an order — public market data only.
        self._daily_exchange: Any = self._build_public_exchange(exchange_id)

        # Isolated per-asset state — no cross-asset sharing, ever.
        self._buffers: Dict[str, Deque[OHLCVRow]] = {
            s: deque(maxlen=lookback) for s in self._symbols
        }
        self._live_candle: Dict[str, Optional[OHLCVRow]] = {
            s: None for s in self._symbols
        }
        self._books: Dict[str, Optional[Dict[str, Any]]] = {
            s: None for s in self._symbols
        }
        # Trade-flow window (Phase B): raw prints + level-1 OFI accumulator.
        self._trades: Dict[str, Deque[TradeTick]] = {
            s: deque(maxlen=TRADE_WINDOW_MAXLEN) for s in self._symbols
        }
        self._ofi_acc: Dict[str, float] = {s: 0.0 for s in self._symbols}
        self._daily: Dict[str, Optional[DailyContext]] = {
            s: None for s in self._symbols
        }
        self._prev_best: Dict[str, Optional[Tuple[float, float, float, float]]] = {
            s: None for s in self._symbols
        }
        self._reseeding: Dict[str, bool] = {s: False for s in self._symbols}

        self._queue: "asyncio.Queue[Optional[FeedEvent]]" = asyncio.Queue(
            maxsize=QUEUE_MAXSIZE
        )
        self._tasks: List["asyncio.Task[None]"] = []
        self._aux_tasks: Set["asyncio.Task[None]"] = set()
        self._hooks: List[ReconciliationHook] = []
        self._closed: bool = False
        self._started: bool = False

    # ------------------------------------------------------------------ #
    # Construction helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_timeframe_ms(timeframe: str) -> int:
        units: Dict[str, int] = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
        suffix = timeframe[-1]
        if suffix not in units or not timeframe[:-1].isdigit():
            raise ValueError(f"unsupported timeframe: {timeframe!r}")
        return int(timeframe[:-1]) * units[suffix]

    @staticmethod
    def _build_exchange(exchange_id: Optional[str]) -> Any:
        resolved: str = exchange_id or os.getenv("EXCHANGE_ID", "binance")
        if not hasattr(ccxtpro, resolved):
            raise ValueError(f"unknown CCXT Pro exchange id: {resolved!r}")
        klass: Any = getattr(ccxtpro, resolved)

        config: Dict[str, Any] = {
            "enableRateLimit": True,
            "newUpdates": True,
            # Binance rejects signed requests whose timestamp is >1000ms ahead
            # of the venue clock (-1021 InvalidNonce); let CCXT measure and
            # apply the local-vs-server offset instead of trusting the OS clock.
            "options": {"adjustForTimeDifference": True},
        }
        # Public market-data streams need no auth; keys are attached only if
        # the environment provides them. Never hardcoded.
        api_key: Optional[str] = os.getenv("EXCHANGE_API_KEY")
        api_secret: Optional[str] = os.getenv("EXCHANGE_API_SECRET")
        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret

        exchange: Any = klass(config)
        exchange.userAgent = random.choice(DESKTOP_USER_AGENTS)
        return exchange

    @staticmethod
    def _build_public_exchange(exchange_id: Optional[str]) -> Any:
        """A keyless, non-sandbox client used ONLY for public daily candles.

        Market regime is a property of the REAL market, so daily OHLCV is read
        from mainnet public endpoints rather than the data-starved testnet.
        No API keys are attached and sandbox mode is never set, so it cannot
        place orders — it only ever calls ``fetch_ohlcv``.
        """
        resolved: str = exchange_id or os.getenv("EXCHANGE_ID", "binance")
        klass: Any = getattr(ccxtpro, resolved)
        exchange: Any = klass(
            {"enableRateLimit": True, "options": {"adjustForTimeDifference": True}}
        )
        exchange.userAgent = random.choice(DESKTOP_USER_AGENTS)
        return exchange

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Seed history via REST, then launch all stream workers."""
        if self._started:
            raise RuntimeError("MultiAssetFeed.start() called twice")
        self._started = True

        # Symbol guard — drop any configured pair the exchange doesn't list
        # (e.g. a token that trades on mainnet but not on testnet). One bad
        # SYMBOLS entry must never crash the feed. Best-effort: on any lookup
        # failure we keep the configured list untouched so nothing is silently
        # lost.
        try:
            markets: Any = await self._exchange.load_markets()
        except Exception:  # noqa: BLE001 — market discovery must never block boot
            logger.warning(
                "could not load markets to verify symbols — using SYMBOLS as-is",
                exc_info=True,
            )
        else:
            available: Tuple[str, ...] = tuple(
                s for s in self._symbols if s in markets
            )
            dropped: Tuple[str, ...] = tuple(
                s for s in self._symbols if s not in markets
            )
            if dropped:
                logger.warning(
                    "dropping %d unlisted symbol(s): %s",
                    len(dropped),
                    ", ".join(dropped),
                )
            if available:
                self._symbols = available
            else:
                logger.error(
                    "none of the configured symbols are listed on the exchange — "
                    "keeping SYMBOLS as-is so the problem stays visible"
                )

        for symbol in self._symbols:
            await self._seed_history(symbol)
            logger.info(
                "%s: seeded %d confirmed bars (%s)",
                symbol,
                len(self._buffers[symbol]),
                self._timeframe,
            )

        for symbol in self._symbols:
            self._tasks.append(
                asyncio.create_task(self._ohlcv_worker(symbol), name=f"ohlcv:{symbol}")
            )
            self._tasks.append(
                asyncio.create_task(self._book_worker(symbol), name=f"book:{symbol}")
            )
            self._tasks.append(
                asyncio.create_task(
                    self._trades_worker(symbol), name=f"trades:{symbol}"
                )
            )
            self._tasks.append(
                asyncio.create_task(
                    self._daily_worker(symbol), name=f"daily:{symbol}"
                )
            )

    async def stop(self) -> None:
        """Cancel every worker, close the exchange, release stream consumers."""
        self._closed = True
        all_tasks: List["asyncio.Task[None]"] = [*self._tasks, *self._aux_tasks]
        for task in all_tasks:
            task.cancel()
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        self._tasks.clear()
        self._aux_tasks.clear()

        try:
            await self._exchange.close()
        except Exception:  # noqa: BLE001 — shutdown must never raise upward
            logger.exception("exchange close failed during shutdown")

        try:
            await self._daily_exchange.close()
        except Exception:  # noqa: BLE001 — shutdown must never raise upward
            logger.exception("daily exchange close failed during shutdown")

        self._enqueue(None)  # sentinel: unblocks stream() consumers

    async def __aenter__(self) -> "MultiAssetFeed":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------ #
    # Public interface                                                    #
    # ------------------------------------------------------------------ #

    @property
    def exchange(self) -> Any:
        """Shared CCXT Pro instance — handed to ExecutionRouter by main.py.

        Exposed so the whole system runs on one authenticated connection;
        ownership (and ``close()``) stays with the feed lifecycle.
        """
        return self._exchange

    async def stream(self) -> AsyncIterator[FeedEvent]:
        """Yield ``(symbol, dataframe, l2_order_book)`` on each bar close."""
        while True:
            event: Optional[FeedEvent] = await self._queue.get()
            if event is None:
                return
            yield event

    def add_reconciliation_hook(self, hook: ReconciliationHook) -> None:
        """Register an async callback fired (per symbol) after reconnection."""
        self._hooks.append(hook)

    def dataframe(self, symbol: str) -> pd.DataFrame:
        """Materialize the current confirmed-bar buffer for ``symbol``."""
        return self._frame(symbol)

    def order_book(self, symbol: str) -> L2OrderBook:
        """Latest L2 snapshot for ``symbol`` (unpopulated if none received)."""
        return self._book_snapshot(symbol)

    def buffer_length(self, symbol: str) -> int:
        return len(self._buffers[symbol])

    # ------------------------------------------------------------------ #
    # Resilience engine                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff capped at BACKOFF_MAX_S, plus 0–1 s of jitter."""
        capped: float = min(BACKOFF_BASE_S * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX_S)
        return capped + random.uniform(0.0, 1.0)

    async def _connection_guard(self) -> None:
        """Anti-bot guard: rotate desktop UA and apply 150–450 ms jitter."""
        self._exchange.userAgent = random.choice(DESKTOP_USER_AGENTS)
        await asyncio.sleep(random.uniform(JITTER_MIN_S, JITTER_MAX_S))

    def _fire_reconciliation(self, symbol: str) -> None:
        """Dispatch all registered reconciliation hooks for ``symbol``.

        Hooks run as supervised fire-and-forget tasks: a failing hook is
        logged and isolated; it can never take the feed down with it.
        """
        for hook in self._hooks:
            task: "asyncio.Task[None]" = asyncio.create_task(
                self._run_hook(hook, symbol), name=f"reconcile:{symbol}"
            )
            self._aux_tasks.add(task)
            task.add_done_callback(self._aux_tasks.discard)

    @staticmethod
    async def _run_hook(hook: ReconciliationHook, symbol: str) -> None:
        try:
            await hook(symbol)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — hook faults must stay contained
            logger.exception("reconciliation hook failed for %s", symbol)

    # ------------------------------------------------------------------ #
    # Stream workers                                                      #
    # ------------------------------------------------------------------ #

    async def _ohlcv_worker(self, symbol: str) -> None:
        """Supervised watch_ohlcv loop with in-place dropout recovery."""
        attempt: int = 0
        while not self._closed:
            try:
                await self._connection_guard()
                while not self._closed:
                    payload: Any = await self._exchange.watch_ohlcv(
                        symbol, self._timeframe
                    )
                    if attempt:
                        # First successful frame after a dropout: stream is
                        # live again — let downstream re-verify order state.
                        logger.info("%s: ohlcv stream recovered", symbol)
                        attempt = 0
                        self._fire_reconciliation(symbol)
                    self._ingest_candles(symbol, payload)
            except asyncio.CancelledError:
                raise
            except _RECOVERABLE as exc:
                attempt += 1
                delay: float = self._backoff_delay(attempt)
                logger.warning(
                    "%s: ohlcv stream dropout (%s: %s) — retry %d in %.1fs",
                    symbol,
                    type(exc).__name__,
                    exc,
                    attempt,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception:  # noqa: BLE001 — worker must outlive surprises
                attempt += 1
                delay = self._backoff_delay(attempt)
                logger.exception(
                    "%s: unexpected ohlcv worker fault — retry %d in %.1fs",
                    symbol,
                    attempt,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _book_worker(self, symbol: str) -> None:
        """Supervised watch_order_book loop mirroring the ohlcv worker."""
        attempt: int = 0
        while not self._closed:
            try:
                await self._connection_guard()
                while not self._closed:
                    book: Any = await self._exchange.watch_order_book(symbol)
                    if attempt:
                        logger.info("%s: order book stream recovered", symbol)
                        attempt = 0
                        self._fire_reconciliation(symbol)
                    self._books[symbol] = book
                    self._track_ofi(symbol, book)
            except asyncio.CancelledError:
                raise
            except _RECOVERABLE as exc:
                attempt += 1
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "%s: order book dropout (%s: %s) — retry %d in %.1fs",
                    symbol,
                    type(exc).__name__,
                    exc,
                    attempt,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception:  # noqa: BLE001
                attempt += 1
                delay = self._backoff_delay(attempt)
                logger.exception(
                    "%s: unexpected book worker fault — retry %d in %.1fs",
                    symbol,
                    attempt,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _trades_worker(self, symbol: str) -> None:
        """Supervised watch_trades loop feeding the trade-flow window."""
        attempt: int = 0
        while not self._closed:
            try:
                await self._connection_guard()
                while not self._closed:
                    trades: Any = await self._exchange.watch_trades(symbol)
                    if attempt:
                        logger.info("%s: trade stream recovered", symbol)
                        attempt = 0
                    buffer: Deque[TradeTick] = self._trades[symbol]
                    for trade in trades or ():
                        try:
                            buffer.append(
                                (
                                    int(trade.get("timestamp") or 0),
                                    float(trade.get("price") or 0.0),
                                    float(trade.get("amount") or 0.0),
                                    str(trade.get("side") or "") == "buy",
                                )
                            )
                        except (TypeError, ValueError):
                            continue  # one malformed print never kills the loop
            except asyncio.CancelledError:
                raise
            except _RECOVERABLE as exc:
                attempt += 1
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "%s: trade stream dropout (%s: %s) — retry %d in %.1fs",
                    symbol,
                    type(exc).__name__,
                    exc,
                    attempt,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception:  # noqa: BLE001
                attempt += 1
                delay = self._backoff_delay(attempt)
                logger.exception(
                    "%s: unexpected trades worker fault — retry %d in %.1fs",
                    symbol,
                    attempt,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _daily_worker(self, symbol: str) -> None:
        """Fetch ~250 daily candles, refresh every 6h. Fail-safe: errors
        retry in 15 minutes; ``daily_context`` is None until the first
        successful fetch (absent evidence stays absent)."""
        while not self._closed:
            try:
                payload: Any = await self._daily_exchange.fetch_ohlcv(
                    symbol, timeframe="1d", limit=250
                )
                closes: List[float] = [float(row[4]) for row in payload or []]
                self._daily[symbol] = self._daily_metrics(closes)
                logger.info(
                    "%s: daily macro context refreshed (%d candles)",
                    symbol,
                    len(closes),
                )
                await asyncio.sleep(6 * 3600)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — macro context is optional
                logger.warning(
                    "%s: daily context fetch failed — retry in 15 min",
                    symbol,
                    exc_info=True,
                )
                await asyncio.sleep(900)

    @staticmethod
    def _daily_metrics(closes: Sequence[float]) -> DailyContext:
        """Pure macro math over daily closes (unit-tested, no network)."""
        n: int = len(closes)
        last: Optional[float] = closes[-1] if n else None
        trend_1d: Optional[float] = None
        if last is not None and n >= 50:
            sma50: float = sum(closes[-50:]) / 50.0
            trend_1d = 1.0 if last > sma50 else -1.0
        macro_trend: Optional[float] = None
        if last is not None and n >= 200:
            sma200: float = sum(closes[-200:]) / 200.0
            macro_trend = 1.0 if last > sma200 else -1.0
        dist_30d_high: Optional[float] = None
        if last is not None and n >= 5:
            high: float = max(closes[-30:])
            if high > 0.0:
                dist_30d_high = last / high - 1.0
        vol_pct: Optional[float] = None
        if n >= 60:
            returns: List[float] = [
                (closes[i] / closes[i - 1] - 1.0)
                for i in range(1, n)
                if closes[i - 1] > 0.0
            ]

            def _window_vol(end: int) -> float:
                window = returns[end - 30 : end]
                mean: float = sum(window) / 30.0
                return (sum((r - mean) ** 2 for r in window) / 30.0) ** 0.5

            vols: List[float] = [
                _window_vol(end) for end in range(30, len(returns) + 1)
            ]
            current: float = vols[-1]
            vol_pct = sum(1 for v in vols if v <= current) / len(vols)
        return DailyContext(
            trend_1d=trend_1d,
            macro_trend=macro_trend,
            dist_30d_high=dist_30d_high,
            vol_pct=vol_pct,
        )

    def daily_context(self, symbol: str) -> Optional[DailyContext]:
        """Latest macro context, or None before the first daily fetch."""
        return self._daily.get(symbol)

    def _track_ofi(self, symbol: str, book: Any) -> None:
        """Fold one best-bid/ask update into the symbol's OFI accumulator."""
        try:
            bids: Any = book.get("bids") or []
            asks: Any = book.get("asks") or []
            if not bids or not asks:
                return
            current: Tuple[float, float, float, float] = (
                float(bids[0][0]),
                float(bids[0][1]),
                float(asks[0][0]),
                float(asks[0][1]),
            )
        except (TypeError, ValueError, IndexError):
            return
        previous = self._prev_best.get(symbol)
        if previous is not None:
            self._ofi_acc[symbol] = self._ofi_acc.get(symbol, 0.0) + self._ofi_event(
                previous, current
            )
        self._prev_best[symbol] = current

    @staticmethod
    def _ofi_event(
        previous: Tuple[float, float, float, float],
        current: Tuple[float, float, float, float],
    ) -> float:
        """Level-1 OFI contribution of one book update (Cont et al. 2014).

        e_n =  1{Pb_n >= Pb_n-1} * qb_n  -  1{Pb_n <= Pb_n-1} * qb_n-1
             - 1{Pa_n <= Pa_n-1} * qa_n  +  1{Pa_n >= Pa_n-1} * qa_n-1
        Positive = net buying pressure (bid building / ask depleting).
        """
        prev_bid_p, prev_bid_q, prev_ask_p, prev_ask_q = previous
        bid_p, bid_q, ask_p, ask_q = current
        contribution: float = 0.0
        if bid_p >= prev_bid_p:
            contribution += bid_q
        if bid_p <= prev_bid_p:
            contribution -= prev_bid_q
        if ask_p <= prev_ask_p:
            contribution -= ask_q
        if ask_p >= prev_ask_p:
            contribution += prev_ask_q
        return contribution

    @staticmethod
    def _window_metrics(
        trades: Sequence[TradeTick], now_ms: int, window_ms: int
    ) -> Tuple[int, float, float, Optional[float]]:
        """(count, buy_volume, sell_volume, micro_vwap) inside the window."""
        cutoff: int = now_ms - window_ms
        buy = sell = notional = volume = 0.0
        count: int = 0
        for ts, price, amount, is_buy in trades:
            if ts < cutoff or amount <= 0.0:
                continue
            count += 1
            volume += amount
            notional += price * amount
            if is_buy:
                buy += amount
            else:
                sell += amount
        vwap: Optional[float] = (notional / volume) if volume > 0.0 else None
        return count, buy, sell, vwap

    def trade_flow(self, symbol: str) -> TradeFlowSnapshot:
        """Trade-flow metrics over the last bar window.

        READS AND RESETS the OFI accumulator — the supervisor calls this
        exactly once per dispatched bar. A second read in the same bar
        would see OFI = 0.
        """
        now_ms: int = int(time.time() * 1_000)
        count, buy, sell, vwap = self._window_metrics(
            tuple(self._trades[symbol]), now_ms, self._timeframe_ms
        )
        total: float = buy + sell
        ofi: float = self._ofi_acc.get(symbol, 0.0)
        self._ofi_acc[symbol] = 0.0
        return TradeFlowSnapshot(
            window_ms=self._timeframe_ms,
            trade_count=count,
            buy_volume=buy,
            sell_volume=sell,
            cvd=buy - sell,
            trade_imbalance=((buy - sell) / total) if total > 0.0 else None,
            micro_vwap=vwap,
            ofi=ofi,
        )

    # ------------------------------------------------------------------ #
    # Candle ingestion + barstate confirmation                            #
    # ------------------------------------------------------------------ #

    def _ingest_candles(self, symbol: str, payload: Sequence[Sequence[Any]]) -> None:
        """Route raw websocket candles through barstate validation.

        Confirmation rule: a bar is final only once the exchange emits a bar
        with a strictly newer open timestamp. Updates to the current open
        timestamp mutate the held live candle and are never dispatched.
        """
        for raw in payload:
            row: OHLCVRow = (
                int(raw[0]),
                float(raw[1]),
                float(raw[2]),
                float(raw[3]),
                float(raw[4]),
                float(raw[5]),
            )
            live: Optional[OHLCVRow] = self._live_candle[symbol]

            if live is None:
                buf = self._buffers[symbol]
                if not buf or row[0] > buf[-1][0]:
                    self._live_candle[symbol] = row
                continue

            if row[0] == live[0]:
                # In-progress update of the live bar — hold, never dispatch.
                self._live_candle[symbol] = row
            elif row[0] > live[0]:
                # A newer bar opened: the previous live bar is formally closed.
                self._confirm_bar(symbol, live)
                self._live_candle[symbol] = row
            # row[0] < live[0]: stale or out-of-order frame — discard silently.

    def _confirm_bar(self, symbol: str, bar: OHLCVRow) -> None:
        """Append a formally closed bar, guarding continuity, then dispatch."""
        buf: Deque[OHLCVRow] = self._buffers[symbol]

        if buf and bar[0] <= buf[-1][0]:
            logger.debug("%s: duplicate confirmed bar @ %d ignored", symbol, bar[0])
            return

        if buf and (bar[0] - buf[-1][0]) != self._timeframe_ms:
            missing: int = (bar[0] - buf[-1][0]) // self._timeframe_ms - 1
            logger.warning(
                "%s: continuity gap (%d missing bar(s)) — scheduling REST reseed",
                symbol,
                missing,
            )
            # A gapped buffer must never reach the model: skip dispatch and
            # rebuild the full window from REST instead.
            self._schedule_reseed(symbol)
            return

        buf.append(bar)
        self._dispatch(symbol)

    def _dispatch(self, symbol: str) -> None:
        """Publish (symbol, dataframe, l2_book) to the subscriber queue."""
        event: FeedEvent = (symbol, self._frame(symbol), self._book_snapshot(symbol))
        self._enqueue(event)

    def _enqueue(self, event: Optional[FeedEvent]) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Freshest market state wins: evict the oldest queued event.
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(event)
            logger.warning("subscriber queue full — evicted oldest event")

    # ------------------------------------------------------------------ #
    # Seeding / gap healing                                               #
    # ------------------------------------------------------------------ #

    async def _seed_history(self, symbol: str) -> None:
        """Fill the ring buffer from REST; hold the in-progress bar as live.

        Used both at boot and as the gap-healing reseed: the buffer is
        rebuilt atomically (single event-loop thread) from exchange truth.
        """
        raw: Optional[List[List[Any]]] = None
        for attempt in range(1, SEED_MAX_ATTEMPTS + 1):
            try:
                await self._connection_guard()
                raw = await self._exchange.fetch_ohlcv(
                    symbol, self._timeframe, limit=self._lookback + 1
                )
                break
            except _RECOVERABLE as exc:
                if attempt == SEED_MAX_ATTEMPTS:
                    raise
                delay: float = self._backoff_delay(attempt)
                logger.warning(
                    "%s: history seed failed (%s) — retry %d/%d in %.1fs",
                    symbol,
                    type(exc).__name__,
                    attempt,
                    SEED_MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)

        if raw is None or len(raw) < 2:
            raise RuntimeError(f"{symbol}: exchange returned insufficient history")

        rows: List[OHLCVRow] = [
            (
                int(r[0]),
                float(r[1]),
                float(r[2]),
                float(r[3]),
                float(r[4]),
                float(r[5]),
            )
            for r in raw
        ]
        # The trailing REST row is the in-progress bar on virtually every
        # venue: hold it as the live candle, never buffer it. If it was in
        # fact closed, the websocket confirms it the moment a newer bar opens.
        *closed, live = rows
        buf: Deque[OHLCVRow] = self._buffers[symbol]
        buf.clear()
        buf.extend(closed[-self._lookback :])
        self._live_candle[symbol] = live

    def _schedule_reseed(self, symbol: str) -> None:
        if self._reseeding[symbol]:
            return
        self._reseeding[symbol] = True
        task: "asyncio.Task[None]" = asyncio.create_task(
            self._reseed(symbol), name=f"reseed:{symbol}"
        )
        self._aux_tasks.add(task)
        task.add_done_callback(self._aux_tasks.discard)

    async def _reseed(self, symbol: str) -> None:
        try:
            await self._seed_history(symbol)
            logger.info("%s: gap healed — buffer rebuilt from REST", symbol)
            self._dispatch(symbol)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — next gap detection retriggers
            logger.exception("%s: reseed failed; awaiting next confirmed bar", symbol)
        finally:
            self._reseeding[symbol] = False

    # ------------------------------------------------------------------ #
    # Materialization                                                     #
    # ------------------------------------------------------------------ #

    def _frame(self, symbol: str) -> pd.DataFrame:
        """Confirmed bars as a float-native DataFrame in Kronos layout.

        Columns: timestamps (UTC), open, high, low, close, volume, amount.
        ``amount`` (quote turnover) is approximated as close * volume since
        spot OHLCV streams do not carry native turnover.
        """
        rows: List[OHLCVRow] = list(self._buffers[symbol])
        frame: pd.DataFrame = pd.DataFrame(
            {
                "timestamps": pd.to_datetime(
                    [r[0] for r in rows], unit="ms", utc=True
                ),
                "open": [r[1] for r in rows],
                "high": [r[2] for r in rows],
                "low": [r[3] for r in rows],
                "close": [r[4] for r in rows],
                "volume": [r[5] for r in rows],
            }
        )
        frame["amount"] = frame["close"] * frame["volume"]
        float_cols: List[str] = ["open", "high", "low", "close", "volume", "amount"]
        frame[float_cols] = frame[float_cols].astype("float64")
        return frame

    def _book_snapshot(self, symbol: str) -> L2OrderBook:
        raw: Optional[Dict[str, Any]] = self._books[symbol]
        if raw is None:
            return L2OrderBook(symbol=symbol, bids=(), asks=(), timestamp_ms=None)

        bids: Tuple[PriceLevel, ...] = tuple(
            (float(level[0]), float(level[1]))
            for level in (raw.get("bids") or [])[:ORDER_BOOK_DEPTH]
        )
        asks: Tuple[PriceLevel, ...] = tuple(
            (float(level[0]), float(level[1]))
            for level in (raw.get("asks") or [])[:ORDER_BOOK_DEPTH]
        )
        ts: Optional[int] = (
            int(raw["timestamp"]) if raw.get("timestamp") is not None else None
        )
        return L2OrderBook(symbol=symbol, bids=bids, asks=asks, timestamp_ms=ts)


# --------------------------------------------------------------------------- #
# Standalone smoke test                                                        #
# --------------------------------------------------------------------------- #


async def _demo() -> None:
    """Run the feed standalone and print each confirmed bar close."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    async with MultiAssetFeed() as feed:
        async for symbol, frame, book in feed.stream():
            last = frame.iloc[-1]
            best_bid: float = book.bids[0][0] if book.is_populated else float("nan")
            best_ask: float = book.asks[0][0] if book.is_populated else float("nan")
            logger.info(
                "%s bar closed @ %s | c=%.6f v=%.4f | bars=%d | bb=%.6f ba=%.6f",
                symbol,
                last["timestamps"],
                float(last["close"]),
                float(last["volume"]),
                len(frame),
                best_bid,
                best_ask,
            )


# --------------------------------------------------------------------------- #
# Embedded tests — pure trade-flow math only (zero network, no event loop)     #
# --------------------------------------------------------------------------- #


class DailyMetricsTests(unittest.TestCase):
    def test_uptrend_reads_bullish(self) -> None:
        closes = [100.0 * (1.002 ** i) for i in range(220)]
        ctx = MultiAssetFeed._daily_metrics(closes)
        self.assertEqual(ctx.trend_1d, 1.0)
        self.assertEqual(ctx.macro_trend, 1.0)
        assert ctx.dist_30d_high is not None
        self.assertAlmostEqual(ctx.dist_30d_high, 0.0, places=6)  # at the high

    def test_bear_phase_reads_bearish(self) -> None:
        closes = [100.0] * 150 + [100.0 * (0.99 ** i) for i in range(1, 71)]
        ctx = MultiAssetFeed._daily_metrics(closes)
        self.assertEqual(ctx.trend_1d, -1.0)
        self.assertEqual(ctx.macro_trend, -1.0)
        assert ctx.dist_30d_high is not None
        self.assertLess(ctx.dist_30d_high, -0.2)  # deep under the 30d high
        assert ctx.vol_pct is not None
        self.assertGreater(ctx.vol_pct, 0.5)  # selloff = elevated vol regime

    def test_short_history_yields_none(self) -> None:
        ctx = MultiAssetFeed._daily_metrics([100.0] * 40)
        self.assertIsNone(ctx.trend_1d)
        self.assertIsNone(ctx.macro_trend)
        self.assertIsNone(ctx.vol_pct)

    def test_empty_is_safe(self) -> None:
        ctx = MultiAssetFeed._daily_metrics([])
        self.assertEqual(
            (ctx.trend_1d, ctx.macro_trend, ctx.dist_30d_high, ctx.vol_pct),
            (None, None, None, None),
        )


class TradeFlowMathTests(unittest.TestCase):
    def test_ofi_event_bid_up_ask_up_is_buying_pressure(self) -> None:
        prev = (100.0, 5.0, 101.0, 7.0)
        # Bid improves and ask lifts: +new bid queue, +old ask queue.
        self.assertEqual(
            MultiAssetFeed._ofi_event(prev, (100.5, 3.0, 101.5, 2.0)), 3.0 + 7.0
        )

    def test_ofi_event_bid_down_ask_down_is_selling_pressure(self) -> None:
        prev = (100.0, 5.0, 101.0, 7.0)
        self.assertEqual(
            MultiAssetFeed._ofi_event(prev, (99.5, 9.0, 100.5, 4.0)), -5.0 - 4.0
        )

    def test_ofi_event_static_prices_nets_queue_changes(self) -> None:
        prev = (100.0, 5.0, 101.0, 7.0)
        # Same prices: bid queue +3 (buying), ask queue −1 (buying).
        self.assertEqual(
            MultiAssetFeed._ofi_event(prev, (100.0, 8.0, 101.0, 6.0)), 3.0 + 1.0
        )

    def test_window_metrics_filters_by_time_and_aggregates(self) -> None:
        trades = (
            (100, 9.0, 9.9, True),     # outside the window — ignored
            (1000, 10.0, 2.0, True),
            (2000, 11.0, 1.0, False),
            (5000, 12.0, 3.0, True),
        )
        count, buy, sell, vwap = MultiAssetFeed._window_metrics(trades, 5000, 4500)
        self.assertEqual(count, 3)
        self.assertEqual(buy, 5.0)
        self.assertEqual(sell, 1.0)
        assert vwap is not None
        self.assertAlmostEqual(vwap, (10.0 * 2 + 11.0 * 1 + 12.0 * 3) / 6.0)

    def test_window_metrics_empty_is_safe(self) -> None:
        self.assertEqual(
            MultiAssetFeed._window_metrics((), 0, 1_000), (0, 0.0, 0.0, None)
        )

    def test_snapshot_math_via_fake_state(self) -> None:
        feed = MultiAssetFeed.__new__(MultiAssetFeed)  # no network, no init
        feed._timeframe_ms = 300_000
        now = int(time.time() * 1_000)
        feed._trades = {"X/Y": deque([(now - 1_000, 100.0, 2.0, True),
                                      (now - 2_000, 99.0, 1.0, False)])}
        feed._ofi_acc = {"X/Y": 42.0}
        snap = feed.trade_flow("X/Y")
        self.assertEqual(snap.trade_count, 2)
        self.assertEqual(snap.cvd, 1.0)
        assert snap.trade_imbalance is not None
        self.assertAlmostEqual(snap.trade_imbalance, 1.0 / 3.0)
        self.assertEqual(snap.ofi, 42.0)
        self.assertEqual(feed._ofi_acc["X/Y"], 0.0)  # read-and-reset
        snap2 = feed.trade_flow("X/Y")
        self.assertEqual(snap2.ofi, 0.0)


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "demo":
        try:
            asyncio.run(_demo())
        except KeyboardInterrupt:
            logger.info("feed demo interrupted — shutting down cleanly")
    else:
        logging.basicConfig(level=logging.INFO)
        unittest.main()

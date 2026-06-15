"""main.py — Phase 5: master orchestration loop and supervisor lifecycle.

Boot sequence (any failure is a loud refusal, never a degraded start):
  1. Load ``.env`` into the process environment (os.getenv remains the only
     configuration read path — nothing is ever hardcoded).
  2. Refuse to boot if ``emergency_lock.lock`` exists: the kill switch fired
     on a previous run and a human has not yet cleared it.
  3. Atomically create the single-instance lockfile ``bot.lock``
     (O_CREAT | O_EXCL — cross-platform); if it already exists, refuse to
     boot so two instances can never double-execute trades.
  4. Verify ``USE_SANDBOX=True`` and force sandbox mode onto the shared
     CCXT Pro instance before any module touches the venue.

Runtime pipeline (per confirmed 5-minute bar close from MultiAssetFeed):
  Step A — 3% rolling daily drawdown circuit breaker (baseline re-snapshotted
           at 00:00 UTC). Breach => cancel/flatten everything through
           ExecutionRouter, write the empty ``emergency_lock.lock``, freeze,
           and terminate with a non-zero exit code. The lockfile then blocks
           every reboot until human intervention removes it.
  Step B — MarketGatekeeper.verify_regime(); a False is a peaceful skip.
  Step C — KronosInferenceEngine.generate_signal() (30-path Monte Carlo).
  Step D — STRAT_LONG / STRAT_SHORT hand off to ExecutionRouter.route_trade()
           for the slippage, liquidity, and volatility-Kelly sieves.

Shutdown: SIGINT/SIGTERM are intercepted (with a Windows-safe fallback),
the feed's websocket connections are closed cleanly, the instance lockfile
is released, and the process logs a controlled exit.

Strict typing: annotated for ``mypy --strict``. Embedded unittest suite runs
the supervisor against injected fakes (``python -m unittest main``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import pandas as pd

from ccxt.base.errors import InvalidNonce  # type: ignore[import-untyped]

from execution import (
    ExecutionResult,
    ExecutionRouter,
    ExecutionStatus,
    PerformanceTracker,
)
from feed import (
    SYMBOLS,
    DailyContext,
    FeedEvent,
    L2OrderBook,
    MultiAssetFeed,
    TradeFlowSnapshot,
)
from gatekeeper import ConfluenceReport, MarketGatekeeper, RegimeReport
from journal import ObservationJournal, OutcomeMonitor, TradeJournal, TradeOutcome
from learner import MetaFilter, features_from_context
from predictor import KronosInferenceEngine, PredictionReport, SignalDirection
from sentiment_shadow_client import SentimentShadowClient
from visualizer import (
    ConfluenceUpdate,
    EquityUpdate,
    ExecutionUpdate,
    InferenceUpdate,
    LedgerLine,
    PerformanceUpdate,
    RegimeUpdate,
    TradingBotVisualizer,
    VisualizerEvent,
)

__all__ = ["TradingSupervisor", "main"]

logger: Final[logging.Logger] = logging.getLogger("bot.main")

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
ENV_FILE: Final[Path] = BASE_DIR / ".env"
INSTANCE_LOCKFILE: Final[Path] = BASE_DIR / "bot.lock"
EMERGENCY_LOCKFILE: Final[Path] = BASE_DIR / "emergency_lock.lock"
JOURNAL_DB: Final[Path] = BASE_DIR / "journal.db"
META_MODEL_FILE: Final[Path] = BASE_DIR / "meta_model.json"
META_MODES: Final[Tuple[str, ...]] = ("off", "shadow", "veto")

DAILY_DRAWDOWN_LIMIT: Final[Decimal] = Decimal("0.03")  # 3.0%

EXIT_OK: Final[int] = 0
EXIT_BOOT_REFUSED: Final[int] = 1
EXIT_KILL_SWITCH: Final[int] = 2

_ZERO: Final[Decimal] = Decimal("0")

# --------------------------------------------------------------------------- #
# Environment + lockfile primitives                                            #
# --------------------------------------------------------------------------- #


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean environment flag (true/false, case-insensitive)."""
    raw: str = os.getenv(name, "").strip().lower()
    if raw == "":
        return default
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    raise RuntimeError(
        f"BOOT REFUSED: {name} must be true or false, found {raw!r}"
    )


def load_env_file(path: Path) -> None:
    """Fold ``KEY=VALUE`` lines from ``.env`` into the environment.

    Existing environment variables always win (``setdefault``): the shell is
    a more deliberate configuration act than a file on disk.
    """
    if not path.exists():
        logger.info(".env not found at %s — relying on process environment", path)
        return
    # utf-8-sig: Windows editors love writing BOMs in front of the first key.
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line: str = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _read_lock_pid(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_is_alive(pid: int) -> bool:
    """True when a process with this PID currently exists.

    Deliberately conservative: a recycled PID reads as alive and refuses
    boot — a false "running" costs a manual file delete, a false "stale"
    could double-execute trades.

    Windows note: ``os.kill(pid, 0)`` is NOT a probe there — CPython maps
    any non-console signal to TerminateProcess, which would kill the other
    instance. A query-only process handle is the safe equivalent.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION: int = 0x1000
        STILL_ACTIVE: int = 259
        kernel32: Any = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle: int = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            exit_code: Any = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True  # cannot prove it is dead — refuse, stay safe
            return bool(exit_code.value == STILL_ACTIVE)
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # POSIX: signal 0 is a pure existence probe
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def acquire_instance_lock(path: Path) -> bool:
    """Atomic single-instance lock: O_CREAT|O_EXCL is exclusive on all OSes.

    Self-healing: if the lockfile exists but its recorded owner PID is no
    longer running (hard kill, closed terminal), the stale lock is reclaimed
    automatically. A lock held by a live PID always refuses the boot.
    """
    for attempt in range(2):
        try:
            fd: int = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt == 1:
                return False  # lost the reclaim race to another booting instance
            owner_pid: Optional[int] = _read_lock_pid(path)
            if owner_pid is not None and _pid_is_alive(owner_pid):
                logger.error(
                    "instance lock at %s is held by live PID %d — refusing "
                    "duplicate boot",
                    path,
                    owner_pid,
                )
                return False
            logger.warning(
                "stale instance lock at %s (owner PID %s no longer running) "
                "— reclaiming",
                path,
                owner_pid,
            )
            try:
                path.unlink()
            except OSError:
                logger.error("could not remove stale lock", exc_info=True)
                return False
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return True
    return False


def release_instance_lock(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.error("failed to release instance lock at %s", path, exc_info=True)


# --------------------------------------------------------------------------- #
# Supervisor                                                                   #
# --------------------------------------------------------------------------- #


class TradingSupervisor:
    """Owns the component lifecycle and the per-bar decision pipeline.

    All collaborators are injectable for testing; production boot uses the
    real Phase 1–4 modules wired onto one shared sandboxed exchange.
    """

    def __init__(
        self,
        *,
        symbols: Sequence[str] = SYMBOLS,
        feed: Optional[MultiAssetFeed] = None,
        gatekeeper: Optional[MarketGatekeeper] = None,
        engine: Optional[KronosInferenceEngine] = None,
        router: Optional[ExecutionRouter] = None,
        journal: Optional[TradeJournal] = None,
        meta_filter: Optional[MetaFilter] = None,
        visualizer: Optional[TradingBotVisualizer] = None,
        headless: bool = False,
        emergency_lockfile: Path = EMERGENCY_LOCKFILE,
        drawdown_limit: Decimal = DAILY_DRAWDOWN_LIMIT,
        install_signal_handlers: bool = True,
    ) -> None:
        self._symbols: Tuple[str, ...] = tuple(symbols)
        self._quote_currency: str = self._symbols[0].split("/")[1].split(":")[0]
        self._start_capital: Decimal = Decimal(
            os.getenv("STRATEGY_START_CAPITAL", "10000")
        )
        self._emergency_lockfile: Path = emergency_lockfile
        self._drawdown_limit: Decimal = Decimal(
            os.getenv("RISK_TOTAL_DRAWDOWN_LIMIT") or str(drawdown_limit)
        )
        self._install_handlers: bool = install_signal_handlers

        self._feed: MultiAssetFeed = feed or MultiAssetFeed(symbols=self._symbols)
        self._exchange: Any = self._feed.exchange
        self._configure_sandbox(self._exchange)

        # CONFLUENCE_MIN_VOTES (0..3, default 2): how many of the three
        # directional confirmations (DI, RSI, book imbalance) must agree
        # with the model before a trade may route. 0 disables the veto.
        # REGIME_MIN_ADX (default 25): trend sieve floor — variant B runs 20.
        self._gatekeeper: MarketGatekeeper = gatekeeper or MarketGatekeeper(
            confluence_min_votes=int(os.getenv("CONFLUENCE_MIN_VOTES", "2")),
            adx_threshold=Decimal(os.getenv("REGIME_MIN_ADX", "25")),
        )
        # EDGE_THRESHOLD / DEAD_BAND_LOW / DEAD_BAND_HIGH: Monte Carlo gate.
        # The engine refuses combinations violating
        # dead_band_low < dead_band_high <= edge_threshold, so a relaxed
        # variant lowering the edge must lower the dead band with it
        # (e.g. EDGE_THRESHOLD=0.50 DEAD_BAND_LOW=0.48 DEAD_BAND_HIGH=0.50).
        self._engine: KronosInferenceEngine = engine or KronosInferenceEngine(
            edge_threshold=Decimal(os.getenv("EDGE_THRESHOLD", "0.53")),
            dead_band_low=Decimal(os.getenv("DEAD_BAND_LOW", "0.48")),
            dead_band_high=Decimal(os.getenv("DEAD_BAND_HIGH", "0.52")),
        )

        # SENTIMENT_SHADOW (default on): pure-observation hook into the
        # standalone sentiment/microstructure engine. Fire-and-forget; the
        # result is never read here and can never delay or alter routing.
        # If the engine is down the call degrades to a logged no-op.
        # SENTIMENT_ENGINE_URL overrides the default http://127.0.0.1:8787.
        self._sentiment_shadow: Optional[SentimentShadowClient] = None
        if os.getenv("SENTIMENT_SHADOW", "on").lower() not in ("off", "0", "false"):
            self._sentiment_shadow = SentimentShadowClient(
                base_url=os.getenv("SENTIMENT_ENGINE_URL", "http://127.0.0.1:8787"),
            )

        # Multi-variant data farm (harvester support): the regime and
        # confluence gates are ALWAYS computed and journaled, but a variant
        # may choose not to enforce them so that vetoed setups stop being
        # censored data. Safety rails (sandbox check, drawdown breaker,
        # insufficient-data skip, execution sieves) are NOT affected.
        self._regime_enforce: bool = _env_flag("REGIME_ENFORCE", True)
        self._confluence_enforce: bool = _env_flag("CONFLUENCE_ENFORCE", True)
        if not self._regime_enforce:
            logger.warning(
                "REGIME_ENFORCE=false — regime verdicts are journaled but "
                "do NOT block inference (data-harvester mode)"
            )
        if not self._confluence_enforce:
            logger.warning(
                "CONFLUENCE_ENFORCE=false — confluence votes are journaled "
                "but do NOT block routing (data-harvester mode)"
            )

        # Phase 7 — trade journal + Kelly feedback. The tracker is re-seeded
        # from realized journal history on every boot, so position sizing
        # remembers the system's actual record across restarts.
        self._journal: TradeJournal = journal or TradeJournal(
            Path(os.getenv("JOURNAL_DB") or JOURNAL_DB)
        )
        self._tracker: PerformanceTracker = PerformanceTracker()
        self._journal.replay_into(self._tracker)
        # Observation journal — every directional setup (traded or not) written
        # to its own db for offline learning (the 10x data). Isolated, optional,
        # fail-safe: if it can't open we log and keep trading.
        self._observations: Optional[ObservationJournal] = None
        try:
            self._observations = ObservationJournal(
                Path(os.getenv("OBSERVATIONS_DB") or "observations.db")
            )
        except Exception:  # noqa: BLE001 — observation store is never load-bearing
            logger.warning(
                "observation journal unavailable — continuing without it",
                exc_info=True,
            )
        self._monitor: OutcomeMonitor = OutcomeMonitor(
            self._exchange,
            self._journal,
            self._tracker,
            quote_currency=self._quote_currency,
        )

        # Phase 9 — meta-label filter. Shadow by default: it scores and is
        # journaled, but only META_FILTER_MODE=veto lets it block capital.
        self._meta_mode: str = (
            os.getenv("META_FILTER_MODE", "shadow").strip().lower()
        )
        if self._meta_mode not in META_MODES:
            raise RuntimeError(
                f"BOOT REFUSED: META_FILTER_MODE must be one of {META_MODES}, "
                f"found {self._meta_mode!r}"
            )
        self._meta_min_pwin: Decimal = Decimal(os.getenv("META_MIN_PWIN", "0.5"))
        self._meta_filter: MetaFilter = meta_filter or MetaFilter(
            Path(os.getenv("META_MODEL") or META_MODEL_FILE)
        )

        # Testnet venues are throttled: default to a 2 s order boundary there
        # (the live matching engine setting is 0.2 s — set ORDER_TIMEOUT_S).
        # TP_ATR_MULT / SL_ATR_MULT: bracket distances (walk-forward 2026-06-11
        # favoured TP 2.5 / SL 2.5 over the original 1.5 / 2.5).
        # SLIPPAGE_LIMIT: max trigger-to-market deviation (0.0005 = 0.05%).
        self._router: ExecutionRouter = router or ExecutionRouter(
            self._exchange,
            tracker=self._tracker,
            order_timeout_s=float(os.getenv("ORDER_TIMEOUT_S", "2.0")),
            take_profit_atr_mult=Decimal(os.getenv("TP_ATR_MULT", "1.5")),
            stop_loss_atr_mult=Decimal(os.getenv("SL_ATR_MULT", "2.5")),
            slippage_limit=Decimal(os.getenv("SLIPPAGE_LIMIT", "0.0005")),
        )
        self._feed.add_reconciliation_hook(self._router.on_reconnect)

        # Mission Control dashboard — pure observer, drop-oldest telemetry.
        self._visualizer: Optional[TradingBotVisualizer] = (
            None
            if headless
            else visualizer
            or TradingBotVisualizer(
                symbols=self._symbols,
                exchange_label=f"{os.getenv('EXCHANGE_ID', 'binance')} sandbox",
            )
        )

        self._stop_event: asyncio.Event = asyncio.Event()
        # Settles may now be triggered from two places (per-bar pipeline and
        # the user-data websocket); a lock guarantees a trade can never be
        # recorded twice into the Kelly tracker.
        self._settle_lock: asyncio.Lock = asyncio.Lock()
        # USER_DATA_WS (default on): authenticated order stream that settles
        # brackets seconds after they fill instead of on the next bar close.
        # Degrades silently when the exchange lacks watch_orders (test fakes)
        # and is pure acceleration — the per-bar poll remains authoritative.
        self._user_data_ws: bool = _env_flag("USER_DATA_WS", True)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._feed_stopped: bool = False
        self._killed: bool = False
        self._exit_code: int = EXIT_OK
        self._daily_baseline_equity: Optional[Decimal] = None
        self._baseline_date: Optional[date] = None

    # ------------------------------------------------------------------ #
    # Boot safeguards                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _configure_sandbox(exchange: Any) -> None:
        """Refuse to exist outside the sandbox unless explicitly configured."""
        use_sandbox: str = os.getenv("USE_SANDBOX", "").strip().lower()
        if use_sandbox != "true":
            raise RuntimeError(
                "BOOT REFUSED: USE_SANDBOX=True is mandatory in the environment "
                f"(found {use_sandbox!r}) — live trading is not enabled"
            )
        set_mode: Any = getattr(exchange, "set_sandbox_mode", None)
        if callable(set_mode):
            set_mode(True)
        if not bool(
            getattr(exchange, "isSandboxModeEnabled", False)
            or getattr(exchange, "sandboxMode", False)
        ):
            raise RuntimeError(
                "BOOT REFUSED: sandbox mode could not be verified on the "
                "shared exchange instance"
            )
        logger.warning(
            "WARNING: Running in TESTNET/PAPER TRADING MODE — every order "
            "routes to the sandbox venue, no real capital is at risk"
        )

    # ------------------------------------------------------------------ #
    # Signal interception                                                 #
    # ------------------------------------------------------------------ #

    def _install_signal_interceptors(self) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._loop = loop
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_shutdown, sig.name)
            except (NotImplementedError, RuntimeError):
                # Windows / non-main-thread fallback: classic handler that
                # marshals back onto the loop thread-safely.
                signal.signal(sig, self._sync_signal_handler)

    def _request_shutdown(self, source: str) -> None:
        logger.info("graceful shutdown requested (%s)", source)
        self._stop_event.set()

    def _sync_signal_handler(self, signum: int, frame: Any) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._request_shutdown, signal.Signals(signum).name
            )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def _user_data_worker(self) -> None:
        """Authenticated order-stream listener: instant bracket settles.

        Pure acceleration with fail-safe degradation: any fault backs off
        and retries; a dead stream simply returns the system to per-bar
        polling, which remains authoritative either way.
        """
        watch_orders: Any = getattr(self._exchange, "watch_orders", None)
        if not callable(watch_orders):
            logger.info("user-data stream unavailable — per-bar polling only")
            return
        attempt: int = 0
        terminal: Tuple[str, ...] = ("closed", "canceled", "cancelled", "expired")
        while not self._stop_event.is_set():
            try:
                orders: Any = await watch_orders()
                attempt = 0
                touched: set[str] = set()
                for order in orders or ():
                    symbol: str = str(order.get("symbol") or "")
                    if (
                        symbol in self._symbols
                        and str(order.get("status") or "") in terminal
                    ):
                        touched.add(symbol)
                for symbol in touched:
                    logger.info(
                        "%s: user-data stream reports a terminal order — "
                        "settling immediately",
                        symbol,
                    )
                    await self._settle_outcomes(symbol)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — acceleration must never kill
                attempt += 1
                delay: float = min(60.0, 2.0 * attempt)
                logger.warning(
                    "user-data stream fault — retry %d in %.0fs (polling "
                    "remains active)",
                    attempt,
                    delay,
                    exc_info=attempt <= 2,
                )
                await asyncio.sleep(delay)

    async def run(self) -> int:
        """Boot the pipeline, consume bar closes, and exit deliberately."""
        if self._install_handlers:
            self._install_signal_interceptors()
        else:
            self._loop = asyncio.get_running_loop()

        logger.info(
            "supervisor boot: symbols=%s drawdown_limit=%s%%",
            ", ".join(self._symbols),
            self._drawdown_limit * Decimal("100"),
        )

        await self._feed.start()
        await self._snapshot_baseline()
        self._publish_performance()
        shutdown_watcher: "asyncio.Task[None]" = asyncio.create_task(
            self._shutdown_watcher(), name="shutdown-watcher"
        )
        visualizer_task: Optional["asyncio.Task[None]"] = None
        if self._visualizer is not None:
            visualizer_task = asyncio.create_task(
                self._visualizer.run(), name="visualizer"
            )
        user_data_task: Optional["asyncio.Task[None]"] = None
        if self._user_data_ws:
            user_data_task = asyncio.create_task(
                self._user_data_worker(), name="user-data"
            )

        try:
            async for symbol, frame, book in self._feed.stream():
                if self._stop_event.is_set():
                    break
                await self._handle_event(symbol, frame, book)
                if self._killed:
                    break
        finally:
            shutdown_watcher.cancel()
            await asyncio.gather(shutdown_watcher, return_exceptions=True)
            if user_data_task is not None:
                user_data_task.cancel()
                await asyncio.gather(user_data_task, return_exceptions=True)
            await self._stop_feed_once()
            if self._visualizer is not None:
                self._visualizer.stop()
            if visualizer_task is not None:
                await asyncio.gather(visualizer_task, return_exceptions=True)
            self._journal.close()
            if self._observations is not None:
                self._observations.close()
            logger.info("supervisor loop closed (exit code %d)", self._exit_code)

        return self._exit_code

    async def _shutdown_watcher(self) -> None:
        """Releases the stream the moment a termination signal arrives."""
        await self._stop_event.wait()
        await self._stop_feed_once()

    async def _stop_feed_once(self) -> None:
        if self._feed_stopped:
            return
        self._feed_stopped = True
        await self._feed.stop()
        logger.info("websocket connections closed cleanly")

    # ------------------------------------------------------------------ #
    # Telemetry                                                           #
    # ------------------------------------------------------------------ #

    def _publish(self, event: VisualizerEvent) -> None:
        """Non-blocking telemetry tap — a no-op in headless mode."""
        if self._visualizer is not None:
            self._visualizer.publish(event)

    def _publish_performance(self) -> None:
        snapshot = self._journal.performance()
        self._publish(
            PerformanceUpdate(
                wins=snapshot.wins,
                losses=snapshot.losses,
                scratches=snapshot.scratches,
                open_trades=snapshot.open_trades,
                realized_pnl=snapshot.realized_pnl,
                win_rate=snapshot.win_rate,
                kelly_fraction=self._tracker.kelly_allocation(),
            )
        )

    async def _settle_outcomes(self, symbol: str) -> None:
        """Phase 7 learning loop: realized results update Kelly + dashboard.

        Serialized: the bar pipeline and the user-data websocket may both
        request a settle; OutcomeMonitor.poll must never run twice
        concurrently for overlapping trades (double Kelly counting).
        """
        async with self._settle_lock:
            outcomes: List[TradeOutcome] = await self._monitor.poll(symbol)
            for outcome in outcomes:
                logger.info(
                    "%s: trade #%d settled %s — pnl %s %s",
                    symbol,
                    outcome.trade_id,
                    outcome.status,
                    outcome.pnl,
                    self._quote_currency,
                )
                sign: str = "+" if outcome.pnl >= Decimal("0") else ""
                self._publish(
                    LedgerLine(
                        message=(
                            f"{symbol} TRADE #{outcome.trade_id} {outcome.status} — "
                            f"entry {outcome.entry_price} exit {outcome.exit_price} "
                            f"pnl {sign}{outcome.pnl} {self._quote_currency}"
                        ),
                        style="bold green" if outcome.status == "WIN" else "bold red",
                    )
                )
            if outcomes:
                self._publish_performance()
                # The venue is flat for this symbol again; tell the router.
                await self._router.reconcile_exchange_truth(symbol)

    # ------------------------------------------------------------------ #
    # Step A — daily drawdown circuit breaker                             #
    # ------------------------------------------------------------------ #

    async def _fetch_total_equity(self) -> Decimal:
        """Strategy equity: the bot's funded capital plus its own realized and
        unrealized PnL — deliberately NOT the raw wallet value.

        The (testnet) wallet is pre-funded with large balances the bot never
        deployed; counting them would swamp the strategy's real result and make
        the drawdown breaker track the market instead of the bot. This measures
        only what the bot did: start capital, plus realized PnL of closed
        trades, plus each open position marked to the current price.
        """
        realized: Decimal = self._journal.performance().realized_pnl
        unrealized: Decimal = _ZERO
        for trade in self._journal.open_trades():
            try:
                mark: Decimal = await self._fetch_mark_price(trade.symbol)
            except Exception as exc:  # noqa: BLE001 — pricing must never crash the breaker
                logger.warning(
                    "%s: cannot price open trade #%d (%s) — unrealized PnL omitted",
                    trade.symbol,
                    trade.trade_id,
                    exc,
                )
                continue
            move: Decimal = (
                mark - trade.entry_price if trade.is_long else trade.entry_price - mark
            )
            unrealized += move * trade.amount
        return self._start_capital + realized + unrealized

    async def _fetch_mark_price(self, symbol: str) -> Decimal:
        """Current mark price for valuing an open position: bid/ask mid, else
        last trade. Retries once on a clock-drift rejection."""
        try:
            ticker: Dict[str, Any] = await self._exchange.fetch_ticker(symbol)
        except InvalidNonce:
            resync: Any = getattr(self._exchange, "load_time_difference", None)
            if callable(resync):
                await resync()
            ticker = await self._exchange.fetch_ticker(symbol)
        bid: Any = ticker.get("bid")
        ask: Any = ticker.get("ask")
        if bid is not None and ask is not None:
            return (Decimal(str(bid)) + Decimal(str(ask))) / Decimal(2)
        last: Any = ticker.get("last")
        if last is None:
            raise RuntimeError(f"{symbol}: ticker carries no usable price")
        return Decimal(str(last))

    async def _snapshot_baseline(self) -> None:
        equity: Decimal = await self._fetch_total_equity()
        if equity <= _ZERO:
            raise RuntimeError(
                "BOOT REFUSED: baseline wallet equity is non-positive — "
                "the drawdown circuit breaker would be undefined"
            )
        self._daily_baseline_equity = equity
        self._baseline_date = datetime.now(timezone.utc).date()
        logger.info(
            "daily baseline equity snapshot: %s %s (%s UTC)",
            equity,
            self._quote_currency,
            self._baseline_date,
        )
        self._publish(
            EquityUpdate(
                equity=equity,
                baseline=equity,
                drawdown_limit=self._drawdown_limit,
                killed=self._killed,
            )
        )

    async def _drawdown_check(self) -> bool:
        """True when trading may proceed; False after the kill switch fires."""
        if datetime.now(timezone.utc).date() != self._baseline_date:
            # 00:00 UTC rollover — a new trading day gets a fresh baseline.
            await self._snapshot_baseline()

        baseline: Optional[Decimal] = self._daily_baseline_equity
        if baseline is None or baseline <= _ZERO:
            logger.critical("drawdown check without a valid baseline — freezing")
            await self._trigger_kill_switch(_ZERO, Decimal("1"))
            return False

        current: Decimal = await self._fetch_total_equity()
        drawdown: Decimal = (baseline - current) / baseline
        # Equity heartbeat — parsed by the local/web dashboards. Keep the
        # format stable: "EQUITY <current> baseline <baseline> <quote>".
        logger.info(
            "EQUITY %s baseline %s %s", current, baseline, self._quote_currency
        )
        tripped: bool = drawdown >= self._drawdown_limit
        self._publish(
            EquityUpdate(
                equity=current,
                baseline=baseline,
                drawdown_limit=self._drawdown_limit,
                killed=self._killed or tripped,
            )
        )
        if tripped:
            await self._trigger_kill_switch(current, drawdown)
            return False
        return True

    async def _trigger_kill_switch(self, equity: Decimal, drawdown: Decimal) -> None:
        """Flatten-confirm first, lockfile second, then freeze and terminate."""
        self._killed = True
        self._exit_code = EXIT_KILL_SWITCH
        logger.critical(
            "KILL SWITCH: rolling daily drawdown %.4f%% >= %.2f%% "
            "(baseline=%s current=%s %s) — flattening all exposure",
            drawdown * Decimal("100"),
            self._drawdown_limit * Decimal("100"),
            self._daily_baseline_equity,
            equity,
            self._quote_currency,
        )
        try:
            await self._router.emergency_flatten_all(self._symbols)
        finally:
            # Lockfile only after the flatten attempt: a locked-out process
            # with open exposure would be the worse failure mode.
            self._emergency_lockfile.write_text("", encoding="utf-8")
            logger.critical(
                "emergency lockfile written at %s — boot is refused until a "
                "human removes it. Runtime frozen; terminating.",
                self._emergency_lockfile,
            )
            self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Steps B/C/D — per-bar decision pipeline                             #
    # ------------------------------------------------------------------ #

    async def _handle_event(
        self, symbol: str, frame: pd.DataFrame, book: L2OrderBook
    ) -> None:
        """One confirmed bar close through the full decision pipeline.

        Error boundary: a fault while handling one bar is logged and
        contained — the supervisor loop itself must never die to a single
        bad event.
        """
        try:
            # Step A — circuit breaker before every trading decision.
            if not await self._drawdown_check():
                return

            # Step A½ — settle the past before judging the present: detect
            # bracket resolutions, fold realized PnL into the Kelly tracker.
            await self._settle_outcomes(symbol)

            # Step B — regime admission (report API feeds the dashboard).
            trigger_price: Decimal = Decimal(str(float(frame["close"].iloc[-1])))
            regime: RegimeReport = self._gatekeeper.evaluate(frame, book)
            self._publish(
                RegimeUpdate(symbol=symbol, report=regime, close_price=trigger_price)
            )
            if not regime.passed:
                logger.info(
                    "%s: regime rejected — no inference this bar "
                    "(data=%s trend=%s volatility=%s volume=%s book=%s)",
                    symbol,
                    regime.sufficient_data,
                    regime.trend_ok,
                    regime.volatility_ok,
                    regime.volume_ok,
                    regime.book_fresh,
                )
                # An unwarmed indicator window is never bypassable — the
                # indicators are None and every downstream stage would be
                # reasoning about nothing.
                if not regime.sufficient_data:
                    return
                if self._regime_enforce:
                    return
                logger.info(
                    "%s: REGIME_ENFORCE=false — proceeding past failed "
                    "regime for the journal's sake",
                    symbol,
                )

            # Step C — Monte Carlo edge sieve, with the predictor's safe-state
            # boundary replicated here so the full report reaches the panel.
            direction: SignalDirection
            try:
                prediction: PredictionReport = await self._engine.evaluate(
                    symbol, frame
                )
                self._publish(InferenceUpdate(symbol=symbol, report=prediction))
                direction = prediction.signal
                logger.info(
                    "%s: %s | paths up/down/flat=%d/%d/%d p_up=%.4f p_down=%.4f",
                    symbol,
                    prediction.signal.value,
                    prediction.paths_up,
                    prediction.paths_down,
                    prediction.paths_flat,
                    prediction.p_up,
                    prediction.p_down,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — safe-state boundary, by mandate
                logger.critical(
                    "%s: inference failure — safe-state %s emitted",
                    symbol,
                    SignalDirection.NEUTRAL.value,
                    exc_info=True,
                )
                direction = SignalDirection.NEUTRAL
            if direction is SignalDirection.NEUTRAL:
                logger.info("%s: %s — standing down", symbol, direction.value)
                return

            # Step B¾ — trade-flow snapshot (Phase B). Read-and-reset, so
            # exactly one consult per bar; fail-safe getattr keeps test
            # fakes and degraded feeds harmless.
            flow_fn: Any = getattr(self._feed, "trade_flow", None)
            flow: Any = flow_fn(symbol) if callable(flow_fn) else None
            flow_imbalance: Optional[Decimal] = None
            ofi_rel: Optional[Decimal] = None
            mvwap_gap_bps: Optional[Decimal] = None
            if flow is not None:
                if flow.trade_imbalance is not None:
                    flow_imbalance = Decimal(str(round(flow.trade_imbalance, 6)))
                if (
                    regime.candle_volume is not None
                    and regime.candle_volume > _ZERO
                ):
                    ofi_rel = Decimal(
                        str(round(flow.ofi / float(regime.candle_volume), 6))
                    )
                if flow.micro_vwap is not None and flow.micro_vwap > 0.0:
                    mvwap_gap_bps = Decimal(
                        str(
                            round(
                                (float(trigger_price) - flow.micro_vwap)
                                / flow.micro_vwap
                                * 10_000,
                                4,
                            )
                        )
                    )

            # Step B⅞ — daily macro context (Phase B½): regime label per
            # trade; near-constant day to day, decisive across months.
            daily_fn: Any = getattr(self._feed, "daily_context", None)
            daily: Any = daily_fn(symbol) if callable(daily_fn) else None

            def _ctx_dec(value: Any) -> Optional[Decimal]:
                return None if value is None else Decimal(str(round(value, 6)))

            trend_1d = _ctx_dec(getattr(daily, "trend_1d", None))
            macro_trend = _ctx_dec(getattr(daily, "macro_trend", None))
            dist_30d_high = _ctx_dec(getattr(daily, "dist_30d_high", None))
            vol_pct_1d = _ctx_dec(getattr(daily, "vol_pct", None))

            # Step C½ — directional confluence: the model proposes, three
            # independent confirmations (DI direction, RSI exhaustion, L2
            # book imbalance) vote on the side before capital moves.
            confluence: ConfluenceReport = self._gatekeeper.confluence(
                regime, long_side=direction is SignalDirection.LONG
            )
            self._publish(ConfluenceUpdate(symbol=symbol, report=confluence))
            if not confluence.passed:
                logger.info(
                    "%s: confluence veto on %s — %d/%d votes "
                    "(DI=%s RSI=%s BOOK=%s | +DI=%s -DI=%s RSI=%s bid%%=%s)",
                    symbol,
                    direction.value,
                    confluence.votes,
                    confluence.required,
                    confluence.di_vote,
                    confluence.rsi_vote,
                    confluence.book_vote,
                    confluence.plus_di,
                    confluence.minus_di,
                    confluence.rsi,
                    confluence.book_imbalance,
                )
                if self._confluence_enforce:
                    return
                logger.info(
                    "%s: CONFLUENCE_ENFORCE=false — proceeding past failed "
                    "confluence for the journal's sake",
                    symbol,
                )

            # Step C¾ — meta-label filter (Phase 9). Shadow mode observes
            # and journals its opinion; veto mode may refuse the trade.
            meta_p: Optional[Decimal] = None
            if self._meta_mode != "off" and self._meta_filter.ready:
                raw_score: Optional[float] = self._meta_filter.score(
                    features_from_context(
                        long_side=direction is SignalDirection.LONG,
                        p_up=prediction.p_up,
                        p_down=prediction.p_down,
                        adx=regime.adx,
                        rsi=regime.rsi,
                        plus_di=regime.plus_di,
                        minus_di=regime.minus_di,
                        book_imbalance=regime.book_imbalance,
                        atr=regime.atr,
                        atr_sma=regime.atr_sma,
                        spread_bps=regime.spread_bps,
                        relative_volume=regime.relative_volume,
                        depth_imbalance=regime.depth_imbalance,
                        trade_imbalance=flow_imbalance,
                        ofi_rel=ofi_rel,
                        mvwap_gap_bps=mvwap_gap_bps,
                        microprice_gap_bps=regime.microprice_gap_bps,
                        trend_1h=regime.trend_1h,
                        trend_4h=regime.trend_4h,
                        rsi_1h=regime.rsi_1h,
                        day_range_pos=regime.day_range_pos,
                        trend_1d=trend_1d,
                        macro_trend=macro_trend,
                        dist_30d_high=dist_30d_high,
                        vol_pct_1d=vol_pct_1d,
                    )
                )
                if raw_score is not None:
                    meta_p = Decimal(str(round(raw_score, 6)))
                    if meta_p < self._meta_min_pwin:
                        if self._meta_mode == "veto":
                            logger.info(
                                "%s: META VETO %s — p_win %s < %s",
                                symbol,
                                direction.value,
                                meta_p,
                                self._meta_min_pwin,
                            )
                            self._publish(
                                LedgerLine(
                                    message=(
                                        f"{symbol} META VETO {direction.value} — "
                                        f"p_win {meta_p} < {self._meta_min_pwin}"
                                    ),
                                    style="bold yellow",
                                )
                            )
                            return
                        logger.info(
                            "%s: META SHADOW %s — p_win %s < %s (would veto; "
                            "trade proceeds, score journaled)",
                            symbol,
                            direction.value,
                            meta_p,
                            self._meta_min_pwin,
                        )
                        self._publish(
                            LedgerLine(
                                message=(
                                    f"{symbol} META SHADOW — p_win {meta_p}: "
                                    f"would veto {direction.value}"
                                ),
                                style="yellow",
                            )
                        )

            # Step C⅞ — sentiment/microstructure shadow evaluation
            # (Phase: shadow only). Fire-and-forget by design: returns in
            # microseconds, the Future is intentionally dropped, and the
            # trade proceeds exactly as if this block did not exist. The
            # sentiment engine journals its confirm/neutral/veto opinion
            # plus realized price outcomes for offline comparison.
            if self._sentiment_shadow is not None:
                self._sentiment_shadow.evaluate_async(
                    symbol=symbol,
                    direction=(
                        "STRAT_LONG"
                        if direction is SignalDirection.LONG
                        else "STRAT_SHORT"
                    ),
                    bot_confidence=float(
                        prediction.p_up
                        if direction is SignalDirection.LONG
                        else prediction.p_down
                    ),
                    trigger_price=float(trigger_price),
                )

            # Step D — protective execution routing. The live open-trade
            # count feeds the concurrent-position cap (harvester mode);
            # single-position mode ignores it.
            result: ExecutionResult = await self._router.route_trade(
                symbol,
                direction,
                frame,
                book,
                trigger_price,
                open_positions=len(self._journal.open_trades(symbol)),
            )
            self._publish(ExecutionUpdate(result=result))
            logger.info(
                "%s: routing outcome %s — %s",
                symbol,
                result.status.value,
                result.reason,
            )

            # Capture the sentiment engine's current alt-data signals for this
            # setup so the search can test them as ingredients (news sentiment,
            # Fear & Greed, crowd positioning, funding...). Off the event loop
            # and fail-safe: blanks on any failure, never blocks the trading bar.
            sentiment: Dict[str, Any] = {}
            if self._sentiment_shadow is not None:
                try:
                    sentiment = await asyncio.get_running_loop().run_in_executor(
                        None, self._sentiment_shadow.signals, symbol
                    )
                except Exception:  # noqa: BLE001 — sentiment capture is never load-bearing
                    sentiment = {}

            # Observation journal — record this directional setup for offline
            # learning whether or not it routed to a real trade. The blocked-by-
            # cap and vetoed bars are the 10x data the learner is starved for.
            # Fail-safe: bookkeeping must never disturb a trading bar.
            if self._observations is not None:
                try:
                    self._observations.record(
                        symbol=symbol,
                        direction=direction.value,
                        entry_price=trigger_price,
                        adx=regime.adx,
                        atr=regime.atr,
                        atr_sma=regime.atr_sma,
                        rsi=regime.rsi,
                        plus_di=regime.plus_di,
                        minus_di=regime.minus_di,
                        book_imbalance=regime.book_imbalance,
                        p_up=prediction.p_up,
                        p_down=prediction.p_down,
                        confluence_votes=confluence.votes,
                        spread_bps=regime.spread_bps,
                        relative_volume=regime.relative_volume,
                        depth_imbalance=regime.depth_imbalance,
                        total_depth=regime.total_depth,
                        trade_imbalance=flow_imbalance,
                        ofi_rel=ofi_rel,
                        mvwap_gap_bps=mvwap_gap_bps,
                        microprice_gap_bps=regime.microprice_gap_bps,
                        trend_1h=regime.trend_1h,
                        trend_4h=regime.trend_4h,
                        rsi_1h=regime.rsi_1h,
                        day_range_pos=regime.day_range_pos,
                        trend_1d=trend_1d,
                        macro_trend=macro_trend,
                        dist_30d_high=dist_30d_high,
                        vol_pct_1d=vol_pct_1d,
                        sent_score=sentiment.get("sentiment_score"),
                        sent_velocity=sentiment.get("sentiment_velocity"),
                        attention_spike=sentiment.get("attention_spike"),
                        fear_greed=sentiment.get("fear_greed"),
                        long_short_ratio=sentiment.get("long_short_ratio"),
                        funding_rate=sentiment.get("funding_rate"),
                        open_interest=sentiment.get("open_interest"),
                        outlook_1h=sentiment.get("outlook_1h"),
                    )
                except Exception:  # noqa: BLE001 — observation logging is never load-bearing
                    logger.debug("%s: observation record skipped", symbol, exc_info=True)

            # Step D½ — journal the bracket with its full decision context.
            if (
                result.status is ExecutionStatus.EXECUTED
                and result.executed_amount is not None
                and result.entry_fill_price is not None
            ):
                self._journal.open_trade(
                    result,
                    adx=regime.adx,
                    atr=regime.atr,
                    atr_sma=regime.atr_sma,
                    rsi=regime.rsi,
                    plus_di=regime.plus_di,
                    minus_di=regime.minus_di,
                    book_imbalance=regime.book_imbalance,
                    p_up=prediction.p_up,
                    p_down=prediction.p_down,
                    confluence_votes=confluence.votes,
                    meta_p_win=meta_p,
                    spread_bps=regime.spread_bps,
                    relative_volume=regime.relative_volume,
                    depth_imbalance=regime.depth_imbalance,
                    total_depth=regime.total_depth,
                    trade_imbalance=flow_imbalance,
                    ofi_rel=ofi_rel,
                    mvwap_gap_bps=mvwap_gap_bps,
                    microprice_gap_bps=regime.microprice_gap_bps,
                    trend_1h=regime.trend_1h,
                    trend_4h=regime.trend_4h,
                    rsi_1h=regime.rsi_1h,
                    day_range_pos=regime.day_range_pos,
                    trend_1d=trend_1d,
                    macro_trend=macro_trend,
                    dist_30d_high=dist_30d_high,
                    vol_pct_1d=vol_pct_1d,
                )
                self._publish_performance()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — supervisor error boundary
            logger.critical(
                "%s: unhandled fault in decision pipeline — bar skipped",
                symbol,
                exc_info=True,
            )


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def _headless_mode() -> bool:
    return os.getenv("HEADLESS", "").strip().lower() == "true"


async def _async_main() -> int:
    supervisor: TradingSupervisor = TradingSupervisor(headless=_headless_mode())
    return await supervisor.run()


def main() -> int:
    load_env_file(ENV_FILE)

    handlers: List[logging.Handler]
    if _headless_mode():
        # Server mode: console for systemd/journald AND bot.log — the
        # dashboard parses equity, decisions and the activity feed from
        # the file, so headless must keep writing it.
        handlers = [
            logging.FileHandler(BASE_DIR / "bot.log", encoding="utf-8"),
            logging.StreamHandler(),
        ]
    else:
        # Dashboard mode: rich.Live owns the console. Full detail goes to
        # bot.log; only CRITICAL (boot refusals, kill switch) hits stderr.
        file_handler = logging.FileHandler(BASE_DIR / "bot.log", encoding="utf-8")
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.CRITICAL)
        handlers = [file_handler, stream_handler]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=handlers,
    )

    if EMERGENCY_LOCKFILE.exists():
        logger.critical(
            "BOOT REFUSED: %s present — the kill switch fired on a previous "
            "run. A human must inspect the account and delete the lockfile "
            "before this system will start again.",
            EMERGENCY_LOCKFILE,
        )
        return EXIT_KILL_SWITCH

    if not acquire_instance_lock(INSTANCE_LOCKFILE):
        logger.critical(
            "BOOT REFUSED: %s is held by a live process (stale locks from "
            "crashed runs are reclaimed automatically — this one is not "
            "stale). Duplicate instances would double-execute trades.",
            INSTANCE_LOCKFILE,
        )
        return EXIT_BOOT_REFUSED

    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.info("controlled exit (keyboard interrupt)")
        return EXIT_OK
    except Exception:  # noqa: BLE001 — boot/runtime faults exit loudly
        logger.critical("fatal supervisor fault", exc_info=True)
        return EXIT_BOOT_REFUSED
    finally:
        release_instance_lock(INSTANCE_LOCKFILE)
        logger.info("instance lock released — controlled exit complete")


# --------------------------------------------------------------------------- #
# Embedded test architecture (injected fakes — no network, no model)           #
# --------------------------------------------------------------------------- #

_TEST_SYMBOL: Final[str] = "BTC/USDT"


def _tiny_frame() -> pd.DataFrame:
    frame: pd.DataFrame = pd.DataFrame(
        {
            "timestamps": pd.to_datetime([1_750_000_500_000], unit="ms", utc=True),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [10.0],
        }
    )
    frame["amount"] = frame["close"] * frame["volume"]
    return frame


def _tiny_book() -> L2OrderBook:
    return L2OrderBook(
        symbol=_TEST_SYMBOL,
        bids=((99.9, 5.0),),
        asks=((100.1, 5.0),),
        timestamp_ms=1_750_000_800_000,
    )


class _FakeBootExchange:
    """Sandbox-capable exchange double with a scripted equity sequence."""

    def __init__(self, equity_sequence: Sequence[float]) -> None:
        self.isSandboxModeEnabled: bool = False
        self._equity: List[float] = list(equity_sequence)
        #: Order states served to the OutcomeMonitor, keyed by order id.
        self.order_states: Dict[str, Dict[str, Any]] = {}
        self.cancelled_orders: List[str] = []

    def set_sandbox_mode(self, enabled: bool) -> None:
        self.isSandboxModeEnabled = enabled

    async def fetch_balance(self) -> Dict[str, Any]:
        value: float = self._equity.pop(0) if len(self._equity) > 1 else self._equity[0]
        return {"USDT": {"total": value}}

    async def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return dict(self.order_states[order_id])

    async def cancel_order(self, order_id: str, symbol: str) -> None:
        self.cancelled_orders.append(order_id)


class _FakeFeed:
    """Feed double yielding a scripted list of bar-close events."""

    def __init__(self, exchange: _FakeBootExchange, events: Sequence[FeedEvent]) -> None:
        self._exchange: _FakeBootExchange = exchange
        self._events: List[FeedEvent] = list(events)
        self.hooks: List[Callable[[str], Awaitable[None]]] = []
        self.started: bool = False
        self.stopped: bool = False

    @property
    def exchange(self) -> Any:
        return self._exchange

    def add_reconciliation_hook(self, hook: Callable[[str], Awaitable[None]]) -> None:
        self.hooks.append(hook)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def stream(self) -> AsyncIterator[FeedEvent]:
        for event in self._events:
            yield event

    def trade_flow(self, symbol: str) -> TradeFlowSnapshot:
        return TradeFlowSnapshot(
            window_ms=300_000,
            trade_count=12,
            buy_volume=7.5,
            sell_volume=2.5,
            cvd=5.0,
            trade_imbalance=0.5,
            micro_vwap=100.0,
            ofi=40.0,
        )

    def daily_context(self, symbol: str) -> DailyContext:
        return DailyContext(
            trend_1d=1.0,
            macro_trend=-1.0,
            dist_30d_high=-0.12,
            vol_pct=0.85,
        )


class _FakeGatekeeper:
    def __init__(self, verdict: bool, confluence_ok: bool = True) -> None:
        self._verdict: bool = verdict
        self._confluence_ok: bool = confluence_ok
        self.calls: int = 0
        self.confluence_calls: int = 0

    def evaluate(self, dataframe: pd.DataFrame, book: L2OrderBook) -> RegimeReport:
        self.calls += 1
        return RegimeReport(
            sufficient_data=True,
            trend_ok=self._verdict,
            volatility_ok=self._verdict,
            volume_ok=self._verdict,
            book_fresh=self._verdict,
            adx=Decimal("30"),
            atr=Decimal("2"),
            atr_sma=Decimal("1.5"),
            candle_volume=Decimal("200"),
            average_volume=Decimal("100"),
            book_age_ms=50,
            plus_di=Decimal("28"),
            minus_di=Decimal("12"),
            rsi=Decimal("55"),
            book_imbalance=Decimal("0.6"),
        )

    def confluence(self, report: RegimeReport, *, long_side: bool) -> ConfluenceReport:
        self.confluence_calls += 1
        ok: bool = self._confluence_ok
        return ConfluenceReport(
            long_side=long_side,
            di_vote=ok,
            rsi_vote=ok,
            book_vote=ok,
            required=2,
            plus_di=report.plus_di,
            minus_di=report.minus_di,
            rsi=report.rsi,
            book_imbalance=report.book_imbalance,
        )


class _FakeEngine:
    def __init__(self, direction: SignalDirection) -> None:
        self._direction: SignalDirection = direction
        self.calls: int = 0

    async def evaluate(
        self, symbol: str, dataframe: pd.DataFrame
    ) -> PredictionReport:
        self.calls += 1
        up: int = 16 if self._direction is SignalDirection.LONG else 14
        down: int = 16 if self._direction is SignalDirection.SHORT else 14
        if self._direction is SignalDirection.NEUTRAL:
            up, down = 15, 15
        return PredictionReport(
            symbol=symbol,
            signal=self._direction,
            sample_count=30,
            paths_up=up,
            paths_down=down,
            paths_flat=30 - up - down,
            p_up=Decimal(up) / Decimal(30),
            p_down=Decimal(down) / Decimal(30),
            anchor_close=Decimal("100.0"),
        )


class _FakeRouter:
    def __init__(self) -> None:
        self.routed: List[Tuple[str, SignalDirection, Decimal]] = []
        self.flattened: List[Tuple[str, ...]] = []
        self.reconciled: List[str] = []
        self.open_positions_seen: List[Optional[int]] = []

    async def on_reconnect(self, symbol: str) -> None:
        return None

    async def emergency_flatten_all(self, symbols: Sequence[str]) -> None:
        self.flattened.append(tuple(symbols))

    async def reconcile_exchange_truth(self, symbol: str) -> bool:
        self.reconciled.append(symbol)
        return True

    async def route_trade(
        self,
        symbol: str,
        direction: SignalDirection,
        dataframe: pd.DataFrame,
        l2_order_book: L2OrderBook,
        trigger_price: Decimal,
        *,
        open_positions: Optional[int] = None,
    ) -> ExecutionResult:
        self.open_positions_seen.append(open_positions)
        self.routed.append((symbol, direction, trigger_price))
        index: int = len(self.routed)
        return ExecutionResult(
            status=ExecutionStatus.EXECUTED,
            symbol=symbol,
            direction=direction,
            reason="fake routed",
            executed_amount=Decimal("0.010"),
            entry_fill_price=Decimal("100.0"),
            take_profit_price=Decimal("101.0"),
            stop_loss_price=Decimal("98.5"),
            take_profit_order_id=f"tp-{index}",
            stop_loss_order_id=f"sl-{index}",
        )


class _FakeMetaFilter:
    """Always-armed meta filter double with a scripted score."""

    def __init__(self, score: float) -> None:
        self._score: float = score
        self.scored: int = 0

    @property
    def ready(self) -> bool:
        return True

    def score(self, features: Sequence[float]) -> float:
        self.scored += 1
        return self._score


def _supervisor(
    *,
    equity_sequence: Sequence[float],
    events: int,
    regime_ok: bool,
    direction: SignalDirection,
    lockfile: Path,
    visualizer: Optional[TradingBotVisualizer] = None,
    confluence_ok: bool = True,
    meta_filter: Optional[_FakeMetaFilter] = None,
) -> Tuple[TradingSupervisor, _FakeFeed, _FakeGatekeeper, _FakeEngine, _FakeRouter]:
    exchange = _FakeBootExchange(equity_sequence)
    event: FeedEvent = (_TEST_SYMBOL, _tiny_frame(), _tiny_book())
    feed = _FakeFeed(exchange, [event] * events)
    gatekeeper = _FakeGatekeeper(regime_ok, confluence_ok=confluence_ok)
    engine = _FakeEngine(direction)
    router = _FakeRouter()
    journal = TradeJournal(lockfile.parent / "journal-test.db")
    try:
        supervisor = TradingSupervisor(
            symbols=(_TEST_SYMBOL,),
            feed=cast(MultiAssetFeed, feed),
            gatekeeper=cast(MarketGatekeeper, gatekeeper),
            engine=cast(KronosInferenceEngine, engine),
            router=cast(ExecutionRouter, router),
            journal=journal,
            meta_filter=cast(MetaFilter, meta_filter) if meta_filter else None,
            visualizer=visualizer,
            headless=visualizer is None,
            emergency_lockfile=lockfile,
            install_signal_handlers=False,
        )
    except BaseException:
        # Constructor refused (e.g. sandbox check): release the SQLite handle
        # so Windows can delete the test's temporary directory.
        journal.close()
        raise
    return supervisor, feed, gatekeeper, engine, router


class TradingSupervisorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        os.environ["USE_SANDBOX"] = "True"
        # Zero network in tests: the sentiment shadow hook would otherwise
        # fire (harmless, fail-safe) HTTP attempts at localhost per signal.
        os.environ["SENTIMENT_SHADOW"] = "off"
        self._tmp = tempfile.TemporaryDirectory()
        self.lockfile: Path = Path(self._tmp.name) / "emergency_lock.lock"

    def tearDown(self) -> None:
        os.environ.pop("SENTIMENT_SHADOW", None)
        self._tmp.cleanup()

    async def test_strategy_equity_is_capital_plus_pnl(self) -> None:
        # Strategy equity ignores the testnet-inflated wallet entirely:
        # start capital + realized PnL + open-position unrealized PnL.
        sup, *_rest = _supervisor(
            equity_sequence=[10_000.0],
            events=0,
            regime_ok=True,
            direction=SignalDirection.LONG,
            lockfile=self.lockfile,
        )
        real_journal = sup._journal
        try:
            sup._start_capital = Decimal("10000")

            class _Perf:
                realized_pnl = Decimal("25")

            class _OpenTrade:
                symbol = _TEST_SYMBOL
                trade_id = 1
                amount = Decimal("0.1")
                entry_price = Decimal("9000")
                is_long = True

            class _Journal:
                def performance(self_inner: Any) -> Any:
                    return _Perf()

                def open_trades(self_inner: Any, symbol: Any = None) -> Any:
                    return [_OpenTrade()]

            class _Ex:
                async def fetch_ticker(self_inner: Any, symbol: str) -> Dict[str, Any]:
                    return {"bid": 9490.0, "ask": 9510.0}

            sup._journal = cast(TradeJournal, _Journal())
            sup._exchange = _Ex()
            # 10000 + 25 realized + (9500-9000)*0.1 = 50 unrealized = 10075
            self.assertEqual(await sup._fetch_total_equity(), Decimal("10075"))

            # Flat: no realized, no open positions -> exactly the start capital.
            class _FlatPerf:
                realized_pnl = Decimal("0")

            class _FlatJournal:
                def performance(self_inner: Any) -> Any:
                    return _FlatPerf()

                def open_trades(self_inner: Any, symbol: Any = None) -> Any:
                    return []

            sup._journal = cast(TradeJournal, _FlatJournal())
            self.assertEqual(await sup._fetch_total_equity(), Decimal("10000"))
        finally:
            real_journal.close()

    async def test_full_pipeline_routes_long_signal(self) -> None:
        supervisor, feed, gatekeeper, engine, router = _supervisor(
            equity_sequence=[10_000.0, 10_000.0],
            events=1,
            regime_ok=True,
            direction=SignalDirection.LONG,
            lockfile=self.lockfile,
        )
        exit_code: int = await supervisor.run()
        self.assertEqual(exit_code, EXIT_OK)
        self.assertEqual(gatekeeper.calls, 1)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(
            router.routed, [(_TEST_SYMBOL, SignalDirection.LONG, Decimal("100.0"))]
        )
        self.assertTrue(feed.started and feed.stopped)
        self.assertFalse(self.lockfile.exists())

    async def test_user_data_stream_settles_immediately(self) -> None:
        from journal import _executed_result

        class _StreamingExchange(_FakeBootExchange):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.stream_calls: int = 0

            async def watch_orders(self) -> List[Dict[str, Any]]:
                self.stream_calls += 1
                if self.stream_calls > 1:
                    raise asyncio.CancelledError  # one batch, then stop
                return [
                    {"symbol": _TEST_SYMBOL, "status": "closed", "id": "tp-1"}
                ]

        exchange = _StreamingExchange([10_000.0, 10_000.0])
        exchange.order_states = {
            "tp-1": {"status": "closed", "average": 101.0},
            "sl-1": {"status": "open"},
        }
        feed = _FakeFeed(exchange, [])
        journal = TradeJournal(self.lockfile.parent / "journal-test.db")
        try:
            supervisor = TradingSupervisor(
                symbols=(_TEST_SYMBOL,),
                feed=cast(MultiAssetFeed, feed),
                gatekeeper=cast(MarketGatekeeper, _FakeGatekeeper(True)),
                engine=cast(KronosInferenceEngine, _FakeEngine(SignalDirection.LONG)),
                router=cast(ExecutionRouter, _FakeRouter()),
                journal=journal,
                headless=True,
                emergency_lockfile=self.lockfile,
                install_signal_handlers=False,
            )
        except BaseException:
            journal.close()
            raise
        journal.open_trade(_executed_result(fill="100", tp="101", sl="98.5"))
        with self.assertRaises(asyncio.CancelledError):
            await supervisor._user_data_worker()
        # The terminal order event triggered an immediate settle: WIN booked
        # without any bar ever closing.
        self.assertEqual(journal.performance().wins, 1)
        self.assertEqual(supervisor._tracker._wins, 1)
        journal.close()

    async def test_user_data_worker_degrades_without_watch_orders(self) -> None:
        supervisor, _, _, _, _ = _supervisor(
            equity_sequence=[10_000.0, 10_000.0],
            events=0,
            regime_ok=True,
            direction=SignalDirection.LONG,
            lockfile=self.lockfile,
        )
        await supervisor._user_data_worker()  # returns quietly, no network

    async def test_flow_and_context_features_are_journaled(self) -> None:
        supervisor, _, _, _, _ = _supervisor(
            equity_sequence=[10_000.0, 10_000.0],
            events=1,
            regime_ok=True,
            direction=SignalDirection.LONG,
            lockfile=self.lockfile,
        )
        self.assertEqual(await supervisor.run(), EXIT_OK)
        reopened = TradeJournal(self.lockfile.parent / "journal-test.db")
        try:
            (trade,) = reopened.open_trades()
        finally:
            reopened.close()
        self.assertEqual(trade.trade_imbalance, Decimal("0.5"))
        self.assertEqual(trade.macro_trend, Decimal("-1.0"))
        self.assertEqual(trade.dist_30d_high, Decimal("-0.12"))
        self.assertEqual(trade.vol_pct_1d, Decimal("0.85"))
        # OFI 40.0 / fake candle volume 200 = 0.2.
        self.assertEqual(trade.ofi_rel, Decimal("0.2"))
        # Trigger 100.0 vs micro-VWAP 100.0 -> 0 bps gap.
        self.assertEqual(trade.mvwap_gap_bps, Decimal("0.0"))

    async def test_executed_trade_is_journaled_and_settled(self) -> None:
        # Bar 1 routes a LONG and journals it; before bar 2's decision the
        # monitor sees the TP leg filled, records the WIN, feeds Kelly, and
        # reconciles the router. The full Phase 7 learning loop, end to end.
        supervisor, feed, _, _, router = _supervisor(
            equity_sequence=[10_000.0] * 4,
            events=2,
            regime_ok=True,
            direction=SignalDirection.LONG,
            lockfile=self.lockfile,
        )
        exchange = cast(_FakeBootExchange, feed.exchange)
        exchange.order_states = {
            "tp-1": {"status": "closed", "average": 101.0},
            "sl-1": {"status": "open"},
            "tp-2": {"status": "open"},
            "sl-2": {"status": "open"},
        }
        self.assertEqual(await supervisor.run(), EXIT_OK)
        # run() closed the journal on shutdown — reopen the file to inspect.
        reopened = TradeJournal(self.lockfile.parent / "journal-test.db")
        try:
            snapshot = reopened.performance()
        finally:
            reopened.close()
        self.assertEqual(snapshot.wins, 1)
        # (101 - 100) * 0.010 = +0.01 USDT realized.
        self.assertEqual(snapshot.realized_pnl, Decimal("0.010"))
        self.assertEqual(snapshot.open_trades, 1)  # bar 2's trade still open
        self.assertIn("sl-1", exchange.cancelled_orders)  # sibling cancelled
        self.assertIn(_TEST_SYMBOL, router.reconciled)  # router saw flat venue
        self.assertEqual(supervisor._tracker._wins, 1)  # Kelly learned the win

    async def test_meta_shadow_mode_never_blocks(self) -> None:
        os.environ["META_FILTER_MODE"] = "shadow"
        try:
            meta = _FakeMetaFilter(score=0.10)  # far below the 0.5 floor
            supervisor, _, _, _, router = _supervisor(
                equity_sequence=[10_000.0, 10_000.0],
                events=1,
                regime_ok=True,
                direction=SignalDirection.LONG,
                lockfile=self.lockfile,
                meta_filter=meta,
            )
            self.assertEqual(await supervisor.run(), EXIT_OK)
            self.assertEqual(meta.scored, 1)  # consulted...
            self.assertEqual(len(router.routed), 1)  # ...but never obeyed
        finally:
            os.environ.pop("META_FILTER_MODE", None)

    async def test_meta_veto_mode_blocks_low_scores(self) -> None:
        os.environ["META_FILTER_MODE"] = "veto"
        try:
            meta = _FakeMetaFilter(score=0.10)
            supervisor, _, _, engine, router = _supervisor(
                equity_sequence=[10_000.0, 10_000.0],
                events=1,
                regime_ok=True,
                direction=SignalDirection.LONG,
                lockfile=self.lockfile,
                meta_filter=meta,
            )
            self.assertEqual(await supervisor.run(), EXIT_OK)
            self.assertEqual(engine.calls, 1)
            self.assertEqual(router.routed, [])  # p_win 0.10 < 0.5: refused
        finally:
            os.environ.pop("META_FILTER_MODE", None)

    async def test_confluence_veto_blocks_routing(self) -> None:
        supervisor, _, gatekeeper, engine, router = _supervisor(
            equity_sequence=[10_000.0, 10_000.0],
            events=1,
            regime_ok=True,
            direction=SignalDirection.LONG,
            lockfile=self.lockfile,
            confluence_ok=False,
        )
        self.assertEqual(await supervisor.run(), EXIT_OK)
        self.assertEqual(engine.calls, 1)  # the model did propose a side
        self.assertEqual(gatekeeper.confluence_calls, 1)
        self.assertEqual(router.routed, [])  # but the vote refused capital

    async def test_regime_rejection_skips_inference(self) -> None:
        supervisor, _, gatekeeper, engine, router = _supervisor(
            equity_sequence=[10_000.0, 10_000.0],
            events=1,
            regime_ok=False,
            direction=SignalDirection.LONG,
            lockfile=self.lockfile,
        )
        self.assertEqual(await supervisor.run(), EXIT_OK)
        self.assertEqual(gatekeeper.calls, 1)
        self.assertEqual(engine.calls, 0)  # inference never reached
        self.assertEqual(router.routed, [])

    async def test_regime_enforce_false_harvests_past_failed_regime(self) -> None:
        os.environ["REGIME_ENFORCE"] = "false"
        try:
            supervisor, _, gatekeeper, engine, router = _supervisor(
                equity_sequence=[10_000.0, 10_000.0],
                events=1,
                regime_ok=False,
                direction=SignalDirection.LONG,
                lockfile=self.lockfile,
            )
            self.assertEqual(await supervisor.run(), EXIT_OK)
            self.assertEqual(gatekeeper.calls, 1)  # gate still computed...
            self.assertEqual(engine.calls, 1)  # ...but inference proceeded
            self.assertEqual(len(router.routed), 1)  # and the trade routed
        finally:
            os.environ.pop("REGIME_ENFORCE", None)

    async def test_confluence_enforce_false_harvests_past_veto(self) -> None:
        os.environ["CONFLUENCE_ENFORCE"] = "false"
        try:
            supervisor, _, gatekeeper, engine, router = _supervisor(
                equity_sequence=[10_000.0, 10_000.0],
                events=1,
                regime_ok=True,
                direction=SignalDirection.LONG,
                lockfile=self.lockfile,
                confluence_ok=False,
            )
            self.assertEqual(await supervisor.run(), EXIT_OK)
            self.assertEqual(gatekeeper.confluence_calls, 1)  # vote journaled
            self.assertEqual(len(router.routed), 1)  # veto not enforced
        finally:
            os.environ.pop("CONFLUENCE_ENFORCE", None)

    async def test_bad_enforce_flag_refuses_boot(self) -> None:
        os.environ["REGIME_ENFORCE"] = "maybe"
        try:
            with self.assertRaises(RuntimeError):
                _supervisor(
                    equity_sequence=[10_000.0, 10_000.0],
                    events=1,
                    regime_ok=True,
                    direction=SignalDirection.LONG,
                    lockfile=self.lockfile,
                )
        finally:
            os.environ.pop("REGIME_ENFORCE", None)

    async def test_neutral_signal_never_routes(self) -> None:
        supervisor, _, _, engine, router = _supervisor(
            equity_sequence=[10_000.0, 10_000.0],
            events=1,
            regime_ok=True,
            direction=SignalDirection.NEUTRAL,
            lockfile=self.lockfile,
        )
        self.assertEqual(await supervisor.run(), EXIT_OK)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(router.routed, [])

    async def test_kill_switch_flattens_locks_and_terminates(self) -> None:
        # Baseline 10_000 -> live equity 9_600 = 4% drawdown >= 3% limit.
        supervisor, feed, gatekeeper, engine, router = _supervisor(
            equity_sequence=[10_000.0, 9_600.0],
            events=3,  # further events must never be evaluated
            regime_ok=True,
            direction=SignalDirection.LONG,
            lockfile=self.lockfile,
        )
        exit_code: int = await supervisor.run()
        self.assertEqual(exit_code, EXIT_KILL_SWITCH)
        self.assertEqual(router.flattened, [(_TEST_SYMBOL,)])
        self.assertTrue(self.lockfile.exists())
        self.assertEqual(self.lockfile.read_text(encoding="utf-8"), "")
        self.assertEqual(gatekeeper.calls, 0)  # frozen before any signal work
        self.assertEqual(engine.calls, 0)
        self.assertEqual(router.routed, [])
        self.assertTrue(feed.stopped)

    async def test_exact_3_percent_drawdown_trips_breaker(self) -> None:
        supervisor, _, _, _, router = _supervisor(
            equity_sequence=[10_000.0, 9_700.0],  # exactly 3.0%
            events=1,
            regime_ok=True,
            direction=SignalDirection.LONG,
            lockfile=self.lockfile,
        )
        self.assertEqual(await supervisor.run(), EXIT_KILL_SWITCH)
        self.assertEqual(router.flattened, [(_TEST_SYMBOL,)])

    async def test_boot_refused_without_sandbox_env(self) -> None:
        os.environ["USE_SANDBOX"] = "False"
        with self.assertRaises(RuntimeError):
            _supervisor(
                equity_sequence=[10_000.0],
                events=0,
                regime_ok=True,
                direction=SignalDirection.NEUTRAL,
                lockfile=self.lockfile,
            )

    async def test_visualizer_receives_pipeline_telemetry(self) -> None:
        import io

        from rich.console import Console

        dashboard = TradingBotVisualizer(
            symbols=(_TEST_SYMBOL,),
            exchange_label="test sandbox",
            console=Console(file=io.StringIO(), width=120),
            refresh_interval_s=0.05,
        )
        supervisor, _, _, _, router = _supervisor(
            equity_sequence=[10_000.0, 10_000.0],
            events=1,
            regime_ok=True,
            direction=SignalDirection.LONG,
            lockfile=self.lockfile,
            visualizer=dashboard,
        )
        self.assertEqual(await supervisor.run(), EXIT_OK)
        self.assertEqual(len(router.routed), 1)
        # The dashboard drained its queue and holds the pipeline's reports.
        self.assertTrue(dashboard.queue.empty())
        self.assertIn(_TEST_SYMBOL, dashboard._regimes)
        self.assertIn(_TEST_SYMBOL, dashboard._inferences)
        self.assertIsNotNone(dashboard._equity)

    def test_instance_lock_is_exclusive(self) -> None:
        lock_path: Path = Path(self._tmp.name) / "bot.lock"
        self.assertTrue(acquire_instance_lock(lock_path))
        # The lock records this live test process — a second boot must refuse.
        self.assertFalse(acquire_instance_lock(lock_path))
        release_instance_lock(lock_path)
        self.assertTrue(acquire_instance_lock(lock_path))
        release_instance_lock(lock_path)

    def test_stale_instance_lock_is_reclaimed(self) -> None:
        import subprocess

        lock_path: Path = Path(self._tmp.name) / "bot.lock"
        # A real PID that is guaranteed dead: a child that already exited.
        child = subprocess.Popen(
            [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL
        )
        child.wait()
        lock_path.write_text(str(child.pid), encoding="utf-8")
        self.assertTrue(acquire_instance_lock(lock_path))  # reclaimed
        self.assertEqual(lock_path.read_text(encoding="utf-8"), str(os.getpid()))
        release_instance_lock(lock_path)

    def test_garbage_instance_lock_is_reclaimed(self) -> None:
        # An empty/corrupt lock can only come from a crash mid-write —
        # reclaim it (loudly) instead of bricking every future boot.
        lock_path: Path = Path(self._tmp.name) / "bot.lock"
        lock_path.write_text("not-a-pid", encoding="utf-8")
        self.assertTrue(acquire_instance_lock(lock_path))
        release_instance_lock(lock_path)

    def test_pid_liveness_probe(self) -> None:
        self.assertTrue(_pid_is_alive(os.getpid()))  # we are running
        self.assertFalse(_pid_is_alive(-1))
        self.assertFalse(_pid_is_alive(0))


if __name__ == "__main__":
    sys.exit(main())

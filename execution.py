"""execution.py — Phase 4: protective position sizing and trade routing.

Final barrier between a strategy signal and exchange capital. Every number
that touches money lives in ``decimal.Decimal``; floats appear only at the
CCXT wire seam, converted in the last possible expression.

Pipeline enforced by ``ExecutionRouter.route_trade`` (in order):
  0. Sandbox verification     — constructor refuses to exist unless the
                                shared CCXT Pro instance is in sandbox mode.
  1. State machine lock       — per-asset IDLE -> PENDING_ENTRY -> ACTIVE;
                                any non-IDLE state blocks multi-entry cold.
  2. Exchange truth check     — ``reconcile_exchange_truth`` queries live
                                positions/open orders; the exchange always
                                overrides the local cache. Any exposure on
                                the venue refuses a new entry.
  3. Slippage sieve           — fresh REST mid price vs the snapshot trigger
                                price; > 0.05% deviation aborts entirely.
  4. Volatility capital sizer — fractional Kelly from tracked system win
                                rates, dampened by ATR_baseline/ATR_current
                                (clamped <= 1: spikes downscale, calm never
                                upscales past the Kelly baseline).
  5. Liquidity sieve          — trade size capped at exactly 5% of the
                                cumulative resting depth in the top 3 tiers
                                of the consumed book side.
  6. Precision quantization   — banker's rounding (ROUND_HALF_EVEN) onto the
                                venue's contract step; sub-minimum aborts.
  7. Bracket routing          — market entry, linked limit TP, linked stop
                                market SL at exactly 2.5x ATR from the fill;
                                each placement bounded by a 200 ms timeout.
                                A bracket failure after a filled entry
                                triggers an emergency reduce-only flatten:
                                this router never leaves a naked position by
                                choice.

ATR inputs reuse the Phase 2 ``MarketGatekeeper`` Decimal indicators (14-period
Wilder ATR as current, its 20-period SMA as baseline) — one implementation,
one set of numbers, no drift between modules.

Strict typing: annotated for ``mypy --strict``. Embedded unittest suite runs
against a fake exchange with zero network access (``python -m unittest
execution``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import unittest
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum, unique
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import pandas as pd

from ccxt.base.errors import NotSupported, OrderNotFound  # type: ignore[import-untyped]

from feed import L2OrderBook
from gatekeeper import MarketGatekeeper, RegimeReport
from predictor import SignalDirection

__all__ = [
    "ExecutionRouter",
    "ExecutionResult",
    "ExecutionStatus",
    "AssetState",
    "PerformanceTracker",
]

logger: Final[logging.Logger] = logging.getLogger("bot.execution")

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

SLIPPAGE_LIMIT: Final[Decimal] = Decimal("0.0005")  # 0.05%
LIQUIDITY_TIERS: Final[int] = 3
LIQUIDITY_CAP_FRACTION: Final[Decimal] = Decimal("0.05")  # 5% of resting depth
STOP_LOSS_ATR_MULT: Final[Decimal] = Decimal("2.5")
TAKE_PROFIT_ATR_MULT: Final[Decimal] = Decimal("1.5")  # strategy target
ORDER_TIMEOUT_S: Final[float] = 0.200
FLATTEN_TIMEOUT_S: Final[float] = 2.0

# Binance spot forbids plain STOP_LOSS market orders on the major pairs; the
# protective leg must be STOP_LOSS_LIMIT. The limit price is pushed this far
# through the stop so the order is marketable the instant it triggers —
# a pseudo stop-market with a bounded worst fill.
SL_LIMIT_BUFFER: Final[Decimal] = Decimal("0.005")  # 0.5% through the stop

KELLY_FRACTION: Final[Decimal] = Decimal("0.5")  # half-Kelly
MAX_CAPITAL_FRACTION: Final[Decimal] = Decimal("0.05")  # hard equity cap

_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")

# --------------------------------------------------------------------------- #
# Types                                                                        #
# --------------------------------------------------------------------------- #


@unique
class AssetState(str, Enum):
    """Per-asset execution state machine."""

    IDLE = "IDLE"
    PENDING_ENTRY = "PENDING_ENTRY"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"


@unique
class ExecutionStatus(str, Enum):
    EXECUTED = "EXECUTED"
    BLOCKED_STATE = "BLOCKED_STATE"
    BLOCKED_EXCHANGE_TRUTH = "BLOCKED_EXCHANGE_TRUTH"
    ABORT_NEUTRAL_SIGNAL = "ABORT_NEUTRAL_SIGNAL"
    ABORT_SLIPPAGE = "ABORT_SLIPPAGE"
    ABORT_LIQUIDITY = "ABORT_LIQUIDITY"
    ABORT_MIN_SIZE = "ABORT_MIN_SIZE"
    ABORT_SPREAD = "ABORT_SPREAD"
    ABORT_ENTRY_TIMEOUT = "ABORT_ENTRY_TIMEOUT"
    ABORT_BRACKET_FAILED = "ABORT_BRACKET_FAILED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Execution log record returned for every routing attempt."""

    status: ExecutionStatus
    symbol: str
    direction: SignalDirection
    reason: str
    trigger_price: Optional[Decimal] = None
    market_price: Optional[Decimal] = None
    deviation: Optional[Decimal] = None
    requested_amount: Optional[Decimal] = None
    executed_amount: Optional[Decimal] = None
    entry_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    entry_fill_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None


# --------------------------------------------------------------------------- #
# Kelly performance tracker                                                    #
# --------------------------------------------------------------------------- #


class PerformanceTracker:
    """Rolling system win-rate / payoff statistics feeding the Kelly sizer.

    Seeded with conservative Bayesian priors so the sizer is defined (and
    modest) before live history exists; recorded results blend in over time.
    """

    def __init__(
        self,
        *,
        prior_wins: int = 11,
        prior_losses: int = 9,
        prior_avg_win: Decimal = Decimal("1.2"),
        prior_avg_loss: Decimal = Decimal("1.0"),
        kelly_fraction: Decimal = KELLY_FRACTION,
        max_fraction: Decimal = MAX_CAPITAL_FRACTION,
    ) -> None:
        if prior_wins < 1 or prior_losses < 1:
            raise ValueError("priors must each contain at least one observation")
        if prior_avg_win <= _ZERO or prior_avg_loss <= _ZERO:
            raise ValueError("prior payoff magnitudes must be positive")
        self._prior_wins: int = prior_wins
        self._prior_losses: int = prior_losses
        self._prior_win_sum: Decimal = prior_avg_win * Decimal(prior_wins)
        self._prior_loss_sum: Decimal = prior_avg_loss * Decimal(prior_losses)
        self._wins: int = 0
        self._losses: int = 0
        self._win_sum: Decimal = _ZERO
        self._loss_sum: Decimal = _ZERO
        self._kelly_fraction: Decimal = kelly_fraction
        self._max_fraction: Decimal = max_fraction

    def record_trade(self, pnl: Decimal) -> None:
        """Fold one realized trade result (quote-currency PnL) into the stats."""
        if pnl > _ZERO:
            self._wins += 1
            self._win_sum += pnl
        elif pnl < _ZERO:
            self._losses += 1
            self._loss_sum += abs(pnl)
        # pnl == 0: scratch trade — no information for the edge estimate.

    @property
    def win_rate(self) -> Decimal:
        wins: Decimal = Decimal(self._prior_wins + self._wins)
        total: Decimal = wins + Decimal(self._prior_losses + self._losses)
        return wins / total

    @property
    def payoff_ratio(self) -> Decimal:
        avg_win: Decimal = (self._prior_win_sum + self._win_sum) / Decimal(
            self._prior_wins + self._wins
        )
        avg_loss: Decimal = (self._prior_loss_sum + self._loss_sum) / Decimal(
            self._prior_losses + self._losses
        )
        if avg_loss == _ZERO:
            return _ONE
        return avg_win / avg_loss

    def kelly_allocation(self) -> Decimal:
        """Fractional Kelly: f* = W - (1-W)/R, scaled and hard-capped."""
        w: Decimal = self.win_rate
        r: Decimal = self.payoff_ratio
        full_kelly: Decimal = w - (_ONE - w) / r
        if full_kelly <= _ZERO:
            return _ZERO
        return min(full_kelly * self._kelly_fraction, self._max_fraction)


# --------------------------------------------------------------------------- #
# Execution router                                                             #
# --------------------------------------------------------------------------- #


class ExecutionRouter:
    """Sandbox-locked bracket-order router with layered safety sieves."""

    def __init__(
        self,
        exchange: Any,
        *,
        gatekeeper: Optional[MarketGatekeeper] = None,
        tracker: Optional[PerformanceTracker] = None,
        require_sandbox: bool = True,
        slippage_limit: Decimal = SLIPPAGE_LIMIT,
        liquidity_tiers: int = LIQUIDITY_TIERS,
        liquidity_cap_fraction: Decimal = LIQUIDITY_CAP_FRACTION,
        stop_loss_atr_mult: Decimal = STOP_LOSS_ATR_MULT,
        take_profit_atr_mult: Decimal = TAKE_PROFIT_ATR_MULT,
        order_timeout_s: float = ORDER_TIMEOUT_S,
        fixed_trade_notional: Optional[Decimal] = None,
        max_open_trades: int = 1,
        max_spread_bps: Optional[Decimal] = None,
    ) -> None:
        self._verify_sandbox(exchange, require_sandbox)
        self._exchange: Any = exchange
        self._gatekeeper: MarketGatekeeper = gatekeeper or MarketGatekeeper()
        self._tracker: PerformanceTracker = tracker or PerformanceTracker()

        # Fixed-notional sizing (data-farm harvester). When set — by argument
        # or FIXED_TRADE_NOTIONAL in the environment — every trade requests
        # exactly this much quote currency instead of Kelly × equity. The
        # liquidity cap, quantization, and venue-minimum sieves still apply;
        # only the capital-allocation arithmetic is replaced.
        self._fixed_notional_is_min: bool = False
        if fixed_trade_notional is None:
            env_notional: str = os.getenv("FIXED_TRADE_NOTIONAL", "").strip()
            if env_notional.lower() == "min":
                # Smallest routable trade: venue minimum amount/notional
                # plus 10% headroom so quantization cannot round below it.
                self._fixed_notional_is_min = True
            elif env_notional:
                try:
                    fixed_trade_notional = Decimal(env_notional)
                except ArithmeticError as exc:
                    raise RuntimeError(
                        "BOOT REFUSED: FIXED_TRADE_NOTIONAL is not a valid "
                        f"decimal: {env_notional!r}"
                    ) from exc
        if fixed_trade_notional is not None and (
            not fixed_trade_notional.is_finite() or fixed_trade_notional <= _ZERO
        ):
            raise RuntimeError(
                "BOOT REFUSED: FIXED_TRADE_NOTIONAL must be a positive finite "
                f"number, found {fixed_trade_notional}"
            )
        self._fixed_notional: Optional[Decimal] = fixed_trade_notional
        if self._fixed_notional is not None:
            logger.warning(
                "FIXED_TRADE_NOTIONAL=%s — Kelly sizing bypassed, every "
                "trade requests a fixed %s quote notional (harvester mode)",
                self._fixed_notional,
                self._fixed_notional,
            )
        elif self._fixed_notional_is_min:
            logger.warning(
                "FIXED_TRADE_NOTIONAL=min — Kelly sizing bypassed, every "
                "trade requests the venue-minimum size (harvester mode)"
            )

        # Concurrent-position mode (data farm). 1 (default) keeps the strict
        # one-bracket-per-symbol state machine and the zero-exposure
        # reconcile gate. N>1 allows that many simultaneous brackets per
        # symbol; 0 = unlimited. The supervisor supplies the live open-trade
        # count from the journal on every routing call.
        if max_open_trades == 1:
            env_max: str = os.getenv("MAX_OPEN_TRADES_PER_SYMBOL", "").strip()
            if env_max:
                try:
                    max_open_trades = int(env_max)
                except ValueError as exc:
                    raise RuntimeError(
                        "BOOT REFUSED: MAX_OPEN_TRADES_PER_SYMBOL is not an "
                        f"integer: {env_max!r}"
                    ) from exc
        if max_open_trades < 0:
            raise RuntimeError(
                "BOOT REFUSED: MAX_OPEN_TRADES_PER_SYMBOL must be >= 0 "
                f"(0 = unlimited), found {max_open_trades}"
            )
        self._max_open_trades: int = max_open_trades

        # Spread sieve (Phase A microstructure). A wide spread is a cost the
        # bracket must overcome before the trade has any edge at all, so it
        # blocks at execution time. Disabled when unset.
        if max_spread_bps is None:
            env_spread: str = os.getenv("MAX_SPREAD_BPS", "").strip()
            if env_spread:
                try:
                    max_spread_bps = Decimal(env_spread)
                except ArithmeticError as exc:
                    raise RuntimeError(
                        "BOOT REFUSED: MAX_SPREAD_BPS is not a valid decimal: "
                        f"{env_spread!r}"
                    ) from exc
        if max_spread_bps is not None and (
            not max_spread_bps.is_finite() or max_spread_bps <= _ZERO
        ):
            raise RuntimeError(
                "BOOT REFUSED: MAX_SPREAD_BPS must be a positive finite "
                f"number of basis points, found {max_spread_bps}"
            )
        self._max_spread_bps: Optional[Decimal] = max_spread_bps
        if self._max_open_trades != 1:
            logger.warning(
                "MAX_OPEN_TRADES_PER_SYMBOL=%s — one-position state machine "
                "bypassed; concurrent brackets allowed (harvester mode)",
                "unlimited" if self._max_open_trades == 0 else self._max_open_trades,
            )
        self._slippage_limit: Decimal = slippage_limit
        self._liquidity_tiers: int = liquidity_tiers
        self._liquidity_cap_fraction: Decimal = liquidity_cap_fraction
        self._stop_loss_atr_mult: Decimal = stop_loss_atr_mult
        self._take_profit_atr_mult: Decimal = take_profit_atr_mult
        self._order_timeout_s: float = order_timeout_s
        self._states: Dict[str, AssetState] = {}

    # ------------------------------------------------------------------ #
    # Sandbox lock                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _verify_sandbox(exchange: Any, require_sandbox: bool) -> None:
        if not require_sandbox:
            logger.warning(
                "ExecutionRouter constructed with require_sandbox=False — "
                "LIVE order routing is possible"
            )
            return
        enabled: bool = bool(
            getattr(exchange, "isSandboxModeEnabled", False)
            or getattr(exchange, "sandboxMode", False)
        )
        if not enabled:
            set_mode: Any = getattr(exchange, "set_sandbox_mode", None)
            if callable(set_mode):
                try:
                    set_mode(True)
                    enabled = True
                except Exception:  # noqa: BLE001 — venue without a testnet
                    enabled = False
        if not enabled:
            raise RuntimeError(
                "refusing to construct ExecutionRouter: sandbox mode could "
                "not be verified/enabled on the shared exchange instance"
            )

    # ------------------------------------------------------------------ #
    # State machine                                                       #
    # ------------------------------------------------------------------ #

    def state(self, symbol: str) -> AssetState:
        return self._states.get(symbol, AssetState.IDLE)

    def _set_state(self, symbol: str, state: AssetState) -> None:
        previous: AssetState = self.state(symbol)
        if previous is not state:
            logger.info("%s: state %s -> %s", symbol, previous.value, state.value)
        self._states[symbol] = state

    # ------------------------------------------------------------------ #
    # Exchange truth                                                      #
    # ------------------------------------------------------------------ #

    async def reconcile_exchange_truth(self, symbol: str) -> bool:
        """True only when the venue shows zero exposure for ``symbol``.

        The exchange is the absolute source of truth: whatever it reports
        overwrites the local state cache. Any query failure (other than a
        venue that simply has no positions endpoint) resolves fail-safe to
        False — never trade on unverified state.
        """
        try:
            positions: List[Dict[str, Any]] = await self._exchange.fetch_positions(
                [symbol]
            )
        except NotSupported:
            positions = []  # spot venue: order check below is authoritative
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — fail-safe
            logger.error(
                "%s: fetch_positions failed — refusing entry on unverified state",
                symbol,
                exc_info=True,
            )
            return False

        try:
            open_orders: List[Dict[str, Any]] = await self._exchange.fetch_open_orders(
                symbol
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — fail-safe
            logger.error(
                "%s: fetch_open_orders failed — refusing entry on unverified state",
                symbol,
                exc_info=True,
            )
            return False

        has_position: bool = any(
            p.get("symbol") in (symbol, None)
            and abs(Decimal(str(p.get("contracts") or p.get("amount") or 0))) > _ZERO
            for p in positions
        )
        if has_position or open_orders:
            self._set_state(symbol, AssetState.ACTIVE)
            logger.info(
                "%s: exchange truth shows exposure (positions=%s, open orders=%d) "
                "— local cache overridden, entries blocked",
                symbol,
                has_position,
                len(open_orders),
            )
            return False

        self._set_state(symbol, AssetState.IDLE)
        return True

    async def on_reconnect(self, symbol: str) -> None:
        """Feed reconciliation-hook adapter (registered with MultiAssetFeed)."""
        await self.reconcile_exchange_truth(symbol)

    async def emergency_flatten_all(self, symbols: Sequence[str]) -> None:
        """Kill-switch path: cancel every order, market-flatten every position.

        Invoked by the daily drawdown circuit breaker in main.py. Best-effort
        per symbol — one venue fault must never stop the remaining symbols
        from being flattened — and finishes by reconciling each asset against
        exchange truth.
        """
        for symbol in symbols:
            self._set_state(symbol, AssetState.CLOSING)
            try:
                cancel_all: Any = getattr(self._exchange, "cancel_all_orders", None)
                if callable(cancel_all):
                    await asyncio.wait_for(cancel_all(symbol), timeout=FLATTEN_TIMEOUT_S)
            except asyncio.CancelledError:
                raise
            except OrderNotFound:
                logger.info("%s: no resting orders to cancel", symbol)
            except Exception:  # noqa: BLE001 — proceed to flatten regardless
                logger.critical(
                    "%s: kill-switch cancel_all_orders failed", symbol, exc_info=True
                )

            try:
                positions: List[Dict[str, Any]] = await self._exchange.fetch_positions(
                    [symbol]
                )
            except NotSupported:
                positions = []
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.critical(
                    "%s: kill-switch position query failed — manual check required",
                    symbol,
                    exc_info=True,
                )
                positions = []

            for position in positions:
                size: Decimal = abs(
                    Decimal(str(position.get("contracts") or position.get("amount") or 0))
                )
                if size == _ZERO:
                    continue
                exit_side: str = (
                    "sell"
                    if str(position.get("side", "long")).lower() == "long"
                    else "buy"
                )
                try:
                    await asyncio.wait_for(
                        self._exchange.create_order(
                            symbol,
                            "market",
                            exit_side,
                            float(size),
                            None,
                            self._close_params(symbol),
                        ),
                        timeout=FLATTEN_TIMEOUT_S,
                    )
                    logger.critical(
                        "%s: kill-switch flattened %s %s (market close)",
                        symbol,
                        exit_side,
                        size,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    logger.critical(
                        "%s: KILL-SWITCH FLATTEN FAILED — manual intervention required",
                        symbol,
                        exc_info=True,
                    )

            await self.reconcile_exchange_truth(symbol)

    # ------------------------------------------------------------------ #
    # Routing                                                             #
    # ------------------------------------------------------------------ #

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
        """Route one signal through every sieve; place brackets only if clean.

        ``open_positions`` is the caller's live open-trade count for this
        symbol (from the journal); it is only consulted in concurrent mode.
        """
        if direction is SignalDirection.NEUTRAL:
            return ExecutionResult(
                status=ExecutionStatus.ABORT_NEUTRAL_SIGNAL,
                symbol=symbol,
                direction=direction,
                reason="neutral signal is never routable",
            )

        if self._max_open_trades == 1:
            if self.state(symbol) is not AssetState.IDLE:
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED_STATE,
                    symbol=symbol,
                    direction=direction,
                    reason=f"state machine lock: asset is {self.state(symbol).value}",
                )

            if not await self.reconcile_exchange_truth(symbol):
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED_EXCHANGE_TRUTH,
                    symbol=symbol,
                    direction=direction,
                    reason="exchange truth shows active exposure or is unverifiable",
                )
        else:
            # Concurrent mode: the venue legitimately shows exposure, so the
            # zero-exposure reconcile is replaced by an explicit cap. Only a
            # routing already mid-flight on this symbol still blocks.
            if self.state(symbol) is AssetState.PENDING_ENTRY:
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED_STATE,
                    symbol=symbol,
                    direction=direction,
                    reason="another entry is mid-flight for this symbol",
                )
            if (
                self._max_open_trades > 0
                and open_positions is not None
                and open_positions >= self._max_open_trades
            ):
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED_STATE,
                    symbol=symbol,
                    direction=direction,
                    reason=(
                        f"concurrent position cap reached "
                        f"({open_positions}/{self._max_open_trades})"
                    ),
                )

        self._set_state(symbol, AssetState.PENDING_ENTRY)
        try:
            return await self._route_locked(
                symbol, direction, dataframe, l2_order_book, trigger_price
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — routing must always resolve
            logger.critical("%s: unexpected routing fault", symbol, exc_info=True)
            await self.reconcile_exchange_truth(symbol)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                symbol=symbol,
                direction=direction,
                reason="unexpected routing fault — state re-reconciled",
            )

    async def _route_locked(
        self,
        symbol: str,
        direction: SignalDirection,
        dataframe: pd.DataFrame,
        l2_order_book: L2OrderBook,
        trigger_price: Decimal,
    ) -> ExecutionResult:
        # -- Sieve 1: slippage --------------------------------------------- #
        market_price: Decimal = await self._fetch_mid_price(symbol)
        deviation: Decimal = abs(market_price - trigger_price) / trigger_price
        if deviation > self._slippage_limit:
            self._set_state(symbol, AssetState.IDLE)
            logger.warning(
                "%s: slippage abort — trigger=%s market=%s deviation=%s > %s",
                symbol,
                trigger_price,
                market_price,
                deviation,
                self._slippage_limit,
            )
            return ExecutionResult(
                status=ExecutionStatus.ABORT_SLIPPAGE,
                symbol=symbol,
                direction=direction,
                reason="price slipped past 0.05% execution barrier",
                trigger_price=trigger_price,
                market_price=market_price,
                deviation=deviation,
            )

        # -- Sieve 1½: spread (execution cost) ------------------------------ #
        if self._max_spread_bps is not None and l2_order_book.is_populated:
            best_bid: Decimal = Decimal(str(l2_order_book.bids[0][0]))
            best_ask: Decimal = Decimal(str(l2_order_book.asks[0][0]))
            if best_bid > _ZERO and best_ask >= best_bid:
                mid: Decimal = (best_bid + best_ask) / Decimal("2")
                spread_bps: Decimal = (
                    (best_ask - best_bid) / mid * Decimal("10000")
                )
                if spread_bps > self._max_spread_bps:
                    self._set_state(symbol, AssetState.IDLE)
                    logger.warning(
                        "%s: spread abort — %s bps > %s bps allowed "
                        "(bid %s / ask %s)",
                        symbol,
                        spread_bps,
                        self._max_spread_bps,
                        best_bid,
                        best_ask,
                    )
                    return ExecutionResult(
                        status=ExecutionStatus.ABORT_SPREAD,
                        symbol=symbol,
                        direction=direction,
                        reason=(
                            f"spread {spread_bps:.2f} bps exceeds "
                            f"{self._max_spread_bps} bps limit"
                        ),
                        trigger_price=trigger_price,
                        market_price=market_price,
                    )

        # -- Sieve 2: volatility capital sizer ----------------------------- #
        report: RegimeReport = self._gatekeeper.evaluate(dataframe, l2_order_book)
        if not report.sufficient_data or report.atr is None or report.atr_sma is None:
            self._set_state(symbol, AssetState.IDLE)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                symbol=symbol,
                direction=direction,
                reason="ATR inputs unavailable — cannot size capital",
            )
        atr: Decimal = report.atr
        atr_sma: Decimal = report.atr_sma

        requested_amount: Decimal
        if self._fixed_notional_is_min:
            # Harvester mode at venue minimum: smallest size the exchange
            # will accept, independent of equity and the variant's record.
            requested_amount = self._venue_minimum_amount(symbol, market_price)
        elif self._fixed_notional is not None:
            # Harvester mode: constant tiny notional, no Kelly, no equity
            # read — sizing must not depend on the variant's own record.
            requested_amount = self._fixed_notional / market_price
        else:
            equity: Decimal = await self._fetch_quote_equity(symbol)
            fraction: Decimal = self._allocation_fraction(atr, atr_sma)
            requested_amount = (equity * fraction) / market_price

        # -- Sieve 3: liquidity -------------------------------------------- #
        resting_depth: Decimal = self._top_tier_depth(l2_order_book, direction)
        liquidity_cap: Decimal = resting_depth * self._liquidity_cap_fraction
        if liquidity_cap <= _ZERO:
            self._set_state(symbol, AssetState.IDLE)
            return ExecutionResult(
                status=ExecutionStatus.ABORT_LIQUIDITY,
                symbol=symbol,
                direction=direction,
                reason="no resting liquidity in top tiers — phantom book",
                requested_amount=requested_amount,
            )
        if requested_amount > liquidity_cap:
            logger.info(
                "%s: liquidity sieve downscaled %s -> %s (5%% of top-%d depth %s)",
                symbol,
                requested_amount,
                liquidity_cap,
                self._liquidity_tiers,
                resting_depth,
            )
        sized_amount: Decimal = min(requested_amount, liquidity_cap)

        # -- Precision quantization (banker's rounding) --------------------- #
        amount: Decimal = self._quantize_amount(symbol, sized_amount)
        if amount <= _ZERO or amount < self._min_amount(symbol):
            self._set_state(symbol, AssetState.IDLE)
            return ExecutionResult(
                status=ExecutionStatus.ABORT_MIN_SIZE,
                symbol=symbol,
                direction=direction,
                reason="quantized size below venue minimum — not routable",
                requested_amount=requested_amount,
                executed_amount=amount,
            )

        # -- Bracket routing ------------------------------------------------ #
        return await self._place_bracket(
            symbol,
            direction,
            amount,
            requested_amount,
            trigger_price,
            market_price,
            deviation,
            atr,
        )

    # ------------------------------------------------------------------ #
    # Bracket placement                                                   #
    # ------------------------------------------------------------------ #

    async def _place_bracket(
        self,
        symbol: str,
        direction: SignalDirection,
        amount: Decimal,
        requested_amount: Decimal,
        trigger_price: Decimal,
        market_price: Decimal,
        deviation: Decimal,
        atr: Decimal,
    ) -> ExecutionResult:
        entry_side: str = "buy" if direction is SignalDirection.LONG else "sell"
        exit_side: str = "sell" if entry_side == "buy" else "buy"

        try:
            entry_order: Dict[str, Any] = await asyncio.wait_for(
                self._exchange.create_order(symbol, "market", entry_side, float(amount)),
                timeout=self._order_timeout_s,
            )
        except (asyncio.TimeoutError, TimeoutError):
            logger.critical(
                "%s: entry order timed out after %.0f ms — order state unknown, "
                "reconciling against exchange truth",
                symbol,
                self._order_timeout_s * 1_000,
            )
            await self.reconcile_exchange_truth(symbol)
            return ExecutionResult(
                status=ExecutionStatus.ABORT_ENTRY_TIMEOUT,
                symbol=symbol,
                direction=direction,
                reason="entry placement exceeded 200 ms network boundary",
                trigger_price=trigger_price,
                market_price=market_price,
                deviation=deviation,
                requested_amount=requested_amount,
                executed_amount=amount,
            )

        fill_raw: Any = entry_order.get("average") or entry_order.get("price")
        fill_price: Decimal = (
            Decimal(str(fill_raw)) if fill_raw is not None else market_price
        )
        price_quantum: Decimal = self._price_quantum(symbol)
        offset_tp: Decimal = self._take_profit_atr_mult * atr
        offset_sl: Decimal = self._stop_loss_atr_mult * atr
        if direction is SignalDirection.LONG:
            tp_price: Decimal = self._quantize_step(fill_price + offset_tp, price_quantum)
            sl_price: Decimal = self._quantize_step(fill_price - offset_sl, price_quantum)
        else:
            tp_price = self._quantize_step(fill_price - offset_tp, price_quantum)
            sl_price = self._quantize_step(fill_price + offset_sl, price_quantum)

        try:
            tp_order_id: str
            sl_order_id: str
            if self._is_spot(symbol):
                # Spot: two independent exit orders would double-lock the same
                # balance, so both legs ride one atomic OCO order list.
                tp_order_id, sl_order_id = await asyncio.wait_for(
                    self._place_spot_oco(symbol, exit_side, amount, tp_price, sl_price),
                    timeout=self._order_timeout_s,
                )
            else:
                tp_order: Dict[str, Any]
                sl_order: Dict[str, Any]
                tp_order, sl_order = await asyncio.wait_for(
                    asyncio.gather(
                        self._exchange.create_order(
                            symbol,
                            "limit",
                            exit_side,
                            float(amount),
                            float(tp_price),
                            {"reduceOnly": True},
                        ),
                        self._exchange.create_order(
                            symbol,
                            "market",
                            exit_side,
                            float(amount),
                            None,
                            {"stopPrice": float(sl_price), "reduceOnly": True},
                        ),
                    ),
                    timeout=self._order_timeout_s,
                )
                tp_order_id = str(tp_order.get("id"))
                sl_order_id = str(sl_order.get("id"))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — incl. TimeoutError: naked position
            logger.critical(
                "%s: bracket placement failed after a FILLED entry — "
                "emergency flatten engaged",
                symbol,
                exc_info=True,
            )
            await self._emergency_flatten(symbol, exit_side, amount)
            await self.reconcile_exchange_truth(symbol)
            return ExecutionResult(
                status=ExecutionStatus.ABORT_BRACKET_FAILED,
                symbol=symbol,
                direction=direction,
                reason="bracket legs failed post-fill — position flattened",
                trigger_price=trigger_price,
                market_price=market_price,
                deviation=deviation,
                requested_amount=requested_amount,
                executed_amount=amount,
                entry_order_id=str(entry_order.get("id")),
                entry_fill_price=fill_price,
            )

        # Concurrent mode never holds the ACTIVE lock — the journal count
        # (checked at routing time) is the position limiter instead.
        self._set_state(
            symbol,
            AssetState.ACTIVE if self._max_open_trades == 1 else AssetState.IDLE,
        )
        logger.info(
            "%s: bracket live — %s %s @ %s | TP %s | SL %s (2.5x ATR=%s)",
            symbol,
            direction.value,
            amount,
            fill_price,
            tp_price,
            sl_price,
            atr,
        )
        return ExecutionResult(
            status=ExecutionStatus.EXECUTED,
            symbol=symbol,
            direction=direction,
            reason="all sieves passed — bracket placed",
            trigger_price=trigger_price,
            market_price=market_price,
            deviation=deviation,
            requested_amount=requested_amount,
            executed_amount=amount,
            entry_order_id=str(entry_order.get("id")),
            take_profit_order_id=tp_order_id,
            stop_loss_order_id=sl_order_id,
            entry_fill_price=fill_price,
            take_profit_price=tp_price,
            stop_loss_price=sl_price,
        )

    async def _place_spot_oco(
        self,
        symbol: str,
        exit_side: str,
        amount: Decimal,
        tp_price: Decimal,
        sl_price: Decimal,
    ) -> Tuple[str, str]:
        """Place both bracket legs as one Binance spot OCO order list.

        The profit leg is a resting LIMIT_MAKER; the protective leg is a
        STOP_LOSS_LIMIT (spot forbids plain STOP_LOSS market on major pairs)
        whose limit price sits ``SL_LIMIT_BUFFER`` through the stop so it is
        marketable the instant it triggers. Returns (tp_order_id, sl_order_id).
        """
        market: Dict[str, Any] = self._market(symbol)
        price_quantum: Decimal = self._price_quantum(symbol)
        if exit_side == "sell":
            # Closing a long: profit rests above market, stop guards below.
            sl_limit: Decimal = self._quantize_step(
                sl_price * (_ONE - SL_LIMIT_BUFFER), price_quantum
            )
            legs: Dict[str, Any] = {
                "aboveType": "LIMIT_MAKER",
                "abovePrice": float(tp_price),
                "belowType": "STOP_LOSS_LIMIT",
                "belowStopPrice": float(sl_price),
                "belowPrice": float(sl_limit),
                "belowTimeInForce": "GTC",
            }
        else:
            # Closing a short sale: profit rests below market, stop guards above.
            sl_limit = self._quantize_step(
                sl_price * (_ONE + SL_LIMIT_BUFFER), price_quantum
            )
            legs = {
                "belowType": "LIMIT_MAKER",
                "belowPrice": float(tp_price),
                "aboveType": "STOP_LOSS_LIMIT",
                "aboveStopPrice": float(sl_price),
                "abovePrice": float(sl_limit),
                "aboveTimeInForce": "GTC",
            }
        request: Dict[str, Any] = {
            "symbol": market.get("id") or symbol.replace("/", ""),
            "side": exit_side.upper(),
            "quantity": float(amount),
            **legs,
        }
        response: Dict[str, Any] = await self._exchange.privatePostOrderListOco(request)
        tp_id: str = str(response.get("orderListId", ""))
        sl_id: str = tp_id
        for report in response.get("orderReports", []):
            order_type: str = str(report.get("type", ""))
            if order_type == "LIMIT_MAKER":
                tp_id = str(report.get("orderId"))
            elif order_type.startswith("STOP_LOSS"):
                sl_id = str(report.get("orderId"))
        return tp_id, sl_id

    async def _emergency_flatten(
        self, symbol: str, exit_side: str, amount: Decimal
    ) -> None:
        """Market close of an unprotected position. Last resort."""
        self._set_state(symbol, AssetState.CLOSING)
        try:
            cancel_all: Any = getattr(self._exchange, "cancel_all_orders", None)
            if callable(cancel_all):
                await asyncio.wait_for(cancel_all(symbol), timeout=FLATTEN_TIMEOUT_S)
        except OrderNotFound:
            # Binance answers -2011 when there is simply nothing to cancel.
            logger.info("%s: no resting orders to cancel pre-flatten", symbol)
        except Exception:  # noqa: BLE001 — proceed to flatten regardless
            logger.error("%s: cancel_all_orders failed pre-flatten", symbol, exc_info=True)
        try:
            await asyncio.wait_for(
                self._exchange.create_order(
                    symbol,
                    "market",
                    exit_side,
                    float(amount),
                    None,
                    self._close_params(symbol),
                ),
                timeout=FLATTEN_TIMEOUT_S,
            )
            logger.critical("%s: naked position flattened (market close)", symbol)
        except Exception:  # noqa: BLE001
            logger.critical(
                "%s: EMERGENCY FLATTEN FAILED — manual intervention required",
                symbol,
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # Sizing primitives                                                   #
    # ------------------------------------------------------------------ #

    def _volatility_modifier(self, atr: Decimal, atr_sma: Decimal) -> Decimal:
        """ATR_baseline / ATR_current, clamped to <= 1.

        Chaotic volatility spikes (current ATR above baseline) downscale
        capital proportionally; calm regimes never upscale past the Kelly
        baseline — dampening is one-directional by design.
        """
        if atr <= _ZERO or atr_sma <= _ZERO:
            return _ZERO
        return min(atr_sma / atr, _ONE)

    def _allocation_fraction(self, atr: Decimal, atr_sma: Decimal) -> Decimal:
        return self._tracker.kelly_allocation() * self._volatility_modifier(
            atr, atr_sma
        )

    def _top_tier_depth(
        self, book: L2OrderBook, direction: SignalDirection
    ) -> Decimal:
        """Cumulative resting size in the top tiers of the consumed side.

        Longs consume asks; shorts consume bids.
        """
        side: Sequence[Tuple[float, float]] = (
            book.asks if direction is SignalDirection.LONG else book.bids
        )
        return sum(
            (Decimal(str(size)) for _, size in side[: self._liquidity_tiers]), _ZERO
        )

    # ------------------------------------------------------------------ #
    # Exchange seam helpers                                               #
    # ------------------------------------------------------------------ #

    async def _fetch_mid_price(self, symbol: str) -> Decimal:
        ticker: Dict[str, Any] = await self._exchange.fetch_ticker(symbol)
        bid: Any = ticker.get("bid")
        ask: Any = ticker.get("ask")
        if bid is not None and ask is not None:
            return (Decimal(str(bid)) + Decimal(str(ask))) / Decimal(2)
        last: Any = ticker.get("last")
        if last is None:
            raise RuntimeError(f"{symbol}: ticker carries no usable price")
        return Decimal(str(last))

    async def _fetch_quote_equity(self, symbol: str) -> Decimal:
        quote: str = symbol.split("/")[1].split(":")[0]
        balance: Dict[str, Any] = await self._exchange.fetch_balance()
        entry: Any = balance.get(quote)
        free: Any = entry.get("free") if isinstance(entry, dict) else None
        if free is None:
            logger.warning("%s: no free %s balance reported — sizing to zero", symbol, quote)
            return _ZERO
        return Decimal(str(free))

    def _market(self, symbol: str) -> Dict[str, Any]:
        markets: Any = getattr(self._exchange, "markets", None) or {}
        market: Any = markets.get(symbol)
        return market if isinstance(market, dict) else {}

    def _is_spot(self, symbol: str) -> bool:
        """True when the venue market is spot — no positions, no reduceOnly."""
        market: Dict[str, Any] = self._market(symbol)
        return bool(market.get("spot")) or market.get("type") == "spot"

    def _close_params(self, symbol: str) -> Dict[str, Any]:
        """Params for a position-closing order.

        ``reduceOnly`` exists only on derivatives venues; Binance spot rejects
        any unread parameter outright (-1104), so spot closes send none.
        """
        return {} if self._is_spot(symbol) else {"reduceOnly": True}

    def _amount_quantum(self, symbol: str) -> Decimal:
        return self._quantum(
            self._market(symbol).get("precision", {}).get("amount"), Decimal("1e-8")
        )

    def _price_quantum(self, symbol: str) -> Decimal:
        return self._quantum(
            self._market(symbol).get("precision", {}).get("price"), Decimal("1e-8")
        )

    @staticmethod
    def _quantum(raw_precision: Any, default: Decimal) -> Decimal:
        """Unify CCXT precision conventions (decimal places vs tick size)."""
        if raw_precision is None:
            return default
        if isinstance(raw_precision, int):
            return Decimal(1).scaleb(-raw_precision)
        return Decimal(str(raw_precision))

    @staticmethod
    def _quantize_step(value: Decimal, step: Decimal) -> Decimal:
        """Snap ``value`` onto the venue step grid with banker's rounding."""
        if step <= _ZERO:
            return value
        return (value / step).quantize(_ONE, rounding=ROUND_HALF_EVEN) * step

    def _quantize_amount(self, symbol: str, amount: Decimal) -> Decimal:
        return self._quantize_step(amount, self._amount_quantum(symbol))

    def _min_amount(self, symbol: str) -> Decimal:
        raw: Any = (
            self._market(symbol).get("limits", {}).get("amount", {}).get("min")
        )
        return Decimal(str(raw)) if raw is not None else _ZERO

    def _min_cost(self, symbol: str) -> Decimal:
        """Venue minimum notional (quote); Binance spot default is 10 USDT."""
        raw: Any = (
            (self._market(symbol).get("limits", {}).get("cost") or {}).get("min")
        )
        return Decimal(str(raw)) if raw is not None else Decimal("10")

    def _venue_minimum_amount(self, symbol: str, market_price: Decimal) -> Decimal:
        """Smallest routable base amount, with 10% headroom above both the
        minimum-amount and minimum-notional floors so banker's-rounding
        quantization can never push the order below a venue limit."""
        floor: Decimal = max(
            self._min_amount(symbol), self._min_cost(symbol) / market_price
        )
        return floor * Decimal("1.10")


# --------------------------------------------------------------------------- #
# Embedded test architecture                                                   #
# --------------------------------------------------------------------------- #

_SYMBOL: Final[str] = "BTC/USDT"
_BAR_MS: Final[int] = 5 * 60 * 1_000
_T0_MS: Final[int] = 1_750_000_500_000 - (1_750_000_500_000 % _BAR_MS)


def _trending_frame(num_bars: int = 80) -> pd.DataFrame:
    """Expanding uptrend in feed layout (current ATR > its 20-SMA baseline)."""
    rows: List[Tuple[float, float, float, float, float]] = []
    price: float = 100.0
    for i in range(num_bars):
        step: float = 0.5 + 0.05 * i
        o: float = price
        c: float = o + step
        rows.append((o, c + 0.10 * step, o - 0.05 * step, c, 100.0))
        price = c
    ts_ms: List[int] = [_T0_MS + i * _BAR_MS for i in range(num_bars)]
    frame: pd.DataFrame = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(ts_ms, unit="ms", utc=True),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [100.0] * num_bars,
        }
    )
    frame["amount"] = frame["close"] * frame["volume"]
    cols: List[str] = ["open", "high", "low", "close", "volume", "amount"]
    frame[cols] = frame[cols].astype("float64")
    return frame


def _book(mid: float, *, ask_sizes: Sequence[float], bid_sizes: Sequence[float]) -> L2OrderBook:
    bids: Tuple[Tuple[float, float], ...] = tuple(
        (mid - 0.01 * (i + 1), size) for i, size in enumerate(bid_sizes)
    )
    asks: Tuple[Tuple[float, float], ...] = tuple(
        (mid + 0.01 * (i + 1), size) for i, size in enumerate(ask_sizes)
    )
    return L2OrderBook(symbol=_SYMBOL, bids=bids, asks=asks, timestamp_ms=_T0_MS)


class _FakeExchange:
    """Sandbox-flagged exchange double recording every order request."""

    def __init__(
        self,
        *,
        mid_price: float,
        free_usdt: float = 10_000.0,
        positions: Optional[List[Dict[str, Any]]] = None,
        open_orders: Optional[List[Dict[str, Any]]] = None,
        fail_order_types: Optional[Sequence[str]] = None,
    ) -> None:
        self.isSandboxModeEnabled: bool = True
        self.mid_price: float = mid_price
        self.free_usdt: float = free_usdt
        self.positions: List[Dict[str, Any]] = positions or []
        self.open_orders_list: List[Dict[str, Any]] = open_orders or []
        self.fail_order_types: Tuple[str, ...] = tuple(fail_order_types or ())
        self.orders: List[Dict[str, Any]] = []
        self.cancelled: List[str] = []
        self.markets: Dict[str, Dict[str, Any]] = {
            _SYMBOL: {
                "precision": {"amount": 3, "price": 2},
                "limits": {"amount": {"min": 0.001}},
            }
        }

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return {"bid": self.mid_price, "ask": self.mid_price, "last": self.mid_price}

    async def fetch_balance(self) -> Dict[str, Any]:
        return {"USDT": {"free": self.free_usdt}}

    async def fetch_positions(self, symbols: List[str]) -> List[Dict[str, Any]]:
        return list(self.positions)

    async def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return list(self.open_orders_list)

    async def cancel_all_orders(self, symbol: str) -> None:
        self.cancelled.append(symbol)

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if order_type in self.fail_order_types:
            raise RuntimeError(f"simulated venue rejection for {order_type} order")
        record: Dict[str, Any] = {
            "id": f"ord-{len(self.orders) + 1}",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "params": dict(params or {}),
            "average": self.mid_price,
        }
        self.orders.append(record)
        if (
            order_type == "market"
            and record["params"].get("reduceOnly")
            and "stopPrice" not in record["params"]
        ):
            self.positions = []  # venue truth: a reduce-only close flattens
        return record


class _FakeSpotExchange(_FakeExchange):
    """Spot venue truth: no reduceOnly, brackets must ride a single OCO list."""

    def __init__(self, *args: Any, fail_oco: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.markets[_SYMBOL]["spot"] = True
        self.markets[_SYMBOL]["type"] = "spot"
        self.markets[_SYMBOL]["id"] = _SYMBOL.replace("/", "")
        self.fail_oco: bool = fail_oco
        self.oco_requests: List[Dict[str, Any]] = []

    async def privatePostOrderListOco(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.fail_oco:
            raise RuntimeError("simulated venue rejection for OCO order list")
        self.oco_requests.append(dict(request))
        return {
            "orderListId": 99,
            "orderReports": [
                {"orderId": 901, "type": "LIMIT_MAKER"},
                {"orderId": 902, "type": "STOP_LOSS_LIMIT"},
            ],
        }


def _router(exchange: _FakeExchange) -> ExecutionRouter:
    return ExecutionRouter(exchange)


class ExecutionRouterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.frame: pd.DataFrame = _trending_frame()
        self.mid: float = float(self.frame["close"].iloc[-1])
        self.trigger: Decimal = Decimal(str(self.mid))

    # -- volatility capital sizer ---------------------------------------- #

    def test_sizing_downscales_when_atr_spikes(self) -> None:
        router = _router(_FakeExchange(mid_price=self.mid))
        calm: Decimal = router._allocation_fraction(Decimal("2"), Decimal("2"))
        spiked: Decimal = router._allocation_fraction(Decimal("4"), Decimal("2"))
        self.assertEqual(spiked, calm / 2)  # exactly halved at 2x baseline ATR
        self.assertGreater(calm, Decimal("0"))

    def test_modifier_never_upscales_in_calm_regimes(self) -> None:
        router = _router(_FakeExchange(mid_price=self.mid))
        self.assertEqual(
            router._volatility_modifier(Decimal("1"), Decimal("2")), Decimal("1")
        )
        self.assertEqual(
            router._volatility_modifier(Decimal("4"), Decimal("2")), Decimal("0.5")
        )

    # -- fixed-notional (harvester) sizing -------------------------------- #

    async def test_fixed_notional_bypasses_kelly_and_equity(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        # A deliberately terrible tracker: Kelly would allocate zero.
        tracker = PerformanceTracker()
        for _ in range(50):
            tracker.record_trade(Decimal("-1"))
        router = ExecutionRouter(
            exchange,
            tracker=tracker,
            fixed_trade_notional=self.trigger,  # exactly 1.0 base unit worth
        )
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.EXECUTED)
        # notional/price = 1.0 — routed despite the zero-Kelly record.
        self.assertEqual(result.executed_amount, Decimal("1.000"))

    async def test_fixed_notional_still_obeys_liquidity_cap(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        router = ExecutionRouter(
            exchange, fixed_trade_notional=self.trigger * Decimal("10")
        )
        shallow = _book(
            self.mid, ask_sizes=(0.4, 0.3, 0.3, 99.0), bid_sizes=(50, 50, 50)
        )
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, shallow, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.EXECUTED)
        # requested 10 base units, but 5% of top-3 depth (1.0) caps it.
        self.assertEqual(result.executed_amount, Decimal("0.050"))

    async def test_spread_sieve_blocks_wide_markets(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        router = ExecutionRouter(
            exchange,
            fixed_trade_notional=self.trigger,
            max_spread_bps=Decimal("0.5"),
        )
        # _book builds bid=mid-0.01 / ask=mid+0.01 -> ~0.67 bps at mid~298.
        book = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, book, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.ABORT_SPREAD)
        self.assertEqual(exchange.orders, [])
        self.assertEqual(router.state(_SYMBOL), AssetState.IDLE)

    async def test_spread_sieve_admits_tight_markets(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        router = ExecutionRouter(
            exchange,
            fixed_trade_notional=self.trigger,
            max_spread_bps=Decimal("5"),
        )
        book = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, book, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.EXECUTED)

    def test_bad_spread_limit_refuses_boot(self) -> None:
        os.environ["MAX_SPREAD_BPS"] = "-3"
        try:
            with self.assertRaises(RuntimeError):
                ExecutionRouter(_FakeExchange(mid_price=self.mid))
        finally:
            del os.environ["MAX_SPREAD_BPS"]

    async def test_unlimited_mode_routes_concurrent_brackets(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        router = ExecutionRouter(
            exchange,
            fixed_trade_notional=self.trigger,
            max_open_trades=0,  # unlimited
        )
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        first = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger,
            open_positions=0,
        )
        second = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger,
            open_positions=1,
        )
        self.assertEqual(first.status, ExecutionStatus.EXECUTED)
        self.assertEqual(second.status, ExecutionStatus.EXECUTED)
        # No ACTIVE lock held between entries in concurrent mode.
        self.assertEqual(router.state(_SYMBOL), AssetState.IDLE)
        self.assertEqual(len(exchange.orders), 6)  # two full brackets

    async def test_concurrent_cap_blocks_when_reached(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        router = ExecutionRouter(
            exchange, fixed_trade_notional=self.trigger, max_open_trades=2
        )
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        blocked = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger,
            open_positions=2,
        )
        self.assertEqual(blocked.status, ExecutionStatus.BLOCKED_STATE)
        self.assertEqual(exchange.orders, [])
        allowed = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger,
            open_positions=1,
        )
        self.assertEqual(allowed.status, ExecutionStatus.EXECUTED)

    async def test_min_notional_sizing_uses_venue_floors(self) -> None:
        os.environ["FIXED_TRADE_NOTIONAL"] = "min"
        try:
            exchange = _FakeExchange(mid_price=self.mid)
            router = ExecutionRouter(exchange)
            self.assertTrue(router._fixed_notional_is_min)
            deep = _book(
                self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100)
            )
            result = await router.route_trade(
                _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger
            )
            self.assertEqual(result.status, ExecutionStatus.EXECUTED)
            assert result.executed_amount is not None
            notional: Decimal = result.executed_amount * Decimal(str(self.mid))
            # Floor is min-cost (10 USDT default) with 10% headroom; the
            # quantized order must sit just above it, never far above.
            self.assertGreaterEqual(notional, Decimal("10"))
            self.assertLess(notional, Decimal("14"))
        finally:
            del os.environ["FIXED_TRADE_NOTIONAL"]

    def test_bad_max_open_trades_refuses_boot(self) -> None:
        with self.assertRaises(RuntimeError):
            ExecutionRouter(_FakeExchange(mid_price=self.mid), max_open_trades=-1)
        os.environ["MAX_OPEN_TRADES_PER_SYMBOL"] = "many"
        try:
            with self.assertRaises(RuntimeError):
                ExecutionRouter(_FakeExchange(mid_price=self.mid))
        finally:
            del os.environ["MAX_OPEN_TRADES_PER_SYMBOL"]

    def test_fixed_notional_env_is_honoured_and_validated(self) -> None:
        os.environ["FIXED_TRADE_NOTIONAL"] = "25"
        try:
            router = ExecutionRouter(_FakeExchange(mid_price=self.mid))
            self.assertEqual(router._fixed_notional, Decimal("25"))
        finally:
            del os.environ["FIXED_TRADE_NOTIONAL"]
        os.environ["FIXED_TRADE_NOTIONAL"] = "-5"
        try:
            with self.assertRaises(RuntimeError):
                ExecutionRouter(_FakeExchange(mid_price=self.mid))
        finally:
            del os.environ["FIXED_TRADE_NOTIONAL"]
        os.environ["FIXED_TRADE_NOTIONAL"] = "not-a-number"
        try:
            with self.assertRaises(RuntimeError):
                ExecutionRouter(_FakeExchange(mid_price=self.mid))
        finally:
            del os.environ["FIXED_TRADE_NOTIONAL"]

    # -- slippage sieve ---------------------------------------------------- #

    async def test_aborts_when_price_slips_past_boundary(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        router = _router(exchange)
        slipped_trigger: Decimal = self.trigger * Decimal("0.998")  # 0.2% away
        result = await router.route_trade(
            _SYMBOL,
            SignalDirection.LONG,
            self.frame,
            _book(self.mid, ask_sizes=(50, 50, 50), bid_sizes=(50, 50, 50)),
            slipped_trigger,
        )
        self.assertEqual(result.status, ExecutionStatus.ABORT_SLIPPAGE)
        self.assertEqual(exchange.orders, [])  # nothing ever reached the venue
        self.assertEqual(router.state(_SYMBOL), AssetState.IDLE)
        assert result.deviation is not None
        self.assertGreater(result.deviation, SLIPPAGE_LIMIT)

    # -- liquidity sieve ---------------------------------------------------- #

    async def test_caps_size_on_shallow_top3_depth(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid, free_usdt=1_000_000.0)
        router = _router(exchange)
        shallow = _book(self.mid, ask_sizes=(0.4, 0.3, 0.3, 99.0), bid_sizes=(50, 50, 50))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, shallow, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.EXECUTED)
        # top-3 asks = 1.0 (4th tier excluded) -> 5% cap = 0.05 exactly.
        self.assertEqual(result.executed_amount, Decimal("0.050"))
        assert result.requested_amount is not None
        self.assertGreater(result.requested_amount, Decimal("0.05"))
        self.assertEqual(float(exchange.orders[0]["amount"]), 0.05)

    async def test_aborts_on_phantom_empty_book_side(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        router = _router(exchange)
        empty = L2OrderBook(symbol=_SYMBOL, bids=((1.0, 1.0),), asks=(), timestamp_ms=_T0_MS)
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, empty, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.ABORT_LIQUIDITY)
        self.assertEqual(exchange.orders, [])

    # -- bracket routing ----------------------------------------------------- #

    async def test_full_bracket_with_sl_at_2_5_atr(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        router = _router(exchange)
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.EXECUTED)
        self.assertEqual(len(exchange.orders), 3)
        entry, tp, sl = exchange.orders
        self.assertEqual((entry["type"], entry["side"]), ("market", "buy"))
        self.assertEqual((tp["type"], tp["side"]), ("limit", "sell"))
        self.assertEqual((sl["type"], sl["side"]), ("market", "sell"))
        self.assertTrue(sl["params"]["reduceOnly"])
        self.assertTrue(tp["params"]["reduceOnly"])

        report = MarketGatekeeper().evaluate(self.frame, deep)
        assert report.atr is not None and result.entry_fill_price is not None
        expected_sl: Decimal = ExecutionRouter._quantize_step(
            result.entry_fill_price - Decimal("2.5") * report.atr, Decimal("0.01")
        )
        self.assertEqual(result.stop_loss_price, expected_sl)
        self.assertEqual(Decimal(str(sl["params"]["stopPrice"])), expected_sl)
        self.assertEqual(router.state(_SYMBOL), AssetState.ACTIVE)

    async def test_state_lock_blocks_second_entry(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        router = _router(exchange)
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        first = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger
        )
        self.assertEqual(first.status, ExecutionStatus.EXECUTED)
        second = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger
        )
        self.assertEqual(second.status, ExecutionStatus.BLOCKED_STATE)
        self.assertEqual(len(exchange.orders), 3)  # no additional routing

    async def test_exchange_truth_overrides_local_cache(self) -> None:
        exchange = _FakeExchange(
            mid_price=self.mid,
            positions=[{"symbol": _SYMBOL, "contracts": 1.0}],
        )
        router = _router(exchange)
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.BLOCKED_EXCHANGE_TRUTH)
        self.assertEqual(router.state(_SYMBOL), AssetState.ACTIVE)
        self.assertEqual(exchange.orders, [])

    async def test_bracket_failure_triggers_emergency_flatten(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid, fail_order_types=("limit",))
        router = _router(exchange)
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.ABORT_BRACKET_FAILED)
        self.assertIn(_SYMBOL, exchange.cancelled)
        flatten_orders = [
            o
            for o in exchange.orders
            if o["type"] == "market"
            and o["side"] == "sell"
            and o["params"].get("reduceOnly")
            and "stopPrice" not in o["params"]
        ]
        self.assertEqual(len(flatten_orders), 1)  # naked position was closed

    async def test_neutral_signal_is_never_routable(self) -> None:
        exchange = _FakeExchange(mid_price=self.mid)
        router = _router(exchange)
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.NEUTRAL, self.frame, deep, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.ABORT_NEUTRAL_SIGNAL)
        self.assertEqual(exchange.orders, [])

    async def test_emergency_flatten_all_closes_positions(self) -> None:
        exchange = _FakeExchange(
            mid_price=self.mid,
            positions=[{"symbol": _SYMBOL, "contracts": 2.0, "side": "long"}],
        )
        router = _router(exchange)
        await router.emergency_flatten_all([_SYMBOL])
        self.assertIn(_SYMBOL, exchange.cancelled)
        closes = [o for o in exchange.orders if o["params"].get("reduceOnly")]
        self.assertEqual(len(closes), 1)
        self.assertEqual((closes[0]["type"], closes[0]["side"]), ("market", "sell"))
        self.assertEqual(closes[0]["amount"], 2.0)
        # Post-flatten reconcile sees a flat venue -> asset returns to IDLE.
        self.assertEqual(router.state(_SYMBOL), AssetState.IDLE)

    # -- spot venue routing -------------------------------------------------- #

    async def test_spot_bracket_rides_single_oco_list(self) -> None:
        exchange = _FakeSpotExchange(mid_price=self.mid)
        router = _router(exchange)
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.EXECUTED)
        # Exactly one plain order (the entry) plus one OCO list — two separate
        # exit legs would double-lock the same spot balance.
        self.assertEqual(len(exchange.orders), 1)
        self.assertEqual(len(exchange.oco_requests), 1)
        self.assertNotIn("reduceOnly", exchange.orders[0]["params"])
        oco = exchange.oco_requests[0]
        self.assertEqual(oco["side"], "SELL")
        self.assertEqual(oco["aboveType"], "LIMIT_MAKER")  # TP rests above a long
        self.assertEqual(oco["belowType"], "STOP_LOSS_LIMIT")
        self.assertLess(oco["belowPrice"], oco["belowStopPrice"])  # marketable SL
        self.assertEqual(result.take_profit_order_id, "901")
        self.assertEqual(result.stop_loss_order_id, "902")

    async def test_spot_short_bracket_inverts_oco_legs(self) -> None:
        exchange = _FakeSpotExchange(mid_price=self.mid)
        router = _router(exchange)
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.SHORT, self.frame, deep, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.EXECUTED)
        oco = exchange.oco_requests[0]
        self.assertEqual(oco["side"], "BUY")
        self.assertEqual(oco["belowType"], "LIMIT_MAKER")  # TP rests below a short
        self.assertEqual(oco["aboveType"], "STOP_LOSS_LIMIT")
        self.assertGreater(oco["abovePrice"], oco["aboveStopPrice"])  # marketable SL

    async def test_spot_flatten_never_sends_reduce_only(self) -> None:
        exchange = _FakeSpotExchange(mid_price=self.mid, fail_oco=True)
        router = _router(exchange)
        deep = _book(self.mid, ask_sizes=(100, 100, 100), bid_sizes=(100, 100, 100))
        result = await router.route_trade(
            _SYMBOL, SignalDirection.LONG, self.frame, deep, self.trigger
        )
        self.assertEqual(result.status, ExecutionStatus.ABORT_BRACKET_FAILED)
        # Entry then emergency market close — neither carries futures params.
        self.assertEqual(len(exchange.orders), 2)
        for order in exchange.orders:
            self.assertNotIn("reduceOnly", order["params"])
        self.assertEqual(
            (exchange.orders[1]["type"], exchange.orders[1]["side"]),
            ("market", "sell"),
        )

    def test_refuses_construction_without_sandbox(self) -> None:
        class _LiveOnly:
            pass

        with self.assertRaises(RuntimeError):
            ExecutionRouter(_LiveOnly())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

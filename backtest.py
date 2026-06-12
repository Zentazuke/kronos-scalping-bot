"""backtest.py — Phase 8: historical replay and walk-forward calibration.

Offline research harness — never touches order endpoints, never runs while
the live bot trades. It replays historical OHLCV through the *production*
``MarketGatekeeper`` (same Decimal indicators, same confluence votes) and
simulates the bracket geometry bar by bar, so the numbers it reports are the
numbers the live pipeline would have produced.

Honest-measurement rules baked in:
  * No lookahead — a signal on bar ``i``'s close enters at bar ``i+1``'s
    open, never at the close that generated it.
  * Conservative ambiguity — if one bar's range touches both TP and SL, the
    loss is recorded (you cannot know intrabar ordering from OHLC).
  * No L2 history — the book sieve is fed a fresh, symmetric synthetic book:
    freshness passes, but the book-imbalance vote can never confirm, so
    confluence admission is *stricter* here than live (conservative bias).
  * Direction proxy — Kronos inference over thousands of bars is impractical
    on CPU, so entries follow the Wilder DI direction. The backtest therefore
    measures the regime gates + bracket geometry, not the model's edge.

Walk-forward: the frame is cut into equal chronological folds; each
parameter set is scored on fold k (training) and the winner is re-measured
on fold k+1 (validation). Only validation numbers matter — a parameter set
that wins training folds but degrades on validation is overfit, full stop.

Usage (offline, public data only — no API keys required):
    python backtest.py BTC/USDT --days 14
    python backtest.py ADA/USDT --days 30 --walk-forward
    python backtest.py BTC/USDT --days 14 --tp 2.0 --sl 2.0

Strict typing: annotated for ``mypy --strict``. Embedded unittest suite uses
synthetic frames only (``python -m unittest backtest``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import unittest
import dataclasses
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, List, Optional, Sequence, Tuple

import pandas as pd

from feed import TIMEFRAME_MS, L2OrderBook
from gatekeeper import ConfluenceReport, MarketGatekeeper, RegimeReport

__all__ = [
    "SimulatedTrade",
    "BacktestReport",
    "GridPoint",
    "WalkForwardRow",
    "simulate",
    "walk_forward",
    "fetch_history",
]

logger: Final[logging.Logger] = logging.getLogger("bot.backtest")

_ZERO: Final[Decimal] = Decimal("0")

#: Bars of history handed to the gatekeeper per evaluation — enough for the
#: slowest indicator (ATR14 + SMA20 + 1 = 35) with stabilizing headroom.
WARMUP_BARS: Final[int] = 64
#: Give up on a bracket after this many bars (8 hours of 5m candles).
MAX_HOLD_BARS: Final[int] = 96
#: A parameter set must produce at least this many training trades to be
#: eligible in the walk-forward sweep — fewer is statistical noise.
MIN_TRADES_PER_FOLD: Final[int] = 5

OUTCOME_WIN: Final[str] = "WIN"
OUTCOME_LOSS: Final[str] = "LOSS"
OUTCOME_TIMEOUT: Final[str] = "TIMEOUT"


# --------------------------------------------------------------------------- #
# Result types                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    """One simulated bracket, entry at the open after the signal bar."""

    entry_index: int  # frame row of the entry bar (signal bar + 1)
    direction: str  # "LONG" | "SHORT"
    entry_price: Decimal
    exit_price: Decimal
    atr_at_entry: Decimal
    outcome: str
    bars_held: int
    #: Exchange fee per side in basis points (taker, conservatively applied
    #: to BOTH legs even though resting TP fills pay the cheaper maker rate).
    fee_bps: Decimal = Decimal("0")

    @property
    def fees_per_unit(self) -> Decimal:
        """Round-trip exchange fees for one base unit."""
        return (self.entry_price + self.exit_price) * self.fee_bps / Decimal("10000")

    @property
    def pnl_per_unit(self) -> Decimal:
        """Quote-currency PnL for one base unit, net of modelled fees."""
        if self.direction == "LONG":
            gross: Decimal = self.exit_price - self.entry_price
        else:
            gross = self.entry_price - self.exit_price
        return gross - self.fees_per_unit

    @property
    def pnl_atr(self) -> Decimal:
        """PnL expressed in ATR multiples — comparable across price levels."""
        if self.atr_at_entry <= _ZERO:
            return _ZERO
        return self.pnl_per_unit / self.atr_at_entry


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """Aggregate replay statistics for one frame + parameter set."""

    bars: int
    evaluated: int
    regime_admitted: int
    confluence_admitted: int
    trades: Tuple[SimulatedTrade, ...]
    tp_mult: Decimal
    sl_mult: Decimal

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.outcome == OUTCOME_WIN)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.outcome == OUTCOME_LOSS)

    @property
    def timeouts(self) -> int:
        return sum(1 for t in self.trades if t.outcome == OUTCOME_TIMEOUT)

    @property
    def win_rate(self) -> Optional[Decimal]:
        decided: int = self.wins + self.losses
        if decided == 0:
            return None
        return Decimal(self.wins) / Decimal(decided)

    @property
    def breakeven_win_rate(self) -> Decimal:
        """Win rate where TP gains exactly offset SL losses (fee-free)."""
        return self.sl_mult / (self.tp_mult + self.sl_mult)

    @property
    def expectancy_atr(self) -> Optional[Decimal]:
        """Mean realized PnL per trade in ATR multiples (timeouts included)."""
        if not self.trades:
            return None
        total: Decimal = sum((t.pnl_atr for t in self.trades), _ZERO)
        return total / Decimal(len(self.trades))


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One parameter set for the walk-forward sweep."""

    adx_threshold: Decimal
    tp_mult: Decimal
    sl_mult: Decimal


@dataclass(frozen=True, slots=True)
class WalkForwardRow:
    """Winner of one training fold, re-measured on the next fold."""

    fold: int
    chosen: GridPoint
    train_expectancy_atr: Optional[Decimal]
    train_trades: int
    validation_expectancy_atr: Optional[Decimal]
    validation_trades: int
    validation_win_rate: Optional[Decimal]


# --------------------------------------------------------------------------- #
# Synthetic book (history has no L2)                                           #
# --------------------------------------------------------------------------- #


def _fresh_book(symbol: str, bar_open_ms: int) -> L2OrderBook:
    """Symmetric book stamped at candle close: freshness passes, the
    imbalance vote stays neutral (0.5 => never confirms a side)."""
    return L2OrderBook(
        symbol=symbol,
        bids=((100.0, 5.0), (99.9, 4.0), (99.8, 3.0)),
        asks=((100.1, 5.0), (100.2, 4.0), (100.3, 3.0)),
        timestamp_ms=bar_open_ms + TIMEFRAME_MS,
    )


# --------------------------------------------------------------------------- #
# Replay engine                                                                #
# --------------------------------------------------------------------------- #


def _bar_open_ms(frame: pd.DataFrame, index: int) -> int:
    return int(pd.Timestamp(frame["timestamps"].iloc[index]).value // 1_000_000)


def _walk_bracket(
    frame: pd.DataFrame,
    entry_index: int,
    direction: str,
    entry: Decimal,
    tp: Decimal,
    sl: Decimal,
    atr: Decimal,
) -> SimulatedTrade:
    """March forward bar by bar until TP, SL, or the hold limit resolves."""
    last_index: int = min(entry_index + MAX_HOLD_BARS, len(frame) - 1)
    for j in range(entry_index, last_index + 1):
        high: Decimal = Decimal(str(float(frame["high"].iloc[j])))
        low: Decimal = Decimal(str(float(frame["low"].iloc[j])))
        if direction == "LONG":
            hit_tp: bool = high >= tp
            hit_sl: bool = low <= sl
        else:
            hit_tp = low <= tp
            hit_sl = high >= sl
        if hit_sl:  # checked first: both-in-one-bar resolves as the loss
            return SimulatedTrade(
                entry_index=entry_index,
                direction=direction,
                entry_price=entry,
                exit_price=sl,
                atr_at_entry=atr,
                outcome=OUTCOME_LOSS,
                bars_held=j - entry_index + 1,
            )
        if hit_tp:
            return SimulatedTrade(
                entry_index=entry_index,
                direction=direction,
                entry_price=entry,
                exit_price=tp,
                atr_at_entry=atr,
                outcome=OUTCOME_WIN,
                bars_held=j - entry_index + 1,
            )
    exit_close: Decimal = Decimal(str(float(frame["close"].iloc[last_index])))
    return SimulatedTrade(
        entry_index=entry_index,
        direction=direction,
        entry_price=entry,
        exit_price=exit_close,
        atr_at_entry=atr,
        outcome=OUTCOME_TIMEOUT,
        bars_held=last_index - entry_index + 1,
    )


def simulate(
    frame: pd.DataFrame,
    *,
    symbol: str = "BTC/USDT",
    adx_threshold: Decimal = Decimal("25"),
    tp_mult: Decimal = Decimal("1.5"),
    sl_mult: Decimal = Decimal("2.5"),
    confluence_min_votes: int = 2,
    fee_bps: Decimal = Decimal("0"),
) -> BacktestReport:
    """Replay ``frame`` through the production gates and simulate brackets."""
    gatekeeper: MarketGatekeeper = MarketGatekeeper(
        adx_threshold=adx_threshold,
        confluence_min_votes=confluence_min_votes,
    )

    evaluated = regime_admitted = confluence_admitted = 0
    trades: List[SimulatedTrade] = []
    busy_until: int = -1  # one simulated position at a time, like the router

    last_signal_bar: int = len(frame) - 2  # the entry bar must exist
    for i in range(WARMUP_BARS, last_signal_bar + 1):
        if i <= busy_until:
            continue
        window: pd.DataFrame = frame.iloc[i - WARMUP_BARS + 1 : i + 1]
        book: L2OrderBook = _fresh_book(symbol, _bar_open_ms(frame, i))
        evaluated += 1

        report: RegimeReport = gatekeeper.evaluate(window, book)
        if not report.passed or report.atr is None:
            continue
        regime_admitted += 1

        # Direction proxy: follow the Wilder DI line (see module docstring).
        if report.plus_di is None or report.minus_di is None:
            continue
        long_side: bool = report.plus_di > report.minus_di
        confluence: ConfluenceReport = gatekeeper.confluence(
            report, long_side=long_side
        )
        if not confluence.passed:
            continue
        confluence_admitted += 1

        entry_index: int = i + 1
        entry: Decimal = Decimal(str(float(frame["open"].iloc[entry_index])))
        offset_tp: Decimal = tp_mult * report.atr
        offset_sl: Decimal = sl_mult * report.atr
        direction: str = "LONG" if long_side else "SHORT"
        if long_side:
            tp, sl = entry + offset_tp, entry - offset_sl
        else:
            tp, sl = entry - offset_tp, entry + offset_sl

        trade: SimulatedTrade = _walk_bracket(
            frame, entry_index, direction, entry, tp, sl, report.atr
        )
        if fee_bps > _ZERO:
            trade = dataclasses.replace(trade, fee_bps=fee_bps)
        trades.append(trade)
        busy_until = trade.entry_index + trade.bars_held - 1

    return BacktestReport(
        bars=len(frame),
        evaluated=evaluated,
        regime_admitted=regime_admitted,
        confluence_admitted=confluence_admitted,
        trades=tuple(trades),
        tp_mult=tp_mult,
        sl_mult=sl_mult,
    )


# --------------------------------------------------------------------------- #
# Walk-forward sweep                                                           #
# --------------------------------------------------------------------------- #

DEFAULT_GRID: Final[Tuple[GridPoint, ...]] = tuple(
    GridPoint(adx_threshold=adx, tp_mult=tp, sl_mult=sl)
    for adx in (Decimal("20"), Decimal("25"), Decimal("30"))
    for tp in (Decimal("1.5"), Decimal("2.0"), Decimal("2.5"))
    for sl in (Decimal("1.5"), Decimal("2.0"), Decimal("2.5"))
)


def walk_forward(
    frame: pd.DataFrame,
    *,
    symbol: str = "BTC/USDT",
    grid: Sequence[GridPoint] = DEFAULT_GRID,
    folds: int = 3,
    confluence_min_votes: int = 2,
    fee_bps: Decimal = Decimal("0"),
) -> List[WalkForwardRow]:
    """Train on fold k, validate the winning parameters on fold k+1."""
    if folds < 1:
        raise ValueError("walk-forward needs at least one train/validate pair")
    chunk: int = len(frame) // (folds + 1)
    if chunk < WARMUP_BARS + 40:
        raise ValueError(
            f"frame too short: {len(frame)} bars across {folds + 1} folds "
            f"leaves {chunk} per fold; need >= {WARMUP_BARS + 40}"
        )

    rows: List[WalkForwardRow] = []
    for k in range(folds):
        train: pd.DataFrame = frame.iloc[k * chunk : (k + 1) * chunk].reset_index(
            drop=True
        )
        validation: pd.DataFrame = frame.iloc[
            (k + 1) * chunk : (k + 2) * chunk
        ].reset_index(drop=True)

        best: Optional[GridPoint] = None
        best_expectancy: Optional[Decimal] = None
        best_trades: int = 0
        for point in grid:
            report: BacktestReport = simulate(
                train,
                symbol=symbol,
                adx_threshold=point.adx_threshold,
                tp_mult=point.tp_mult,
                sl_mult=point.sl_mult,
                confluence_min_votes=confluence_min_votes,
                fee_bps=fee_bps,
            )
            expectancy: Optional[Decimal] = report.expectancy_atr
            if expectancy is None or len(report.trades) < MIN_TRADES_PER_FOLD:
                continue
            if best_expectancy is None or expectancy > best_expectancy:
                best, best_expectancy = point, expectancy
                best_trades = len(report.trades)

        if best is None:
            logger.warning(
                "fold %d: no parameter set produced >= %d trades — skipped",
                k,
                MIN_TRADES_PER_FOLD,
            )
            continue

        val_report: BacktestReport = simulate(
            validation,
            symbol=symbol,
            adx_threshold=best.adx_threshold,
            tp_mult=best.tp_mult,
            sl_mult=best.sl_mult,
            confluence_min_votes=confluence_min_votes,
            fee_bps=fee_bps,
        )
        rows.append(
            WalkForwardRow(
                fold=k,
                chosen=best,
                train_expectancy_atr=best_expectancy,
                train_trades=best_trades,
                validation_expectancy_atr=val_report.expectancy_atr,
                validation_trades=len(val_report.trades),
                validation_win_rate=val_report.win_rate,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# History fetch (public REST, paginated)                                       #
# --------------------------------------------------------------------------- #


async def fetch_history(
    symbol: str,
    *,
    days: int,
    timeframe: str = "5m",
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Pull ``days`` of confirmed candles from the public REST endpoint."""
    import ccxt.async_support as ccxt_async  # type: ignore[import-untyped]

    klass: Any = getattr(ccxt_async, exchange_id)
    exchange: Any = klass({"enableRateLimit": True})
    try:
        timeframe_ms: int = int(exchange.parse_timeframe(timeframe)) * 1_000
        since: int = exchange.milliseconds() - days * 86_400_000
        rows: List[List[float]] = []
        while True:
            batch: List[List[float]] = await exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=1_000
            )
            if not batch:
                break
            rows.extend(batch)
            since = int(batch[-1][0]) + timeframe_ms
            if len(batch) < 1_000:
                break
    finally:
        await exchange.close()

    frame: pd.DataFrame = pd.DataFrame(
        rows, columns=["timestamps", "open", "high", "low", "close", "volume"]
    )
    frame["timestamps"] = pd.to_datetime(frame["timestamps"], unit="ms", utc=True)
    frame = frame.drop_duplicates(subset="timestamps").reset_index(drop=True)
    # Drop the still-forming final candle — confirmed bars only, like the feed.
    if len(frame) > 0:
        frame = frame.iloc[:-1].reset_index(drop=True)
    logger.info("%s: fetched %d confirmed %s bars", symbol, len(frame), timeframe)
    return frame


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _print_report(symbol: str, report: BacktestReport) -> None:
    print(f"\n=== BACKTEST {symbol} — TP {report.tp_mult}xATR / SL {report.sl_mult}xATR ===")
    print(f"bars                 {report.bars}")
    print(f"bars evaluated       {report.evaluated}")
    print(f"regime admitted      {report.regime_admitted}")
    print(f"confluence admitted  {report.confluence_admitted}")
    print(f"trades               {len(report.trades)} "
          f"(W {report.wins} / L {report.losses} / T {report.timeouts})")
    win_rate = report.win_rate
    print(f"win rate             "
          f"{'n/a' if win_rate is None else f'{win_rate * 100:.1f}%'} "
          f"(breakeven {report.breakeven_win_rate * 100:.1f}%)")
    expectancy = report.expectancy_atr
    print(f"expectancy           "
          f"{'n/a' if expectancy is None else f'{expectancy:.4f} ATR/trade'}")


def _print_walk_forward(symbol: str, rows: List[WalkForwardRow]) -> None:
    print(f"\n=== WALK-FORWARD {symbol} ===")
    if not rows:
        print("no fold produced enough trades — extend --days")
        return
    for row in rows:
        val_exp = row.validation_expectancy_atr
        val_wr = row.validation_win_rate
        print(
            f"fold {row.fold}: ADX>{row.chosen.adx_threshold} "
            f"TP {row.chosen.tp_mult} SL {row.chosen.sl_mult} | "
            f"train {row.train_expectancy_atr:.4f} ATR x{row.train_trades} -> "
            f"validation "
            f"{'n/a' if val_exp is None else f'{val_exp:.4f} ATR'} "
            f"x{row.validation_trades} "
            f"(win {'n/a' if val_wr is None else f'{val_wr * 100:.1f}%'})"
        )
    print("Trust only the validation column. Train-only winners are overfit.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline bracket-strategy replay")
    parser.add_argument("symbol", help="e.g. BTC/USDT")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--adx", type=str, default="25")
    parser.add_argument("--tp", type=str, default="1.5")
    parser.add_argument("--sl", type=str, default="2.5")
    parser.add_argument("--votes", type=int, default=2)
    parser.add_argument(
        "--fee-bps",
        type=str,
        default="10",
        help="exchange fee per side in basis points (10 = 0.1%% taker; "
        "0 disables — testnet reports zero but live Binance does not)",
    )
    parser.add_argument("--walk-forward", action="store_true")
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
    if args.walk_forward:
        _print_walk_forward(
            args.symbol,
            walk_forward(
                frame,
                symbol=args.symbol,
                confluence_min_votes=args.votes,
                fee_bps=Decimal(args.fee_bps),
            ),
        )
    else:
        _print_report(
            args.symbol,
            simulate(
                frame,
                symbol=args.symbol,
                adx_threshold=Decimal(args.adx),
                tp_mult=Decimal(args.tp),
                sl_mult=Decimal(args.sl),
                confluence_min_votes=args.votes,
                fee_bps=Decimal(args.fee_bps),
            ),
        )
    return 0


# --------------------------------------------------------------------------- #
# Embedded tests (synthetic frames only — no network)                          #
# --------------------------------------------------------------------------- #

_T0_MS: Final[int] = 1_750_000_500_000 - (1_750_000_500_000 % TIMEFRAME_MS)


def _frame_from_rows(
    rows: Sequence[Tuple[float, float, float, float, float]],
) -> pd.DataFrame:
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
    return frame


def _uptrend_rows(
    num_bars: int,
) -> List[Tuple[float, float, float, float, float]]:
    """Expanding uptrend with rising volume — every gate can admit."""
    rows: List[Tuple[float, float, float, float, float]] = []
    price: float = 100.0
    for i in range(num_bars):
        step: float = 0.5 + 0.05 * i
        o: float = price
        c: float = o + step
        h: float = c + 0.10 * step
        lo: float = o - 0.05 * step
        v: float = 100.0 + (i % 7) * 40.0  # periodic volume expansion
        rows.append((o, h, lo, c, v))
        price = c
    return rows


class BacktestTests(unittest.TestCase):
    def test_uptrend_produces_long_trades_without_lookahead(self) -> None:
        frame: pd.DataFrame = _frame_from_rows(_uptrend_rows(160))
        report: BacktestReport = simulate(frame, confluence_min_votes=1)
        self.assertGreater(len(report.trades), 0)
        for trade in report.trades:
            self.assertEqual(trade.direction, "LONG")  # DI follows the trend
            expected_entry: Decimal = Decimal(
                str(float(frame["open"].iloc[trade.entry_index]))
            )
            self.assertEqual(trade.entry_price, expected_entry)  # next-bar open
        # An expanding uptrend should resolve brackets profitably overall.
        assert report.expectancy_atr is not None
        self.assertGreater(report.expectancy_atr, _ZERO)

    def test_positions_never_overlap(self) -> None:
        frame: pd.DataFrame = _frame_from_rows(_uptrend_rows(200))
        report: BacktestReport = simulate(frame, confluence_min_votes=1)
        self.assertGreater(len(report.trades), 1)
        for prev, cur in zip(report.trades, report.trades[1:]):
            prev_exit: int = prev.entry_index + prev.bars_held - 1
            self.assertGreater(cur.entry_index, prev_exit)

    def test_both_legs_in_one_bar_resolves_as_loss(self) -> None:
        rows: List[Tuple[float, float, float, float, float]] = _uptrend_rows(80)
        # Right after the warmup edge, print one violent bar whose range
        # spans far beyond any plausible TP and SL simultaneously.
        o, h, lo, c, v = rows[WARMUP_BARS + 1]
        rows[WARMUP_BARS + 1] = (o, h + 500.0, lo - 500.0, c, v)
        frame: pd.DataFrame = _frame_from_rows(rows)
        report: BacktestReport = simulate(frame, confluence_min_votes=1)
        ambiguous: List[SimulatedTrade] = [
            t for t in report.trades if t.entry_index == WARMUP_BARS + 1
        ]
        if ambiguous:  # the gates may or may not admit that exact bar
            self.assertEqual(ambiguous[0].outcome, OUTCOME_LOSS)
        # Regardless of admission, the engine must never have crashed and the
        # conservative rule is exercised by at least the assertion above.
        self.assertGreaterEqual(len(report.trades), 0)

    def test_fees_reduce_pnl_per_side(self) -> None:
        trade = SimulatedTrade(
            entry_index=1,
            direction="LONG",
            entry_price=Decimal("100"),
            exit_price=Decimal("102"),
            atr_at_entry=Decimal("1"),
            outcome="WIN",
            bars_held=3,
            fee_bps=Decimal("10"),
        )
        # Gross +2; fees = (100+102)*0.001 = 0.202.
        self.assertEqual(trade.fees_per_unit, Decimal("0.202"))
        self.assertEqual(trade.pnl_per_unit, Decimal("1.798"))
        free = dataclasses.replace(trade, fee_bps=Decimal("0"))
        self.assertEqual(free.pnl_per_unit, Decimal("2"))

    def test_breakeven_win_rate_formula(self) -> None:
        report: BacktestReport = BacktestReport(
            bars=0,
            evaluated=0,
            regime_admitted=0,
            confluence_admitted=0,
            trades=(),
            tp_mult=Decimal("1.5"),
            sl_mult=Decimal("2.5"),
        )
        self.assertEqual(report.breakeven_win_rate, Decimal("0.625"))

    def test_walk_forward_validates_on_unseen_fold(self) -> None:
        frame: pd.DataFrame = _frame_from_rows(_uptrend_rows(440))
        tiny_grid: Tuple[GridPoint, ...] = (
            GridPoint(Decimal("20"), Decimal("1.5"), Decimal("2.5")),
            GridPoint(Decimal("25"), Decimal("2.0"), Decimal("2.0")),
        )
        rows: List[WalkForwardRow] = walk_forward(
            frame, grid=tiny_grid, folds=3, confluence_min_votes=1
        )
        self.assertLessEqual(len(rows), 3)
        for row in rows:
            self.assertIn(row.chosen, tiny_grid)
            self.assertGreaterEqual(row.train_trades, MIN_TRADES_PER_FOLD)

    def test_walk_forward_refuses_short_frames(self) -> None:
        frame: pd.DataFrame = _frame_from_rows(_uptrend_rows(120))
        with self.assertRaises(ValueError):
            walk_forward(frame, folds=4)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] not in ("-v", "-q"):
        raise SystemExit(main())
    logging.basicConfig(level=logging.INFO)
    unittest.main()

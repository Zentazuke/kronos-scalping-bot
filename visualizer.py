"""visualizer.py — Phase 6: Mission Control terminal dashboard.

Real-time ``rich``-powered TUI rendering the bot's operational state in four
border-drawn quadrants: header (equity + 3% drawdown bar), gatekeeper sieves
(left), Kronos Monte Carlo histogram (right), and the execution ledger
(bottom). Strictly an *observer*: it consumes report objects the pipeline
already produces and can never influence — or block — a trading decision.

Non-blocking contract:
  * Producers call ``visualizer.publish(event)`` (a drop-oldest, never-await
    wrapper around ``queue.put_nowait``) — O(1), no backpressure onto the
    feed or execution loops.
  * The visualizer itself runs as one background task created with
    ``asyncio.create_task(visualizer.run())`` and repaints at a fixed
    cadence (default 500 ms), draining the queue between frames.

INTEGRATION SEAM — exact wiring inside main.py's TradingSupervisor:

    # __init__ (after the router is constructed):
    #     self._visualizer = TradingBotVisualizer(
    #         symbols=self._symbols, exchange_label="Binance Sandbox"
    #     )
    #
    # run() — right after `await self._feed.start()`:
    #     viz_task = asyncio.create_task(self._visualizer.run(), name="visualizer")
    #   and in the `finally` block, next to the shutdown watcher teardown:
    #     self._visualizer.stop()
    #     await asyncio.gather(viz_task, return_exceptions=True)
    #
    # _drawdown_check() — right after `drawdown` is computed:
    #     self._visualizer.publish(EquityUpdate(
    #         equity=current, baseline=baseline,
    #         drawdown_limit=self._drawdown_limit, killed=self._killed,
    #     ))
    #
    # _handle_event() Step B — switch to the report API for full telemetry:
    #     report = self._gatekeeper.evaluate(frame, book)
    #     self._visualizer.publish(RegimeUpdate(symbol=symbol, report=report))
    #     if not report.passed:
    #         ...peaceful skip as before...
    #
    # _handle_event() Step C — publish the full Monte Carlo report:
    #     pred_report = await self._engine.evaluate(symbol, frame)  # in try/except
    #     self._visualizer.publish(InferenceUpdate(symbol=symbol, report=pred_report))
    #
    # _handle_event() Step D — right after `result = await self._router.route_trade(...)`:
    #     self._visualizer.publish(ExecutionUpdate(result=result))

Strict typing: annotated for ``mypy --strict`` (rich ships py.typed).
Standalone visual smoke test: ``python visualizer.py`` renders the dashboard
with synthetic data so the layout can be verified without the live bot.
"""

from __future__ import annotations

import asyncio
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Deque, Dict, Final, List, Optional, Sequence, Tuple, Union

from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from execution import ExecutionResult, ExecutionStatus
from gatekeeper import (
    ADX_THRESHOLD,
    BOOK_MAX_AGE_MS,
    ConfluenceReport,
    RegimeReport,
)
from predictor import (
    DEAD_BAND_HIGH,
    DEAD_BAND_LOW,
    EDGE_THRESHOLD,
    PredictionReport,
    SignalDirection,
)

__all__ = [
    "TradingBotVisualizer",
    "EquityUpdate",
    "RegimeUpdate",
    "InferenceUpdate",
    "ConfluenceUpdate",
    "ExecutionUpdate",
    "PerformanceUpdate",
    "LedgerLine",
    "VisualizerEvent",
]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

REFRESH_INTERVAL_S: Final[float] = 0.5
QUEUE_MAXSIZE: Final[int] = 512
LEDGER_LINES: Final[int] = 12
DRAIN_BUDGET_PER_FRAME: Final[int] = 256
DRAWDOWN_BAR_WIDTH: Final[int] = 44
PATH_BAR_WIDTH: Final[int] = 30

_ZERO: Final[Decimal] = Decimal("0")
_HUNDRED: Final[Decimal] = Decimal("100")

# --------------------------------------------------------------------------- #
# Event types                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EquityUpdate:
    """Wallet equity snapshot vs the daily baseline (Step A telemetry)."""

    equity: Decimal
    baseline: Decimal
    drawdown_limit: Decimal
    killed: bool = False

    @property
    def drawdown(self) -> Decimal:
        if self.baseline <= _ZERO:
            return _ZERO
        return (self.baseline - self.equity) / self.baseline


@dataclass(frozen=True, slots=True)
class RegimeUpdate:
    """Latest gatekeeper sieve evaluation for one symbol (Step B).

    ``close_price`` is the confirmed bar close — the dashboard's mark price
    for live bracket distance readouts.
    """

    symbol: str
    report: RegimeReport
    close_price: Optional[Decimal] = None


@dataclass(frozen=True, slots=True)
class InferenceUpdate:
    """Latest Kronos Monte Carlo round for one symbol (Step C)."""

    symbol: str
    report: PredictionReport


@dataclass(frozen=True, slots=True)
class ConfluenceUpdate:
    """Directional confirmation votes for one proposed trade (Step C½)."""

    symbol: str
    report: ConfluenceReport


@dataclass(frozen=True, slots=True)
class ExecutionUpdate:
    """One routing outcome from the ExecutionRouter (Step D)."""

    result: ExecutionResult


@dataclass(frozen=True, slots=True)
class PerformanceUpdate:
    """Realized system performance from the trade journal (Phase 7)."""

    wins: int
    losses: int
    scratches: int
    open_trades: int
    realized_pnl: Decimal
    win_rate: Optional[Decimal] = None
    kelly_fraction: Optional[Decimal] = None


@dataclass(frozen=True, slots=True)
class LedgerLine:
    """Free-form operator note appended to the execution ledger."""

    message: str
    style: str = "white"


VisualizerEvent = Union[
    EquityUpdate,
    RegimeUpdate,
    InferenceUpdate,
    ConfluenceUpdate,
    ExecutionUpdate,
    PerformanceUpdate,
    LedgerLine,
]

# --------------------------------------------------------------------------- #
# Formatting helpers                                                           #
# --------------------------------------------------------------------------- #


def _fmt(value: Optional[Decimal], places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _pct(value: Optional[Decimal], places: int = 4) -> str:
    return "—" if value is None else f"{value * _HUNDRED:.{places}f}%"


def _utc_clock() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# --------------------------------------------------------------------------- #
# Visualizer                                                                   #
# --------------------------------------------------------------------------- #


class TradingBotVisualizer:
    """Mission Control: four-quadrant live dashboard over an asyncio queue."""

    def __init__(
        self,
        *,
        symbols: Sequence[str],
        exchange_label: str = "Binance Sandbox",
        refresh_interval_s: float = REFRESH_INTERVAL_S,
        ledger_lines: int = LEDGER_LINES,
        queue_maxsize: int = QUEUE_MAXSIZE,
        console: Optional[Console] = None,
    ) -> None:
        if not symbols:
            raise ValueError("TradingBotVisualizer requires at least one symbol")
        self._symbols: Tuple[str, ...] = tuple(symbols)
        self._exchange_label: str = exchange_label
        self._refresh_interval_s: float = refresh_interval_s
        self._console: Console = console or Console()

        #: Producer-facing ingestion queue (main.py publishes into this).
        self.queue: "asyncio.Queue[VisualizerEvent]" = asyncio.Queue(
            maxsize=queue_maxsize
        )

        # Glyph capability probe: legacy Windows consoles (cp1252) cannot
        # encode block-drawing characters and rich's Win32 renderer raises on
        # them — degrade to ASCII bars there instead of crashing the cockpit.
        encoding: str = getattr(self._console.file, "encoding", None) or "utf-8"
        try:
            "█─·".encode(encoding)
            self._glyph_fill: str = "█"
            self._glyph_track: str = "─"
            self._glyph_dot: str = "·"
        except (UnicodeEncodeError, LookupError):
            self._glyph_fill = "#"
            self._glyph_track = "-"
            self._glyph_dot = "."
        try:
            "✓✗".encode(encoding)
            self._glyph_yes: str = "✓"
            self._glyph_no: str = "✗"
        except (UnicodeEncodeError, LookupError):
            self._glyph_yes = "+"
            self._glyph_no = "x"

        self._stop_event: asyncio.Event = asyncio.Event()
        self._equity: Optional[EquityUpdate] = None
        self._performance: Optional[PerformanceUpdate] = None
        self._regimes: Dict[str, RegimeReport] = {}
        self._inferences: Dict[str, PredictionReport] = {}
        self._confluences: Dict[str, ConfluenceReport] = {}
        self._last_price: Dict[str, Decimal] = {}
        #: Last EXECUTED bracket per symbol, with its placement timestamp.
        self._brackets: Dict[str, Tuple[ExecutionResult, str]] = {}
        #: Session outcome tally: brackets placed, aborts by type, vetoes.
        self._stats: Dict[str, int] = {}
        self._ledger: Deque[Text] = deque(maxlen=ledger_lines)
        self._ledger.append(
            Text(f"{_stamp()}  ledger online — awaiting first bar close", style="dim")
        )

    # ------------------------------------------------------------------ #
    # Producer interface (never blocks, never raises)                     #
    # ------------------------------------------------------------------ #

    def publish(self, event: VisualizerEvent) -> None:
        """Drop-oldest enqueue: telemetry loss is always preferable to
        backpressure on the trading pipeline."""
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(event)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """Background repaint loop — start with ``asyncio.create_task``."""
        with Live(
            self._render(), console=self._console, refresh_per_second=4
        ) as live:
            try:
                while not self._stop_event.is_set():
                    self._drain()
                    live.update(self._render())
                    await asyncio.sleep(self._refresh_interval_s)
                self._drain()
                live.update(self._render())  # final frame on clean stop
            except asyncio.CancelledError:
                live.update(self._render())
                raise

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def _drain(self) -> None:
        for _ in range(DRAIN_BUDGET_PER_FRAME):
            try:
                event: VisualizerEvent = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._apply(event)

    def _apply(self, event: VisualizerEvent) -> None:
        if isinstance(event, EquityUpdate):
            self._equity = event
        elif isinstance(event, PerformanceUpdate):
            self._performance = event
        elif isinstance(event, RegimeUpdate):
            self._regimes[event.symbol] = event.report
            if event.close_price is not None:
                self._last_price[event.symbol] = event.close_price
        elif isinstance(event, InferenceUpdate):
            self._inferences[event.symbol] = event.report
        elif isinstance(event, ConfluenceUpdate):
            self._apply_confluence(event)
        elif isinstance(event, ExecutionUpdate):
            self._apply_execution(event.result)
        else:
            self._ledger.append(
                Text(f"{_stamp()}  {event.message}", style=event.style)
            )

    def _apply_confluence(self, event: ConfluenceUpdate) -> None:
        report: ConfluenceReport = event.report
        self._confluences[event.symbol] = report
        if not report.passed:
            self._stats["CONFLUENCE VETO"] = self._stats.get("CONFLUENCE VETO", 0) + 1
            side: str = "LONG" if report.long_side else "SHORT"
            self._ledger.append(
                Text(
                    f"{_stamp()}  {event.symbol} CONFLUENCE VETO {side} — "
                    f"{report.votes}/{report.required} votes "
                    f"(DI {self._vote_glyph(report.di_vote)} "
                    f"RSI {self._vote_glyph(report.rsi_vote)} "
                    f"BOOK {self._vote_glyph(report.book_vote)})",
                    style="yellow",
                )
            )

    def _vote_glyph(self, vote: bool) -> str:
        return self._glyph_yes if vote else self._glyph_no

    def _apply_execution(self, result: ExecutionResult) -> None:
        stamp: str = _stamp()
        self._stats[result.status.value] = self._stats.get(result.status.value, 0) + 1
        if result.status is ExecutionStatus.EXECUTED:
            self._brackets[result.symbol] = (result, stamp)
            self._ledger.append(
                Text(
                    f"{stamp}  {result.symbol} BRACKET LIVE {result.direction.value} "
                    f"{_fmt(result.executed_amount, 6)} @ {_fmt(result.entry_fill_price, 2)} "
                    f"| TP {_fmt(result.take_profit_price, 2)} "
                    f"SL {_fmt(result.stop_loss_price, 2)} "
                    f"| slip {_pct(result.deviation)}",
                    style="bold green",
                )
            )
        elif result.status is ExecutionStatus.ABORT_SLIPPAGE:
            self._ledger.append(
                Text(
                    f"{stamp}  {result.symbol} SLIPPAGE ABORT — deviation "
                    f"{_pct(result.deviation)} breached 0.0500% barrier",
                    style="bold red",
                )
            )
        elif result.status in (
            ExecutionStatus.ABORT_BRACKET_FAILED,
            ExecutionStatus.ABORT_ENTRY_TIMEOUT,
            ExecutionStatus.ERROR,
        ):
            self._brackets.pop(result.symbol, None)
            self._ledger.append(
                Text(f"{stamp}  {result.symbol} {result.status.value} — "
                     f"{result.reason}", style="bold white on red")
            )
        else:
            self._ledger.append(
                Text(
                    f"{stamp}  {result.symbol} {result.status.value} — {result.reason}",
                    style="yellow",
                )
            )

    # ------------------------------------------------------------------ #
    # Rendering — layout                                                  #
    # ------------------------------------------------------------------ #

    def _render(self) -> Layout:
        layout: Layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=9),
            Layout(name="middle", ratio=1),
            Layout(name="ledger", size=LEDGER_LINES + 5 + len(self._symbols)),
        )
        layout["middle"].split_row(
            Layout(name="gatekeeper", ratio=1),
            Layout(name="brain", ratio=1),
        )
        layout["header"].update(self._header_panel())
        layout["gatekeeper"].update(self._gatekeeper_panel())
        layout["brain"].update(self._brain_panel())
        layout["ledger"].update(self._ledger_panel())
        return layout

    # ------------------------------------------------------------------ #
    # Rendering — header                                                  #
    # ------------------------------------------------------------------ #

    def _header_panel(self) -> Panel:
        meta: Table = Table.grid(expand=True)
        meta.add_column(justify="left")
        meta.add_column(justify="center")
        meta.add_column(justify="right")
        meta.add_row(
            Text(_utc_clock(), style="bold white"),
            Text(self._exchange_label.upper(), style="bold yellow"),
            Text("  ".join(self._symbols), style="bold cyan"),
        )

        metrics: Table = Table.grid(expand=True)
        metrics.add_column(justify="left")
        metrics.add_column(justify="right")
        if self._equity is None:
            metrics.add_row(
                Text("WALLET EQUITY  —", style="bold"),
                Text("DAILY BASELINE  —", style="bold"),
            )
        else:
            metrics.add_row(
                Text(
                    f"WALLET EQUITY  {_fmt(self._equity.equity, 2)} USDT",
                    style="bold bright_white",
                ),
                Text(
                    f"DAILY BASELINE  {_fmt(self._equity.baseline, 2)} USDT",
                    style="bold bright_white",
                ),
            )

        return Panel(
            Group(meta, Text(), metrics, self._performance_line(), self._drawdown_bar()),
            title="[bold]MISSION CONTROL[/bold]",
            border_style="cyan",
        )

    def _performance_line(self) -> Text:
        perf: Optional[PerformanceUpdate] = self._performance
        if perf is None:
            return Text("SYSTEM    no realized trades yet — Kelly on priors", style="dim")
        pnl_style: str = "bold green" if perf.realized_pnl >= _ZERO else "bold red"
        line: Text = Text.assemble(
            ("SYSTEM    ", "bold"),
            (f"W/L {perf.wins}/{perf.losses}", "bold bright_white"),
            (f"  open {perf.open_trades}", "white"),
        )
        if perf.win_rate is not None:
            line.append(f"  win {_pct(perf.win_rate, 1)}", "bold bright_white")
        if perf.kelly_fraction is not None:
            line.append(f"  kelly {_pct(perf.kelly_fraction, 2)}", "cyan")
        sign: str = "+" if perf.realized_pnl >= _ZERO else ""
        line.append(f"  realized {sign}{_fmt(perf.realized_pnl, 2)} USDT", pnl_style)
        return line

    def _drawdown_bar(self) -> Text:
        if self._equity is None:
            return Text("DRAWDOWN  awaiting first equity snapshot", style="dim")

        drawdown: Decimal = self._equity.drawdown
        limit: Decimal = self._equity.drawdown_limit
        tripped: bool = self._equity.killed or (limit > _ZERO and drawdown >= limit)
        ratio: Decimal = (
            min(max(drawdown / limit, _ZERO), Decimal("1")) if limit > _ZERO else _ZERO
        )
        filled: int = int(ratio * DRAWDOWN_BAR_WIDTH)

        if tripped:
            style: str = "blink bold white on red"
        elif ratio >= Decimal("0.66"):
            style = "bold yellow"
        else:
            style = "bold green"

        bar: str = self._glyph_fill * filled + self._glyph_track * (
            DRAWDOWN_BAR_WIDTH - filled
        )
        label: str = (
            "KILL SWITCH TRIPPED"
            if tripped
            else f"{_pct(drawdown, 2)} of {_pct(limit, 2)} daily limit"
        )
        if drawdown < _ZERO:
            label = f"+{_pct(abs(drawdown), 2)} above baseline — {label}"
        return Text.assemble(
            ("DRAWDOWN  ", "bold"), (f"[{bar}] ", style), (label, style)
        )

    # ------------------------------------------------------------------ #
    # Rendering — gatekeeper quadrant                                     #
    # ------------------------------------------------------------------ #

    def _sieve_cell(self, passed: bool, detail: str) -> Text:
        if passed:
            return Text(f"PASS  {detail}", style="bold green")
        return Text(f"BLOCK {detail}", style="dim red")

    def _gatekeeper_panel(self) -> Panel:
        table: Table = Table(expand=True, border_style="grey50", pad_edge=False)
        table.add_column("SIEVE", style="bold", no_wrap=True)
        for symbol in self._symbols:
            table.add_column(symbol, justify="left")

        rows: List[Tuple[str, List[Text]]] = []
        adx_cells: List[Text] = []
        atr_cells: List[Text] = []
        vol_cells: List[Text] = []
        l2_cells: List[Text] = []
        for symbol in self._symbols:
            report: Optional[RegimeReport] = self._regimes.get(symbol)
            if report is None:
                waiting: Text = Text("awaiting bar", style="dim")
                adx_cells.append(waiting)
                atr_cells.append(Text("awaiting bar", style="dim"))
                vol_cells.append(Text("awaiting bar", style="dim"))
                l2_cells.append(Text("awaiting bar", style="dim"))
                continue
            if not report.sufficient_data:
                short: Text = Text("history too short", style="dim red")
                adx_cells.append(short)
                atr_cells.append(Text("history too short", style="dim red"))
                vol_cells.append(Text("history too short", style="dim red"))
                l2_cells.append(Text("history too short", style="dim red"))
                continue
            adx_cells.append(
                self._sieve_cell(
                    report.trend_ok, f"ADX {_fmt(report.adx, 1)} vs {ADX_THRESHOLD}"
                )
            )
            atr_cells.append(
                self._sieve_cell(
                    report.volatility_ok,
                    f"{_fmt(report.atr, 4)} vs {_fmt(report.atr_sma, 4)}",
                )
            )
            multiplier: Optional[Decimal] = None
            if (
                report.candle_volume is not None
                and report.average_volume is not None
                and report.average_volume > _ZERO
            ):
                multiplier = report.candle_volume / report.average_volume
            vol_cells.append(
                self._sieve_cell(report.volume_ok, f"{_fmt(multiplier, 2)}x avg")
            )
            age: str = (
                f"{report.book_age_ms} ms vs {BOOK_MAX_AGE_MS}"
                if report.book_age_ms is not None
                else "no book"
            )
            l2_cells.append(self._sieve_cell(report.book_fresh, age))

        rows.append(("TREND (ADX)", adx_cells))
        rows.append(("ATR EXPANSION", atr_cells))
        rows.append(("VOLUME MULT", vol_cells))
        rows.append(("L2 FRESHNESS", l2_cells))
        rows.append(("+DI / -DI", [self._di_cell(s) for s in self._symbols]))
        rows.append(("RSI(14)", [self._rsi_cell(s) for s in self._symbols]))
        rows.append(("BOOK BID%", [self._book_cell(s) for s in self._symbols]))
        for name, cells in rows:
            table.add_row(name, *cells)

        return Panel(
            table,
            title="[bold]GATEKEEPER — REGIME SIEVES + DIRECTION[/bold]",
            border_style="green",
        )

    def _di_cell(self, symbol: str) -> Text:
        report: Optional[RegimeReport] = self._regimes.get(symbol)
        if report is None or report.plus_di is None or report.minus_di is None:
            return Text("—", style="dim")
        bullish: bool = report.plus_di > report.minus_di
        return Text(
            f"{_fmt(report.plus_di, 1)} / {_fmt(report.minus_di, 1)}  "
            f"{'BULL' if bullish else 'BEAR'}",
            style="green" if bullish else "red",
        )

    def _rsi_cell(self, symbol: str) -> Text:
        report: Optional[RegimeReport] = self._regimes.get(symbol)
        if report is None or report.rsi is None:
            return Text("—", style="dim")
        rsi: Decimal = report.rsi
        if rsi >= Decimal("70"):
            return Text(f"{_fmt(rsi, 1)}  OVERBOUGHT", style="bold yellow")
        if rsi <= Decimal("30"):
            return Text(f"{_fmt(rsi, 1)}  OVERSOLD", style="bold yellow")
        return Text(f"{_fmt(rsi, 1)}  neutral zone", style="green")

    def _book_cell(self, symbol: str) -> Text:
        report: Optional[RegimeReport] = self._regimes.get(symbol)
        if report is None or report.book_imbalance is None:
            return Text("—", style="dim")
        bid_share: Decimal = report.book_imbalance * _HUNDRED
        if report.book_imbalance > Decimal("0.5"):
            return Text(f"{_fmt(bid_share, 1)}%  bid-heavy", style="green")
        if report.book_imbalance < Decimal("0.5"):
            return Text(f"{_fmt(bid_share, 1)}%  ask-heavy", style="red")
        return Text(f"{_fmt(bid_share, 1)}%  balanced", style="white")

    # ------------------------------------------------------------------ #
    # Rendering — inference quadrant                                      #
    # ------------------------------------------------------------------ #

    def _path_bar(self, label: str, count: int, total: int, style: str) -> Text:
        width: int = PATH_BAR_WIDTH
        filled: int = 0 if total <= 0 else round(count / total * width)
        bar: str = self._glyph_fill * filled + self._glyph_dot * (width - filled)
        return Text.assemble(
            (f"{label:<6}", "bold"),
            (bar, style),
            (f" {count:>2}/{total}", "bold"),
        )

    def _brain_panel(self) -> Panel:
        blocks: List[RenderableType] = []
        for symbol in self._symbols:
            report: Optional[PredictionReport] = self._inferences.get(symbol)
            if report is None:
                blocks.append(
                    Text(f"{symbol}   awaiting inference round", style="dim")
                )
                blocks.append(Text())
                continue
            signal_style: str = {
                SignalDirection.LONG: "bold green",
                SignalDirection.SHORT: "bold red",
                SignalDirection.NEUTRAL: "bold yellow",
            }[report.signal]
            blocks.append(
                Text.assemble(
                    (f"{symbol}  ", "bold cyan"),
                    (report.signal.value, signal_style),
                    (f"   anchor {_fmt(report.anchor_close, 2)}", "dim"),
                )
            )
            blocks.append(
                self._path_bar("LONG", report.paths_up, report.sample_count, "green")
            )
            blocks.append(
                self._path_bar("SHORT", report.paths_down, report.sample_count, "red")
            )
            blocks.append(
                self._path_bar(
                    "FLAT", report.paths_flat, report.sample_count, "grey50"
                )
            )
            blocks.append(
                Text(
                    f"p_up {_fmt(report.p_up)}  p_down {_fmt(report.p_down)}  "
                    f"| edge gate >= {EDGE_THRESHOLD}  "
                    f"| dead band ({DEAD_BAND_LOW}, {DEAD_BAND_HIGH})",
                    style="dim",
                )
            )
            blocks.append(self._confluence_line(symbol))
            blocks.append(Text())

        return Panel(
            Group(*blocks),
            title="[bold]KRONOS INFERENCE BRAIN — 30-PATH MONTE CARLO[/bold]",
            border_style="magenta",
        )

    def _confluence_line(self, symbol: str) -> Text:
        report: Optional[ConfluenceReport] = self._confluences.get(symbol)
        if report is None:
            return Text("confirm  awaiting directional signal", style="dim")
        side: str = "LONG" if report.long_side else "SHORT"
        verdict: str = "GO" if report.passed else "VETO"
        style: str = "bold green" if report.passed else "bold yellow"
        return Text(
            f"confirm {side}  "
            f"DI {self._vote_glyph(report.di_vote)}  "
            f"RSI {self._vote_glyph(report.rsi_vote)}  "
            f"BOOK {self._vote_glyph(report.book_vote)}  "
            f"— {report.votes}/{report.required} needed -> {verdict}",
            style=style,
        )

    # ------------------------------------------------------------------ #
    # Rendering — execution ledger                                        #
    # ------------------------------------------------------------------ #

    def _ledger_panel(self) -> Panel:
        exposure_rows: List[RenderableType] = []
        for symbol in self._symbols:
            exposure_rows.append(
                Text.assemble((f"{symbol:<10}  ", "bold cyan"))
                + self._exposure_row(symbol)
            )
        log_lines: List[RenderableType] = list(self._ledger)
        return Panel(
            Group(
                *exposure_rows,
                self._stats_line(),
                Rule(style="grey50", characters=self._glyph_track),
                *log_lines,
            ),
            title="[bold]EXECUTION LEDGER — TESTNET BRACKETS[/bold]",
            border_style="yellow",
        )

    def _exposure_row(self, symbol: str) -> Text:
        entry: Optional[Tuple[ExecutionResult, str]] = self._brackets.get(symbol)
        if entry is None:
            return Text("FLAT — no bracket this session", style="dim")
        result, placed_at = entry
        row: Text = Text.assemble(
            (f"{placed_at} ", "dim"),
            (f"{result.direction.value} ", "bold green"),
            (f"{_fmt(result.executed_amount, 6)} ", "green"),
            (f"@ {_fmt(result.entry_fill_price, 2)}  ", "green"),
            (f"TP {_fmt(result.take_profit_price, 2)}  ", "cyan"),
            (f"SL {_fmt(result.stop_loss_price, 2)}", "magenta"),
        )
        mark: Optional[Decimal] = self._last_price.get(symbol)
        fill: Optional[Decimal] = result.entry_fill_price
        if mark is None or fill is None or fill <= _ZERO or mark <= _ZERO:
            return row
        # Direction-signed move since entry, and distance left to each leg —
        # all against the latest confirmed bar close (the dashboard's mark).
        move: Decimal = (mark - fill) / fill
        if result.direction is SignalDirection.SHORT:
            move = -move
        move_style: str = "bold green" if move >= _ZERO else "bold red"
        row.append(f"  | mark {_fmt(mark, 2)} ", "white")
        row.append(f"{'+' if move >= _ZERO else ''}{_pct(move, 3)}", move_style)
        if result.take_profit_price is not None:
            tp_dist: Decimal = abs(result.take_profit_price - mark) / mark
            row.append(f"  TP {_pct(tp_dist, 3)} away", "cyan")
        if result.stop_loss_price is not None:
            sl_dist: Decimal = abs(result.stop_loss_price - mark) / mark
            row.append(f"  SL {_pct(sl_dist, 3)} away", "magenta")
        return row

    def _stats_line(self) -> Text:
        if not self._stats:
            return Text(
                "SESSION   no routing decisions yet", style="dim"
            )
        parts: List[str] = [
            f"{name} x{count}" for name, count in sorted(self._stats.items())
        ]
        return Text("SESSION   " + "   ".join(parts), style="bold white")


# --------------------------------------------------------------------------- #
# Standalone visual smoke test (synthetic data, no bot required)               #
# --------------------------------------------------------------------------- #


def _fake_regime(passed: bool) -> RegimeReport:
    return RegimeReport(
        sufficient_data=True,
        trend_ok=passed,
        volatility_ok=True,
        volume_ok=passed,
        book_fresh=True,
        adx=Decimal("31.4") if passed else Decimal("17.2"),
        atr=Decimal("42.1500"),
        atr_sma=Decimal("38.9100"),
        candle_volume=Decimal("182.4") if passed else Decimal("61.0"),
        average_volume=Decimal("97.5"),
        book_age_ms=random.randint(40, 700),
        plus_di=Decimal("28.3") if passed else Decimal("14.1"),
        minus_di=Decimal("11.7") if passed else Decimal("19.8"),
        rsi=Decimal("61.2") if passed else Decimal("74.9"),
        book_imbalance=Decimal("0.58") if passed else Decimal("0.41"),
    )


def _fake_confluence(passed: bool) -> ConfluenceReport:
    return ConfluenceReport(
        long_side=True,
        di_vote=True,
        rsi_vote=passed,
        book_vote=passed,
        required=2,
        plus_di=Decimal("28.3"),
        minus_di=Decimal("11.7"),
        rsi=Decimal("61.2") if passed else Decimal("74.9"),
        book_imbalance=Decimal("0.58") if passed else Decimal("0.41"),
    )


def _fake_inference(symbol: str, up: int) -> PredictionReport:
    down: int = 30 - up - 1
    p_up: Decimal = Decimal(up) / Decimal(30)
    p_down: Decimal = Decimal(down) / Decimal(30)
    if p_up >= EDGE_THRESHOLD:
        signal: SignalDirection = SignalDirection.LONG
    elif p_down >= EDGE_THRESHOLD:
        signal = SignalDirection.SHORT
    else:
        signal = SignalDirection.NEUTRAL
    return PredictionReport(
        symbol=symbol,
        signal=signal,
        sample_count=30,
        paths_up=up,
        paths_down=down,
        paths_flat=1,
        p_up=p_up,
        p_down=p_down,
        anchor_close=Decimal("64123.50"),
    )


def _fake_execution(symbol: str, executed: bool) -> ExecutionResult:
    if executed:
        return ExecutionResult(
            status=ExecutionStatus.EXECUTED,
            symbol=symbol,
            direction=SignalDirection.LONG,
            reason="all sieves passed — bracket placed",
            deviation=Decimal("0.00021"),
            executed_amount=Decimal("0.014"),
            entry_fill_price=Decimal("64130.10"),
            take_profit_price=Decimal("64193.32"),
            stop_loss_price=Decimal("64024.73"),
        )
    return ExecutionResult(
        status=ExecutionStatus.ABORT_SLIPPAGE,
        symbol=symbol,
        direction=SignalDirection.SHORT,
        reason="price slipped past 0.05% execution barrier",
        deviation=Decimal("0.0009"),
    )


async def _demo(duration_s: float = 30.0) -> None:
    """Render the dashboard with cycling synthetic telemetry."""
    symbols: Tuple[str, str] = ("BTC/USDT", "ADA/USDT")
    visualizer: TradingBotVisualizer = TradingBotVisualizer(
        symbols=symbols, exchange_label="Binance Sandbox (demo data)"
    )
    task: "asyncio.Task[None]" = asyncio.create_task(visualizer.run())

    baseline: Decimal = Decimal("10000")
    elapsed: float = 0.0
    tick: int = 0
    while elapsed < duration_s:
        equity: Decimal = baseline - Decimal(random.randint(-80, 220))
        visualizer.publish(
            EquityUpdate(
                equity=equity, baseline=baseline, drawdown_limit=Decimal("0.03")
            )
        )
        for symbol in symbols:
            visualizer.publish(
                RegimeUpdate(
                    symbol=symbol,
                    report=_fake_regime(tick % 3 != 0),
                    close_price=Decimal("64123.50") + Decimal(random.randint(-90, 90)),
                )
            )
            visualizer.publish(
                InferenceUpdate(
                    symbol=symbol,
                    report=_fake_inference(symbol, random.randint(10, 19)),
                )
            )
            visualizer.publish(
                ConfluenceUpdate(symbol=symbol, report=_fake_confluence(tick % 5 != 4))
            )
        if tick % 4 == 2:
            visualizer.publish(
                ExecutionUpdate(result=_fake_execution(symbols[tick % 2], tick % 8 != 6))
            )
        await asyncio.sleep(1.0)
        elapsed += 1.0
        tick += 1

    visualizer.stop()
    await task


if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass

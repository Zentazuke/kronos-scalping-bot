"""journal.py — Phase 7: trade journal, outcome detection, Kelly feedback.

The learning foundation. Every routed bracket is recorded with the full
decision context the bot had at entry time (regime metrics, Monte Carlo
probabilities, confluence votes); the ``OutcomeMonitor`` later detects which
bracket leg filled on the venue, computes realized PnL, and feeds the result
into the ``PerformanceTracker`` so Kelly sizing adapts to real performance.

Components:
  * ``TradeJournal``    — SQLite persistence (stdlib ``sqlite3``). Decimal
                          values are stored as TEXT and round-trip exactly;
                          the binary float domain never touches this file.
                          Writes are single-row and millisecond-cheap, so
                          they run inline on the event loop by design.
  * ``OutcomeMonitor``  — polls the exchange for each OPEN trade's TP/SL
                          order pair. One leg filled => WIN/LOSS recorded,
                          the sibling leg cancelled (spot OCO already
                          auto-cancels — OrderNotFound is tolerated), PnL
                          journaled and folded into the tracker.
  * ``replay_into``     — on boot, closed-trade history re-seeds a fresh
                          PerformanceTracker so the Kelly state survives
                          restarts without any extra state file.

PnL convention (quote currency): LONG => (exit - entry) * amount,
SHORT => (entry - exit) * amount. Exit-side fees are subtracted when the
venue reports them in the quote currency; entry fees are not modelled.
Scratch trades (pnl == 0) are journaled but carry no Kelly information.

Strict typing: annotated for ``mypy --strict``. Embedded unittest suite runs
with a fake exchange and a temp database (``python -m unittest journal``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

from ccxt.base.errors import OrderNotFound  # type: ignore[import-untyped]

from execution import ExecutionResult, PerformanceTracker

__all__ = [
    "TradeJournal",
    "OutcomeMonitor",
    "ObservationJournal",
    "TradeRecord",
    "TradeOutcome",
    "PerformanceSnapshot",
]

logger: Final[logging.Logger] = logging.getLogger("bot.journal")

_ZERO: Final[Decimal] = Decimal("0")

#: Trade lifecycle states persisted in the ``status`` column.
STATUS_OPEN: Final[str] = "OPEN"
STATUS_WIN: Final[str] = "WIN"
STATUS_LOSS: Final[str] = "LOSS"
STATUS_SCRATCH: Final[str] = "SCRATCH"
STATUS_UNKNOWN: Final[str] = "UNKNOWN"  # both legs vanished without a fill

#: Variant identity of THIS bot instance (multi-variant data farm).
#: Every journaled trade is stamped with it; Kelly replay and the live
#: dashboard only ever read their own variant, so a harvester's deliberately
#: bad record can never shrink prod sizing. Defaults to "prod".
DEFAULT_VARIANT: Final[str] = "prod"


def _env_variant() -> str:
    return os.getenv("VARIANT", DEFAULT_VARIANT).strip() or DEFAULT_VARIANT

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_open         TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    amount          TEXT NOT NULL,
    entry_price     TEXT NOT NULL,
    tp_price        TEXT,
    sl_price        TEXT,
    tp_order_id     TEXT,
    sl_order_id     TEXT,
    adx             TEXT,
    atr             TEXT,
    atr_sma         TEXT,
    rsi             TEXT,
    plus_di         TEXT,
    minus_di        TEXT,
    book_imbalance  TEXT,
    p_up            TEXT,
    p_down          TEXT,
    confluence_votes INTEGER,
    meta_p_win      TEXT,
    spread_bps      TEXT,
    relative_volume TEXT,
    depth_imbalance TEXT,
    total_depth     TEXT,
    trade_imbalance TEXT,
    ofi_rel         TEXT,
    mvwap_gap_bps   TEXT,
    microprice_gap_bps TEXT,
    trend_1h        TEXT,
    trend_4h        TEXT,
    rsi_1h          TEXT,
    day_range_pos   TEXT,
    trend_1d        TEXT,
    macro_trend     TEXT,
    dist_30d_high   TEXT,
    vol_pct_1d      TEXT,
    sent_score      TEXT,
    sent_velocity   TEXT,
    attention_spike TEXT,
    fear_greed      TEXT,
    long_short_ratio TEXT,
    funding_rate    TEXT,
    open_interest   TEXT,
    outlook_1h      TEXT,
    ta_macd         TEXT,
    ta_supertrend   TEXT,
    ta_stoch        TEXT,
    ta_cci          TEXT,
    ta_boll         TEXT,
    ta_donchian     TEXT,
    ta_obv          TEXT,
    ta_consensus    TEXT,
    variant         TEXT NOT NULL DEFAULT 'prod',
    status          TEXT NOT NULL DEFAULT 'OPEN',
    ts_close        TEXT,
    exit_price      TEXT,
    pnl             TEXT,
    fees            TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, status);
"""

#: Created AFTER migration — a legacy table has no ``variant`` column yet,
#: so this index cannot live inside ``_SCHEMA``.
_VARIANT_INDEX: Final[str] = (
    "CREATE INDEX IF NOT EXISTS idx_trades_variant ON trades (variant, status)"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_text(value: Optional[Decimal]) -> Optional[str]:
    return None if value is None else str(value)


def _to_decimal(value: Optional[str]) -> Optional[Decimal]:
    return None if value is None else Decimal(value)


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One journaled trade — entry context plus (eventually) its outcome."""

    trade_id: int
    ts_open: str
    symbol: str
    direction: str  # SignalDirection.value, e.g. "STRAT_LONG"
    amount: Decimal
    entry_price: Decimal
    tp_price: Optional[Decimal]
    sl_price: Optional[Decimal]
    tp_order_id: Optional[str]
    sl_order_id: Optional[str]
    adx: Optional[Decimal]
    atr: Optional[Decimal]
    atr_sma: Optional[Decimal]
    rsi: Optional[Decimal]
    plus_di: Optional[Decimal]
    minus_di: Optional[Decimal]
    book_imbalance: Optional[Decimal]
    p_up: Optional[Decimal]
    p_down: Optional[Decimal]
    confluence_votes: Optional[int]
    meta_p_win: Optional[Decimal]
    spread_bps: Optional[Decimal]
    relative_volume: Optional[Decimal]
    depth_imbalance: Optional[Decimal]
    total_depth: Optional[Decimal]
    trade_imbalance: Optional[Decimal]
    ofi_rel: Optional[Decimal]
    mvwap_gap_bps: Optional[Decimal]
    microprice_gap_bps: Optional[Decimal]
    trend_1h: Optional[Decimal]
    trend_4h: Optional[Decimal]
    rsi_1h: Optional[Decimal]
    day_range_pos: Optional[Decimal]
    trend_1d: Optional[Decimal]
    macro_trend: Optional[Decimal]
    dist_30d_high: Optional[Decimal]
    vol_pct_1d: Optional[Decimal]
    variant: str
    status: str
    ts_close: Optional[str]
    exit_price: Optional[Decimal]
    pnl: Optional[Decimal]
    fees: Optional[Decimal]

    @property
    def is_long(self) -> bool:
        return self.direction.endswith("LONG")


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    """A freshly detected close, returned by ``OutcomeMonitor.poll``."""

    trade_id: int
    symbol: str
    direction: str
    amount: Decimal
    entry_price: Decimal
    exit_price: Optional[Decimal]
    pnl: Decimal
    status: str


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    """Aggregate realized results across the whole journal."""

    wins: int
    losses: int
    scratches: int
    open_trades: int
    realized_pnl: Decimal

    @property
    def closed_trades(self) -> int:
        return self.wins + self.losses + self.scratches

    @property
    def win_rate(self) -> Optional[Decimal]:
        decided: int = self.wins + self.losses
        if decided == 0:
            return None
        return Decimal(self.wins) / Decimal(decided)


# --------------------------------------------------------------------------- #
# Journal                                                                      #
# --------------------------------------------------------------------------- #


class TradeJournal:
    """Append-mostly SQLite ledger of every bracket the bot has placed."""

    def __init__(self, db_path: Path, *, variant: Optional[str] = None) -> None:
        self._db_path: Path = db_path
        self._variant: str = variant if variant is not None else _env_variant()
        self._conn: sqlite3.Connection = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.execute(_VARIANT_INDEX)
        self._conn.commit()

    @property
    def variant(self) -> str:
        return self._variant

    def _migrate(self) -> None:
        """Add columns that predate-this-version databases are missing.

        Existing rows get DEFAULT 'prod' — correct, because every trade
        journaled before the data farm existed WAS the prod variant.
        """
        columns: set[str] = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(trades)").fetchall()
        }
        if "variant" not in columns:
            self._conn.execute(
                "ALTER TABLE trades ADD COLUMN variant TEXT NOT NULL DEFAULT 'prod'"
            )
            logger.info("journal migrated: variant column added (existing rows = prod)")
        for feature in (
            "spread_bps",
            "relative_volume",
            "depth_imbalance",
            "total_depth",
            "trade_imbalance",
            "ofi_rel",
            "mvwap_gap_bps",
            "microprice_gap_bps",
            "trend_1h",
            "trend_4h",
            "rsi_1h",
            "day_range_pos",
            "trend_1d",
            "macro_trend",
            "dist_30d_high",
            "vol_pct_1d",
        ):
            if feature not in columns:
                self._conn.execute(
                    f"ALTER TABLE trades ADD COLUMN {feature} TEXT"
                )
                logger.info("journal migrated: %s column added", feature)

    def close(self) -> None:
        self._conn.close()

    # -- writes --------------------------------------------------------- #

    def open_trade(
        self,
        result: ExecutionResult,
        *,
        adx: Optional[Decimal] = None,
        atr: Optional[Decimal] = None,
        atr_sma: Optional[Decimal] = None,
        rsi: Optional[Decimal] = None,
        plus_di: Optional[Decimal] = None,
        minus_di: Optional[Decimal] = None,
        book_imbalance: Optional[Decimal] = None,
        p_up: Optional[Decimal] = None,
        p_down: Optional[Decimal] = None,
        confluence_votes: Optional[int] = None,
        meta_p_win: Optional[Decimal] = None,
        spread_bps: Optional[Decimal] = None,
        relative_volume: Optional[Decimal] = None,
        depth_imbalance: Optional[Decimal] = None,
        total_depth: Optional[Decimal] = None,
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
    ) -> int:
        """Journal one EXECUTED bracket with its full decision context."""
        if result.executed_amount is None or result.entry_fill_price is None:
            raise ValueError("an executed bracket must carry amount and fill price")
        cursor: sqlite3.Cursor = self._conn.execute(
            """
            INSERT INTO trades (
                ts_open, symbol, direction, amount, entry_price,
                tp_price, sl_price, tp_order_id, sl_order_id,
                adx, atr, atr_sma, rsi, plus_di, minus_di, book_imbalance,
                p_up, p_down, confluence_votes, meta_p_win,
                spread_bps, relative_volume, depth_imbalance, total_depth,
                trade_imbalance, ofi_rel, mvwap_gap_bps, microprice_gap_bps,
                trend_1h, trend_4h, rsi_1h, day_range_pos,
                trend_1d, macro_trend, dist_30d_high, vol_pct_1d,
                variant, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                result.symbol,
                result.direction.value,
                str(result.executed_amount),
                str(result.entry_fill_price),
                _to_text(result.take_profit_price),
                _to_text(result.stop_loss_price),
                result.take_profit_order_id,
                result.stop_loss_order_id,
                _to_text(adx),
                _to_text(atr),
                _to_text(atr_sma),
                _to_text(rsi),
                _to_text(plus_di),
                _to_text(minus_di),
                _to_text(book_imbalance),
                _to_text(p_up),
                _to_text(p_down),
                confluence_votes,
                _to_text(meta_p_win),
                _to_text(spread_bps),
                _to_text(relative_volume),
                _to_text(depth_imbalance),
                _to_text(total_depth),
                _to_text(trade_imbalance),
                _to_text(ofi_rel),
                _to_text(mvwap_gap_bps),
                _to_text(microprice_gap_bps),
                _to_text(trend_1h),
                _to_text(trend_4h),
                _to_text(rsi_1h),
                _to_text(day_range_pos),
                _to_text(trend_1d),
                _to_text(macro_trend),
                _to_text(dist_30d_high),
                _to_text(vol_pct_1d),
                self._variant,
                STATUS_OPEN,
            ),
        )
        self._conn.commit()
        trade_id: int = int(cursor.lastrowid or 0)
        logger.info(
            "%s: trade #%d journaled — %s %s @ %s",
            result.symbol,
            trade_id,
            result.direction.value,
            result.executed_amount,
            result.entry_fill_price,
        )
        return trade_id

    def close_trade(
        self,
        trade_id: int,
        *,
        status: str,
        exit_price: Optional[Decimal],
        pnl: Decimal,
        fees: Decimal = _ZERO,
    ) -> None:
        self._conn.execute(
            """
            UPDATE trades
            SET status = ?, ts_close = ?, exit_price = ?, pnl = ?, fees = ?
            WHERE id = ?
            """,
            (status, _utc_now(), _to_text(exit_price), str(pnl), str(fees), trade_id),
        )
        self._conn.commit()

    # -- reads ----------------------------------------------------------- #

    @staticmethod
    def _record(row: sqlite3.Row) -> TradeRecord:
        return TradeRecord(
            trade_id=int(row["id"]),
            ts_open=str(row["ts_open"]),
            symbol=str(row["symbol"]),
            direction=str(row["direction"]),
            amount=Decimal(str(row["amount"])),
            entry_price=Decimal(str(row["entry_price"])),
            tp_price=_to_decimal(row["tp_price"]),
            sl_price=_to_decimal(row["sl_price"]),
            tp_order_id=row["tp_order_id"],
            sl_order_id=row["sl_order_id"],
            adx=_to_decimal(row["adx"]),
            atr=_to_decimal(row["atr"]),
            atr_sma=_to_decimal(row["atr_sma"]),
            rsi=_to_decimal(row["rsi"]),
            plus_di=_to_decimal(row["plus_di"]),
            minus_di=_to_decimal(row["minus_di"]),
            book_imbalance=_to_decimal(row["book_imbalance"]),
            p_up=_to_decimal(row["p_up"]),
            p_down=_to_decimal(row["p_down"]),
            confluence_votes=(
                int(row["confluence_votes"])
                if row["confluence_votes"] is not None
                else None
            ),
            meta_p_win=_to_decimal(row["meta_p_win"]),
            spread_bps=_to_decimal(row["spread_bps"]),
            relative_volume=_to_decimal(row["relative_volume"]),
            depth_imbalance=_to_decimal(row["depth_imbalance"]),
            total_depth=_to_decimal(row["total_depth"]),
            trade_imbalance=_to_decimal(row["trade_imbalance"]),
            ofi_rel=_to_decimal(row["ofi_rel"]),
            mvwap_gap_bps=_to_decimal(row["mvwap_gap_bps"]),
            microprice_gap_bps=_to_decimal(row["microprice_gap_bps"]),
            trend_1h=_to_decimal(row["trend_1h"]),
            trend_4h=_to_decimal(row["trend_4h"]),
            rsi_1h=_to_decimal(row["rsi_1h"]),
            day_range_pos=_to_decimal(row["day_range_pos"]),
            trend_1d=_to_decimal(row["trend_1d"]),
            macro_trend=_to_decimal(row["macro_trend"]),
            dist_30d_high=_to_decimal(row["dist_30d_high"]),
            vol_pct_1d=_to_decimal(row["vol_pct_1d"]),
            variant=str(row["variant"]),
            status=str(row["status"]),
            ts_close=row["ts_close"],
            exit_price=_to_decimal(row["exit_price"]),
            pnl=_to_decimal(row["pnl"]),
            fees=_to_decimal(row["fees"]),
        )

    def open_trades(self, symbol: Optional[str] = None) -> List[TradeRecord]:
        """OPEN trades belonging to THIS variant (monitor must never touch
        a sibling bot's brackets, even if journals are ever pooled)."""
        if symbol is None:
            rows: List[sqlite3.Row] = self._conn.execute(
                "SELECT * FROM trades WHERE status = ? AND variant = ? ORDER BY id",
                (STATUS_OPEN, self._variant),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trades"
                " WHERE status = ? AND symbol = ? AND variant = ? ORDER BY id",
                (STATUS_OPEN, symbol, self._variant),
            ).fetchall()
        return [self._record(row) for row in rows]

    def closed_trades(self, *, variant: Optional[str] = None) -> List[TradeRecord]:
        """Closed trades — ALL variants by default (the meta-labeler pools
        across variants on purpose); pass ``variant=`` to scope."""
        if variant is None:
            rows: List[sqlite3.Row] = self._conn.execute(
                "SELECT * FROM trades WHERE status != ? ORDER BY id", (STATUS_OPEN,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE status != ? AND variant = ? ORDER BY id",
                (STATUS_OPEN, variant),
            ).fetchall()
        return [self._record(row) for row in rows]

    def performance(self) -> PerformanceSnapshot:
        """Aggregate results for THIS variant only (dashboard/session tally)."""
        wins = losses = scratches = open_count = 0
        realized: Decimal = _ZERO
        for row in self._conn.execute(
            "SELECT status, pnl FROM trades WHERE variant = ?", (self._variant,)
        ).fetchall():
            status: str = str(row["status"])
            if status == STATUS_OPEN:
                open_count += 1
                continue
            if status == STATUS_WIN:
                wins += 1
            elif status == STATUS_LOSS:
                losses += 1
            else:
                scratches += 1
            pnl: Optional[Decimal] = _to_decimal(row["pnl"])
            if pnl is not None:
                realized += pnl
        return PerformanceSnapshot(
            wins=wins,
            losses=losses,
            scratches=scratches,
            open_trades=open_count,
            realized_pnl=realized,
        )

    def replay_into(self, tracker: PerformanceTracker) -> int:
        """Fold OWN-variant realized PnL into a fresh tracker. Returns count.

        Variant-scoped on purpose: the harvester journals deliberately
        unfiltered trades, and its win rate must never leak into another
        variant's Kelly posterior.
        """
        replayed: int = 0
        for trade in self.closed_trades(variant=self._variant):
            if trade.pnl is not None and trade.status in (STATUS_WIN, STATUS_LOSS):
                tracker.record_trade(trade.pnl)
                replayed += 1
        if replayed:
            logger.info(
                "Kelly state restored from journal (variant=%s): %d trades replayed",
                self._variant,
                replayed,
            )
        return replayed


# --------------------------------------------------------------------------- #
# Observation journal                                                          #
# --------------------------------------------------------------------------- #


class ObservationJournal:
    """Every directional setup the bot *evaluated* — including the ones blocked
    by the position cap or vetoed, which never become real trades. That is the
    10x data the meta-learner is starved for.

    Uses the SAME schema as the trade journal, so the learner reads it directly
    (``learner.py walkforward --db observations.db``). Each row is a
    *hypothetical* trade, written OPEN and unlabeled; ``label_observations.py``
    later replays its bracket against real candles to stamp WIN/LOSS — which
    also means observation labels are free of testnet phantom fills.

    Append-only, in its own file. It can never touch the trade journal.
    """

    _FEATURE_COLS: Final[Tuple[str, ...]] = (
        "adx", "atr", "atr_sma", "rsi", "plus_di", "minus_di", "book_imbalance",
        "p_up", "p_down", "confluence_votes", "spread_bps", "relative_volume",
        "depth_imbalance", "total_depth", "trade_imbalance", "ofi_rel",
        "mvwap_gap_bps", "microprice_gap_bps", "trend_1h", "trend_4h", "rsi_1h",
        "day_range_pos", "trend_1d", "macro_trend", "dist_30d_high", "vol_pct_1d",
        "sent_score", "sent_velocity", "attention_spike", "fear_greed",
        "long_short_ratio", "funding_rate", "open_interest", "outlook_1h",
        "ta_macd", "ta_supertrend", "ta_stoch", "ta_cci", "ta_boll",
        "ta_donchian", "ta_obv", "ta_consensus",
    )

    def __init__(self, db_path: Path) -> None:
        self._conn: sqlite3.Connection = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # Migrate older observation DBs: add any feature column they predate
        # (e.g. the sentiment columns). Idempotent, additive (NULL on old rows).
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(trades)")}
        for col in self._FEATURE_COLS:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE trades ADD COLUMN {col} TEXT")
                logger.info("observations migrated: %s column added", col)
        self._conn.commit()

    def record(
        self,
        *,
        symbol: str,
        direction: str,
        entry_price: Decimal,
        **features: Any,
    ) -> None:
        """Append one OPEN, unlabeled observation. ``features`` accepts the same
        keyword columns the trade journal stores (adx, atr ... vol_pct_1d);
        None values and unknown keys are skipped (left NULL)."""
        cols: List[str] = [
            "ts_open", "symbol", "direction", "amount",
            "entry_price", "variant", "status",
        ]
        vals: List[Any] = [
            _utc_now(), symbol, direction, "1",
            _to_text(entry_price), "observation", STATUS_OPEN,
        ]
        for col in self._FEATURE_COLS:
            value: Any = features.get(col)
            if value is None:
                continue
            cols.append(col)
            vals.append(int(value) if col == "confluence_votes" else _to_text(value))
        placeholders: str = ", ".join("?" for _ in cols)
        self._conn.execute(
            f"INSERT INTO trades ({', '.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        self._conn.commit()

    def open_count(self) -> int:
        """How many observations still await a label (status OPEN)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status = ?", (STATUS_OPEN,)
        ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------- #
# Outcome monitor                                                              #
# --------------------------------------------------------------------------- #


class OutcomeMonitor:
    """Detects bracket resolutions on the venue and records them.

    Polled once per confirmed bar — outcome *detection* may lag the fill by
    up to one bar, but the recorded PnL is exact because it derives from the
    filled order's own average price, not from the detection-time market.
    """

    def __init__(
        self,
        exchange: Any,
        journal: TradeJournal,
        tracker: PerformanceTracker,
        *,
        quote_currency: str = "USDT",
    ) -> None:
        self._exchange: Any = exchange
        self._journal: TradeJournal = journal
        self._tracker: PerformanceTracker = tracker
        self._quote: str = quote_currency

    async def poll(self, symbol: str) -> List[TradeOutcome]:
        """Check every OPEN trade for ``symbol``; record and return closes."""
        outcomes: List[TradeOutcome] = []
        for trade in self._journal.open_trades(symbol):
            try:
                outcome: Optional[TradeOutcome] = await self._check(trade)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — monitor must never kill a bar
                logger.error(
                    "%s: outcome check failed for trade #%d — retrying next bar",
                    symbol,
                    trade.trade_id,
                    exc_info=True,
                )
                continue
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes

    # -- internals ------------------------------------------------------- #

    async def _fetch_order(
        self, order_id: Optional[str], symbol: str
    ) -> Optional[Dict[str, Any]]:
        if not order_id:
            return None
        order: Any = await self._exchange.fetch_order(order_id, symbol)
        return order if isinstance(order, dict) else None

    @staticmethod
    def _is_filled(order: Optional[Dict[str, Any]]) -> bool:
        return order is not None and str(order.get("status")) == "closed"

    @staticmethod
    def _is_gone(order: Optional[Dict[str, Any]]) -> bool:
        return order is None or str(order.get("status")) in (
            "canceled",
            "cancelled",
            "expired",
            "rejected",
        )

    def _exit_price(
        self, order: Dict[str, Any], fallback: Optional[Decimal]
    ) -> Optional[Decimal]:
        raw: Any = order.get("average") or order.get("price")
        if raw is not None:
            return Decimal(str(raw))
        return fallback

    def _quote_fee(self, order: Dict[str, Any]) -> Decimal:
        fee: Any = order.get("fee")
        if (
            isinstance(fee, dict)
            and fee.get("cost") is not None
            and str(fee.get("currency")) == self._quote
        ):
            return Decimal(str(fee["cost"]))
        return _ZERO

    async def _cancel_sibling(self, order_id: Optional[str], symbol: str) -> None:
        """Cancel the surviving bracket leg (no-op when the venue already did:
        spot OCO auto-cancels and answers OrderNotFound)."""
        if not order_id:
            return
        try:
            await self._exchange.cancel_order(order_id, symbol)
        except OrderNotFound:
            pass
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.error(
                "%s: failed to cancel surviving bracket leg %s",
                symbol,
                order_id,
                exc_info=True,
            )

    async def _check(self, trade: TradeRecord) -> Optional[TradeOutcome]:
        tp_order: Optional[Dict[str, Any]] = await self._fetch_order(
            trade.tp_order_id, trade.symbol
        )
        sl_order: Optional[Dict[str, Any]] = await self._fetch_order(
            trade.sl_order_id, trade.symbol
        )

        # If both legs simultaneously report filled (pathological), treat the
        # stop as the exit — the conservative reading.
        if self._is_filled(sl_order):
            assert sl_order is not None
            exit_price = self._exit_price(sl_order, trade.sl_price)
            fees = self._quote_fee(sl_order)
            await self._cancel_sibling(trade.tp_order_id, trade.symbol)
            return self._record_close(trade, exit_price, fees)
        if self._is_filled(tp_order):
            assert tp_order is not None
            exit_price = self._exit_price(tp_order, trade.tp_price)
            fees = self._quote_fee(tp_order)
            await self._cancel_sibling(trade.sl_order_id, trade.symbol)
            return self._record_close(trade, exit_price, fees)

        if self._is_gone(tp_order) and self._is_gone(sl_order):
            # Both legs vanished without a fill — manual cancel or venue
            # cleanup. Position state is unknown; journal it honestly.
            logger.warning(
                "%s: trade #%d bracket legs disappeared without a fill — "
                "recorded UNKNOWN, manual check advised",
                trade.symbol,
                trade.trade_id,
            )
            self._journal.close_trade(
                trade.trade_id,
                status=STATUS_UNKNOWN,
                exit_price=None,
                pnl=_ZERO,
            )
            return TradeOutcome(
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                direction=trade.direction,
                amount=trade.amount,
                entry_price=trade.entry_price,
                exit_price=None,
                pnl=_ZERO,
                status=STATUS_UNKNOWN,
            )
        return None  # both legs still resting — trade remains open

    def _record_close(
        self,
        trade: TradeRecord,
        exit_price: Optional[Decimal],
        fees: Decimal,
    ) -> TradeOutcome:
        if exit_price is None:
            gross: Decimal = _ZERO
        elif trade.is_long:
            gross = (exit_price - trade.entry_price) * trade.amount
        else:
            gross = (trade.entry_price - exit_price) * trade.amount
        pnl: Decimal = gross - fees
        if pnl > _ZERO:
            status: str = STATUS_WIN
        elif pnl < _ZERO:
            status = STATUS_LOSS
        else:
            status = STATUS_SCRATCH

        self._journal.close_trade(
            trade.trade_id,
            status=status,
            exit_price=exit_price,
            pnl=pnl,
            fees=fees,
        )
        self._tracker.record_trade(pnl)
        logger.info(
            "%s: trade #%d closed %s — entry %s exit %s pnl %s %s (fees %s)",
            trade.symbol,
            trade.trade_id,
            status,
            trade.entry_price,
            exit_price,
            pnl,
            self._quote,
            fees,
        )
        return TradeOutcome(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            direction=trade.direction,
            amount=trade.amount,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            status=status,
        )


# --------------------------------------------------------------------------- #
# Embedded tests                                                               #
# --------------------------------------------------------------------------- #

from predictor import SignalDirection  # noqa: E402  (test-only convenience)
from execution import ExecutionStatus  # noqa: E402

_SYMBOL: Final[str] = "BTC/USDT"


def _executed_result(
    direction: SignalDirection = SignalDirection.LONG,
    *,
    amount: str = "0.010",
    fill: str = "64000",
    tp: str = "64100",
    sl: str = "63800",
) -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.EXECUTED,
        symbol=_SYMBOL,
        direction=direction,
        reason="test bracket",
        executed_amount=Decimal(amount),
        entry_fill_price=Decimal(fill),
        take_profit_price=Decimal(tp),
        stop_loss_price=Decimal(sl),
        take_profit_order_id="tp-1",
        stop_loss_order_id="sl-1",
    )


class _FakeOrderExchange:
    """Venue truth for the monitor: a dict of order states per id."""

    def __init__(self, orders: Dict[str, Dict[str, Any]]) -> None:
        self.orders: Dict[str, Dict[str, Any]] = orders
        self.cancelled: List[str] = []
        self.cancel_raises_not_found: bool = False

    async def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        if order_id not in self.orders:
            raise OrderNotFound(f"unknown order {order_id}")
        return dict(self.orders[order_id])

    async def cancel_order(self, order_id: str, symbol: str) -> None:
        if self.cancel_raises_not_found:
            raise OrderNotFound("already gone")
        self.cancelled.append(order_id)


class TradeJournalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.journal: TradeJournal = TradeJournal(Path(self._tmp.name) / "j.db")

    def tearDown(self) -> None:
        self.journal.close()
        self._tmp.cleanup()

    def test_open_trade_round_trips_decimal_context(self) -> None:
        trade_id: int = self.journal.open_trade(
            _executed_result(),
            adx=Decimal("31.5"),
            rsi=Decimal("62.1"),
            book_imbalance=Decimal("0.58"),
            p_up=Decimal("0.6"),
            confluence_votes=2,
            spread_bps=Decimal("4.2"),
            relative_volume=Decimal("2.5"),
            depth_imbalance=Decimal("-0.18"),
            total_depth=Decimal("145"),
            trade_imbalance=Decimal("0.33"),
            ofi_rel=Decimal("-0.07"),
            mvwap_gap_bps=Decimal("1.8"),
            microprice_gap_bps=Decimal("0.6"),
            trend_1h=Decimal("1"),
            trend_4h=Decimal("-1"),
            rsi_1h=Decimal("58.2"),
            day_range_pos=Decimal("0.81"),
            macro_trend=Decimal("-1"),
            dist_30d_high=Decimal("-0.15"),
        )
        (trade,) = self.journal.open_trades(_SYMBOL)
        self.assertEqual(trade.trade_id, trade_id)
        self.assertEqual(trade.amount, Decimal("0.010"))
        self.assertEqual(trade.entry_price, Decimal("64000"))
        self.assertEqual(trade.adx, Decimal("31.5"))
        self.assertEqual(trade.book_imbalance, Decimal("0.58"))
        self.assertEqual(trade.confluence_votes, 2)
        self.assertEqual(trade.spread_bps, Decimal("4.2"))
        self.assertEqual(trade.relative_volume, Decimal("2.5"))
        self.assertEqual(trade.depth_imbalance, Decimal("-0.18"))
        self.assertEqual(trade.total_depth, Decimal("145"))
        self.assertEqual(trade.trade_imbalance, Decimal("0.33"))
        self.assertEqual(trade.ofi_rel, Decimal("-0.07"))
        self.assertEqual(trade.mvwap_gap_bps, Decimal("1.8"))
        self.assertEqual(trade.microprice_gap_bps, Decimal("0.6"))
        self.assertEqual(trade.trend_1h, Decimal("1"))
        self.assertEqual(trade.trend_4h, Decimal("-1"))
        self.assertEqual(trade.rsi_1h, Decimal("58.2"))
        self.assertEqual(trade.day_range_pos, Decimal("0.81"))
        self.assertEqual(trade.macro_trend, Decimal("-1"))
        self.assertEqual(trade.dist_30d_high, Decimal("-0.15"))
        self.assertIsNone(trade.trend_1d)
        self.assertTrue(trade.is_long)

    def test_performance_snapshot_aggregates(self) -> None:
        first: int = self.journal.open_trade(_executed_result())
        second: int = self.journal.open_trade(_executed_result())
        self.journal.open_trade(_executed_result())  # stays open
        self.journal.close_trade(
            first, status=STATUS_WIN, exit_price=Decimal("64100"), pnl=Decimal("1.0")
        )
        self.journal.close_trade(
            second, status=STATUS_LOSS, exit_price=Decimal("63800"), pnl=Decimal("-2.0")
        )
        snapshot: PerformanceSnapshot = self.journal.performance()
        self.assertEqual((snapshot.wins, snapshot.losses), (1, 1))
        self.assertEqual(snapshot.open_trades, 1)
        self.assertEqual(snapshot.realized_pnl, Decimal("-1.0"))
        self.assertEqual(snapshot.win_rate, Decimal("0.5"))

    def test_replay_restores_kelly_state(self) -> None:
        first: int = self.journal.open_trade(_executed_result())
        self.journal.close_trade(
            first, status=STATUS_WIN, exit_price=Decimal("64100"), pnl=Decimal("1.0")
        )
        fresh = PerformanceTracker()
        baseline: Decimal = PerformanceTracker().win_rate
        replayed: int = self.journal.replay_into(fresh)
        self.assertEqual(replayed, 1)
        self.assertGreater(fresh.win_rate, baseline)  # the win moved the posterior

    async def test_monitor_records_tp_win_and_cancels_sibling(self) -> None:
        trade_id: int = self.journal.open_trade(_executed_result())
        exchange = _FakeOrderExchange(
            {
                "tp-1": {"status": "closed", "average": 64100.0,
                         "fee": {"cost": 0.05, "currency": "USDT"}},
                "sl-1": {"status": "open"},
            }
        )
        tracker = PerformanceTracker()
        monitor = OutcomeMonitor(exchange, self.journal, tracker)
        (outcome,) = await monitor.poll(_SYMBOL)
        self.assertEqual(outcome.status, STATUS_WIN)
        # (64100 - 64000) * 0.010 = 1.0 gross, minus 0.05 quote fee.
        self.assertEqual(outcome.pnl, Decimal("0.95"))
        self.assertIn("sl-1", exchange.cancelled)
        self.assertEqual(self.journal.open_trades(_SYMBOL), [])
        self.assertEqual(self.journal.performance().wins, 1)
        self.assertEqual(tracker._wins, 1)  # fed straight into Kelly

    async def test_monitor_records_sl_loss_for_short(self) -> None:
        trade_id: int = self.journal.open_trade(
            _executed_result(
                SignalDirection.SHORT, fill="64000", tp="63900", sl="64200"
            )
        )
        exchange = _FakeOrderExchange(
            {
                "tp-1": {"status": "canceled"},
                "sl-1": {"status": "closed", "average": 64200.0},
            }
        )
        tracker = PerformanceTracker()
        monitor = OutcomeMonitor(exchange, self.journal, tracker)
        (outcome,) = await monitor.poll(_SYMBOL)
        self.assertEqual(outcome.status, STATUS_LOSS)
        # Short: (64000 - 64200) * 0.010 = -2.0
        self.assertEqual(outcome.pnl, Decimal("-2.0"))
        self.assertEqual(tracker._losses, 1)

    async def test_monitor_leaves_resting_brackets_open(self) -> None:
        self.journal.open_trade(_executed_result())
        exchange = _FakeOrderExchange(
            {"tp-1": {"status": "open"}, "sl-1": {"status": "open"}}
        )
        monitor = OutcomeMonitor(exchange, self.journal, PerformanceTracker())
        self.assertEqual(await monitor.poll(_SYMBOL), [])
        self.assertEqual(len(self.journal.open_trades(_SYMBOL)), 1)

    async def test_monitor_flags_vanished_brackets_unknown(self) -> None:
        self.journal.open_trade(_executed_result())
        exchange = _FakeOrderExchange(
            {"tp-1": {"status": "canceled"}, "sl-1": {"status": "canceled"}}
        )
        tracker = PerformanceTracker()
        monitor = OutcomeMonitor(exchange, self.journal, tracker)
        (outcome,) = await monitor.poll(_SYMBOL)
        self.assertEqual(outcome.status, STATUS_UNKNOWN)
        self.assertEqual(outcome.pnl, _ZERO)
        self.assertEqual(tracker._wins + tracker._losses, 0)  # no false signal

    async def test_monitor_tolerates_oco_auto_cancel(self) -> None:
        self.journal.open_trade(_executed_result())
        exchange = _FakeOrderExchange(
            {"tp-1": {"status": "closed", "average": 64100.0},
             "sl-1": {"status": "open"}}
        )
        exchange.cancel_raises_not_found = True  # venue already removed it
        monitor = OutcomeMonitor(exchange, self.journal, PerformanceTracker())
        (outcome,) = await monitor.poll(_SYMBOL)
        self.assertEqual(outcome.status, STATUS_WIN)  # close still recorded

    def test_variant_stamped_and_replay_is_variant_scoped(self) -> None:
        db: Path = Path(self._tmp.name) / "farm.db"
        prod = TradeJournal(db, variant="prod")
        harvester = TradeJournal(db, variant="harvester")
        try:
            p_id: int = prod.open_trade(_executed_result())
            h_id: int = harvester.open_trade(_executed_result())
            prod.close_trade(
                p_id, status=STATUS_WIN, exit_price=Decimal("64100"), pnl=Decimal("1.0")
            )
            harvester.close_trade(
                h_id, status=STATUS_LOSS, exit_price=Decimal("63800"), pnl=Decimal("-2.0")
            )
            # Records carry their variant.
            self.assertEqual(
                {t.variant for t in prod.closed_trades()}, {"prod", "harvester"}
            )
            # replay_into only sees OWN variant: prod replays 1 win, no loss.
            fresh = PerformanceTracker()
            self.assertEqual(prod.replay_into(fresh), 1)
            self.assertEqual((fresh._wins, fresh._losses), (1, 0))
            # performance() is variant-scoped too.
            self.assertEqual(prod.performance().wins, 1)
            self.assertEqual(prod.performance().losses, 0)
            self.assertEqual(harvester.performance().losses, 1)
        finally:
            prod.close()
            harvester.close()

    def test_open_trades_scoped_to_own_variant(self) -> None:
        db: Path = Path(self._tmp.name) / "farm2.db"
        prod = TradeJournal(db, variant="prod")
        relaxed = TradeJournal(db, variant="relaxed")
        try:
            prod.open_trade(_executed_result())
            relaxed.open_trade(_executed_result())
            self.assertEqual(len(prod.open_trades(_SYMBOL)), 1)
            self.assertEqual(prod.open_trades(_SYMBOL)[0].variant, "prod")
            self.assertEqual(len(relaxed.open_trades()), 1)
        finally:
            prod.close()
            relaxed.close()

    def test_legacy_db_migrates_to_prod_variant(self) -> None:
        db: Path = Path(self._tmp.name) / "legacy.db"
        legacy_conn = sqlite3.connect(str(db))
        legacy_conn.executescript(
            _SCHEMA.replace("    variant         TEXT NOT NULL DEFAULT 'prod',\n", "")
        )
        legacy_conn.execute(
            "INSERT INTO trades (ts_open, symbol, direction, amount, entry_price,"
            " status, pnl) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_utc_now(), _SYMBOL, "STRAT_LONG", "0.01", "64000", STATUS_WIN, "1.0"),
        )
        legacy_conn.commit()
        legacy_conn.close()
        migrated = TradeJournal(db)  # VARIANT env unset => "prod"
        try:
            (trade,) = migrated.closed_trades()
            self.assertEqual(trade.variant, "prod")
            self.assertEqual(migrated.performance().wins, 1)
        finally:
            migrated.close()

    def test_variant_defaults_from_env(self) -> None:
        os.environ["VARIANT"] = "relaxed"
        try:
            j = TradeJournal(Path(self._tmp.name) / "env.db")
            self.assertEqual(j.variant, "relaxed")
            j.close()
        finally:
            del os.environ["VARIANT"]

    async def test_monitor_fetch_failure_is_contained(self) -> None:
        self.journal.open_trade(_executed_result())

        class _Broken:
            async def fetch_order(self, order_id: str, symbol: str) -> None:
                raise RuntimeError("venue hiccup")

        monitor = OutcomeMonitor(_Broken(), self.journal, PerformanceTracker())
        self.assertEqual(await monitor.poll(_SYMBOL), [])  # no crash, stays open
        self.assertEqual(len(self.journal.open_trades(_SYMBOL)), 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
# end of journal.py

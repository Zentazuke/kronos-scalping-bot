"""ofi_maker_backtest.py — does MAKER execution make the OFI edge tradeable?

OFI is the one real signal we found, but it's thin and net-NEGATIVE as a TAKER
(you pay the spread chasing flow). The only thing that could flip that is entering
strong-OFI setups as a MAKER: rest a limit, capture the spread/rebate instead of
paying it — at the cost of MISSING the fast favorable moves that run away from
your resting order (adverse selection), and only filling the ones price comes back
to. That trade-off is the whole question, and it IS simulable from forward candles.

For every strong-OFI observation it rests a limit, replays forward mainnet candles
to see if it fills within a window, and if filled runs the bracket — net of maker
fees — against the taker baseline on the same setups. Reports fill rate, maker vs
taker expectancy, by symbol, by month, by side.

HONEST LIMITS (read before trusting a green):
  * Candles can't model queue position: if price TOUCHES the limit we assume a
    fill. Real maker fills are LOWER (you may be behind the queue). So fill rate
    here is an OPTIMISTIC upper bound — read this as 'best case for maker'.
  * OFI only exists in the harvested observation window (no historical order flow),
    so this is the same limited time span, not months/regimes.
  * Exits are modelled with a blended cost; a real stop is a taker fill.

    python ofi_maker_backtest.py --db observations.db --min-ofi 8.34 --offset-bps 2
    python ofi_maker_backtest.py --db observations.db --min-ofi 8.34 --offset-bps 0 --fill-window 6
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from label_observations import _fetch_mainnet_candles, _to_ms

logger = logging.getLogger("bot.ofi_maker")

TP_MULT = 2.5
SL_MULT = 2.5


def _f(v) -> Optional[float]:
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


# obs = (ts_ms, symbol, is_long, entry, atr, aligned_ofi)
Obs = Tuple[int, str, bool, float, float, float]


def load_strong_ofi(db_path: str, min_ofi: float) -> List[Obs]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ts_open, symbol, direction, entry_price, atr, ofi_rel "
        "FROM trades WHERE status IN ('WIN','LOSS','SCRATCH','OPEN') ORDER BY ts_open"
    ).fetchall()
    conn.close()
    out: List[Obs] = []
    for r in rows:
        ts = _to_ms(r["ts_open"])
        entry, atr, ofi = _f(r["entry_price"]), _f(r["atr"]), _f(r["ofi_rel"])
        if ts is None or entry is None or atr is None or ofi is None or atr <= 0:
            continue
        is_long = str(r["direction"]).endswith("LONG")
        aligned = ofi if is_long else -ofi
        if aligned < min_ofi:
            continue
        out.append((ts, str(r["symbol"]), is_long, entry, atr, aligned))
    return out


def _start_idx(candles: Sequence[Sequence[float]], ts: int) -> Optional[int]:
    lo, hi = 0, len(candles) - 1
    if not candles or candles[-1][0] < ts:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if candles[mid][0] < ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


def maker_fill(candles, start: int, is_long: bool, limit: float, fill_window: int) -> Optional[int]:
    """Index of the bar that fills the resting limit within the window, else None."""
    end = min(start + fill_window, len(candles) - 1)
    for k in range(start, end + 1):
        if (candles[k][3] <= limit) if is_long else (candles[k][2] >= limit):
            return k
    return None


def bracket(candles, f: int, is_long: bool, entry: float, atr: float, window: int) -> Optional[float]:
    """Gross per-unit return of a 2.5/2.5 ATR bracket from fill bar f. Pessimistic
    straddle. None if not enough forward bars to resolve."""
    tp = entry + TP_MULT * atr if is_long else entry - TP_MULT * atr
    sl = entry - SL_MULT * atr if is_long else entry + SL_MULT * atr
    last = min(f + window, len(candles) - 1)
    if last - f < 1:
        return None
    ex: Optional[float] = None
    for k in range(f, last + 1):
        hi, lo = candles[k][2], candles[k][3]
        hit_tp = (hi >= tp) if is_long else (lo <= tp)
        hit_sl = (lo <= sl) if is_long else (hi >= sl)
        if hit_sl and hit_tp:
            ex = sl
            break
        if hit_tp:
            ex = tp
            break
        if hit_sl:
            ex = sl
            break
    if ex is None:
        ex = candles[last][4]
    return (ex - entry) / entry if is_long else (entry - ex) / entry


def _perf(nets: Sequence[float]) -> Dict[str, float]:
    n = len(nets)
    if n == 0:
        return {"n": 0, "win": 0.0, "exp": 0.0, "total": 0.0}
    return {"n": n, "win": sum(1 for x in nets if x > 0) / n,
            "exp": sum(nets) / n, "total": sum(nets)}


def _line(label: str, p: Dict[str, float]) -> str:
    if p["n"] == 0:
        return f"  {label:<22} (none)"
    return (f"  {label:<22} n={int(p['n']):<5} win {p['win']*100:>4.0f}%  "
            f"net/trade {p['exp']*100:>+7.3f}%  total {p['total']*100:>+7.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Maker-execution test for the OFI edge")
    ap.add_argument("--db", default="observations.db")
    ap.add_argument("--min-ofi", type=float, default=8.34, help="aligned OFI gate (top ~30%)")
    ap.add_argument("--offset-bps", type=float, default=2.0,
                    help="how far inside the price to rest the maker limit (bps); higher = better "
                         "fills but fewer of them")
    ap.add_argument("--fill-window", type=int, default=6, help="bars to wait for the limit to fill")
    ap.add_argument("--maker-entry-bps", type=float, default=1.0, help="maker entry cost (rebate = negative)")
    ap.add_argument("--exit-bps", type=float, default=5.0, help="blended exit cost (bps)")
    ap.add_argument("--taker-roundtrip-bps", type=float, default=10.0, help="taker baseline cost")
    ap.add_argument("--resolve-window", type=int, default=48)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    offset = args.offset_bps / 10000.0
    maker_cost = (args.maker_entry_bps + args.exit_bps) / 10000.0
    taker_cost = args.taker_roundtrip_bps / 10000.0

    obs = load_strong_ofi(args.db, args.min_ofi)
    if not obs:
        logger.info("no strong-OFI observations at min-ofi=%.2f — lower it or keep harvesting", args.min_ofi)
        return 1

    by_symbol: Dict[str, List[Obs]] = defaultdict(list)
    for o in obs:
        by_symbol[o[1]].append(o)

    # rows: (ts, symbol, side, maker_net or None if missed, taker_net)
    maker_rows: List[Tuple[int, str, str, float]] = []
    taker_rows: List[Tuple[int, str, str, float]] = []
    n_signals = filled = 0

    for symbol, group in by_symbol.items():
        starts = [o[0] for o in group]
        candles = _fetch_mainnet_candles(symbol, min(starts) - 5 * 60_000)
        if not candles:
            logger.warning("%s: no candles — skipped", symbol)
            continue
        for ts, _s, is_long, entry, atr, _ofi in group:
            si = _start_idx(candles, ts)
            if si is None:
                continue
            post = si + 1  # act only from the bar AFTER the signal closes (no lookahead)
            if post >= len(candles) - 1:
                continue
            n_signals += 1
            side = "LONG" if is_long else "SHORT"
            # TAKER baseline: enter at signal price, bracket from the next bar
            t_ret = bracket(candles, post, is_long, entry, atr, args.resolve_window)
            if t_ret is not None:
                taker_rows.append((ts, symbol, side, t_ret - taker_cost))
            # MAKER: rest a limit `offset` better than the signal price
            limit = entry * (1 - offset) if is_long else entry * (1 + offset)
            fi = maker_fill(candles, post, is_long, limit, args.fill_window)
            if fi is None:
                continue  # never filled — the fast mover that ran away
            filled += 1
            m_ret = bracket(candles, fi, is_long, limit, atr, args.resolve_window)
            if m_ret is not None:
                maker_rows.append((ts, symbol, side, m_ret - maker_cost))

    if not maker_rows and not taker_rows:
        logger.info("no resolvable setups — keep harvesting")
        return 1

    fill_rate = (filled / n_signals) if n_signals else 0.0
    mk = _perf([r[3] for r in maker_rows])
    tk = _perf([r[3] for r in taker_rows])

    print(f"\n=== OFI MAKER vs TAKER — min-OFI {args.min_ofi:g}, offset {args.offset_bps:g}bps, "
          f"fill window {args.fill_window} bars ===")
    print(f"{n_signals} strong-OFI signals · maker FILLED {filled} ({fill_rate*100:.0f}%) — "
          f"the other {100-fill_rate*100:.0f}% ran away (missed)\n")
    print("OVERALL:")
    print(_line(f"TAKER ({args.taker_roundtrip_bps:g}bps)", tk))
    print(_line(f"MAKER ({args.maker_entry_bps:g}+{args.exit_bps:g}bps)", mk))

    print("\nMAKER BY SIDE:")
    for side in ("LONG", "SHORT"):
        print(_line(side, _perf([r[3] for r in maker_rows if r[2] == side])))

    print("\nMAKER BY SYMBOL:")
    by_sym: Dict[str, List[float]] = defaultdict(list)
    for r in maker_rows:
        by_sym[r[1]].append(r[3])
    for s in sorted(by_sym, key=lambda k: -_perf(by_sym[k])["total"]):
        print(_line(s, _perf(by_sym[s])))

    print("\nMAKER BY MONTH:")
    by_month: Dict[str, List[float]] = defaultdict(list)
    for r in maker_rows:
        by_month[datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc).strftime("%Y-%m")].append(r[3])
    pos_months = 0
    for ym in sorted(by_month):
        p = _perf(by_month[ym])
        pos_months += 1 if p["exp"] > 0 else 0
        print(_line(ym, p))

    n_months = len(by_month)
    pos_syms = sum(1 for s in by_sym if _perf(by_sym[s])["exp"] > 0)
    print("\n=== read ===")
    print(f"taker net {tk['exp']*100:+.3f}%/trade  →  maker net {mk['exp']*100:+.3f}%/trade  "
          f"(at {fill_rate*100:.0f}% fill)")
    if mk["exp"] > 0 and pos_months >= max(2, n_months * 0.6) and pos_syms >= len(by_sym) * 0.6:
        print("maker execution turns OFI net-positive AND broad/persistent — the real lead. "
              "Next: validate fills on tiny LIVE orders (the part candles can't prove).")
    elif mk["exp"] > 0:
        print(f"maker is net-positive (+{mk['exp']*100:.3f}%) but not broad/persistent "
              f"({pos_months}/{n_months} months, {pos_syms}/{len(by_sym)} symbols) — fragile; "
              f"and remember this fill rate is optimistic (no queue model).")
    else:
        print("even as a (optimistically-filled) maker, OFI does not clear costs — the spread "
              "you save is outweighed by the favorable moves you miss. Honest dead end for retail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

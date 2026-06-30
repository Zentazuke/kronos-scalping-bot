"""cross_exchange_funding.py — backtest the CROSS-EXCHANGE funding differential.

The honest cousin of the pairs trade. Pairs died for two reasons: double fees,
AND crypto majors aren't truly cointegrated (no stable reverting spread). This
trade fixes BOTH:

  * SAME coin on two venues -> the price spread is mechanically tight (just the
    tiny inter-venue perp basis), so there's no cointegration fragility. Price
    direction cancels cleanly.
  * It's direction-free: you don't predict anything. When venue A's funding is
    higher than venue B's, you SHORT the perp on A (collect the high funding) and
    LONG the perp on B (pay the low funding), and pocket the DIFFERENTIAL every
    funding interval — net of the fee you pay to put the pair on / take it off.

Why it might beat the plain carry: the *differential* can persist even when the
absolute level is low (one venue runs hotter than another in crowded regimes),
and it's more market-neutral (a market-wide funding collapse hits both legs).

Why it might still fail (the test will tell us honestly): in calm 2026 the
differential may be as thin as the absolute carry was, and you pay fees to rotate
venues + run margin on TWO exchanges. We test net of fees, by month, brutally.

Public data, no keys. Funding history from each exchange via ccxt unified symbols.

    .venv\\Scripts\\python.exe cross_exchange_funding.py --days 180 --min-ann 6 --fee-bps 10
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

DEFAULT_EXCHANGES = ["binanceusdm", "bybit", "okx"]
DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "DOGE/USDT:USDT", "BNB/USDT:USDT", "ADA/USDT:USDT", "LINK/USDT:USDT",
    "AVAX/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT", "NEAR/USDT:USDT",
]
PERIODS_PER_DAY = 3        # 8h funding
SLOT_MS = 8 * 3_600_000    # bucket funding events into 8h slots so venues align


def _fetch(ex, symbol: str, days: int) -> List[Tuple[int, float]]:
    """Funding history as (timestamp_ms, rate) for one exchange+symbol."""
    import time
    now = int(time.time() * 1000)
    since = now - days * 86_400_000
    out: List[Tuple[int, float]] = []
    cursor = since
    while cursor < now:
        try:
            batch = ex.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        except Exception:  # noqa: BLE001
            break
        if not batch:
            break
        for b in batch:
            r = b.get("fundingRate")
            if r is not None:
                out.append((int(b["timestamp"]), float(r)))
        if len(batch) < 1000:
            break
        nxt = int(batch[-1]["timestamp"]) + 1
        if nxt <= cursor:
            break
        cursor = nxt
    return out


def _slotted(events: List[Tuple[int, float]]) -> Dict[int, float]:
    """Map each funding event to its 8h slot so different venues line up."""
    out: Dict[int, float] = {}
    for ts, r in events:
        out[(ts // SLOT_MS) * SLOT_MS] = r
    return out


def _perf(rets: List[Tuple[int, float]]) -> Dict[str, float]:
    if not rets:
        return {"n": 0, "total": 0.0, "ann": 0.0, "maxdd": 0.0, "days": 0.0}
    cum = peak = maxdd = 0.0
    for _t, r in rets:
        cum += r
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
    days = (rets[-1][0] - rets[0][0]) / 86_400_000 or 1.0
    return {"n": len(rets), "total": cum, "ann": cum * (365.0 / days),
            "maxdd": maxdd, "days": days}


def best_differential(slots: Dict[str, Dict[int, float]], coin: str,
                      venue_rates: Dict[str, float]) -> Optional[Tuple[str, str, float]]:
    """Given {exchange: rate} at one slot, return (short_venue, long_venue, diff)."""
    if len(venue_rates) < 2:
        return None
    hi = max(venue_rates, key=lambda k: venue_rates[k])  # short here (collect high funding)
    lo = min(venue_rates, key=lambda k: venue_rates[k])  # long here (pay low funding)
    return (hi, lo, venue_rates[hi] - venue_rates[lo])


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest the cross-exchange funding differential")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--min-ann", type=float, default=6.0,
                    help="only put a coin on if the venue differential annualizes above this %%")
    ap.add_argument("--fee-bps", type=float, default=10.0,
                    help="round-trip cost to open+close the two perp legs on a venue switch")
    ap.add_argument("--top-n", type=int, default=4, help="max coins held at once")
    ap.add_argument("--exchanges", default=",".join(DEFAULT_EXCHANGES))
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0

    try:
        import ccxt  # type: ignore[import-untyped]
    except ImportError:
        print("ccxt not installed — run:  .venv\\Scripts\\python.exe -m pip install ccxt")
        return 1

    ex_names = [e.strip() for e in args.exchanges.split(",") if e.strip()]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    exchanges = {}
    for name in ex_names:
        try:
            exchanges[name] = getattr(ccxt, name)({"enableRateLimit": True})
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: init failed ({str(exc)[:40]})")
    if len(exchanges) < 2:
        print("need at least 2 exchanges")
        return 1

    # slots[coin][exchange] = {slot_ts: rate}
    slots: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(dict)
    print(f"fetching {len(symbols)} coins x {len(exchanges)} venues, {args.days}d ...")
    for s in symbols:
        coin = s.split("/")[0]
        for name, ex in exchanges.items():
            ev = _fetch(ex, s, args.days)
            if ev:
                slots[coin][name] = _slotted(ev)
        venues = len(slots.get(coin, {}))
        if venues < 2:
            print(f"  {coin:<6} only {venues} venue(s) — skipped")
    coins = [c for c in slots if len(slots[c]) >= 2]
    if not coins:
        print("no coin had funding on >=2 venues — check connectivity")
        return 1

    # universe of 8h slots across everything
    timeline = sorted({t for c in coins for ex in slots[c].values() for t in ex})

    # ---- strategy: each slot, rank coins by differential, hold top-N that clear min-ann
    held: Dict[str, Tuple[str, str]] = {}   # coin -> (short_venue, long_venue)
    rets: List[Tuple[int, float]] = []
    fees_total = 0.0
    deployed = 0
    min_per_slot = args.min_ann / (PERIODS_PER_DAY * 365 * 100)

    for t in timeline:
        # what differential is available on each coin at this slot
        avail: Dict[str, Tuple[str, str, float]] = {}
        for c in coins:
            vr = {ex: slots[c][ex][t] for ex in slots[c] if t in slots[c][ex]}
            bd = best_differential(slots, c, vr)
            if bd and bd[2] >= min_per_slot:
                avail[c] = bd
        # pick top-N by differential
        picks = dict(sorted(avail.items(), key=lambda kv: -kv[1][2])[:args.top_n])

        slot_fee = 0.0
        new_held: Dict[str, Tuple[str, str]] = {}
        for c, (hi, lo, _d) in picks.items():
            prev = held.get(c)
            if prev != (hi, lo):                 # opening or switching venue legs
                slot_fee += fee
            new_held[c] = (hi, lo)
        for c in held:                            # closing coins we dropped
            if c not in picks:
                slot_fee += fee
        fees_total += slot_fee
        held = new_held

        if picks:
            gross = sum(d for (_h, _l, d) in picks.values()) / len(picks)
            deployed += 1
        else:
            gross = 0.0
        rets.append((t, gross - slot_fee))

    perf = _perf(rets)
    depl = 100.0 * deployed / len(rets) if rets else 0.0

    print(f"\n=== CROSS-EXCHANGE FUNDING DIFFERENTIAL — {', '.join(exchanges)} ===")
    print(f"{len(coins)} coins on >=2 venues, ~{perf['days']:.0f} days, "
          f"deployed {depl:.0f}% of slots, min {args.min_ann:g}% ann, {args.fee_bps:g}bps/switch\n")
    print(f"{'metric':<26}{'total ret':>11}{'annualized':>12}{'max DD':>9}")
    print(f"{'collect differential (net)':<26}{perf['total']*100:>+10.2f}%"
          f"{perf['ann']*100:>+11.1f}%{perf['maxdd']*100:>8.2f}%")
    print(f"\ntotal fees paid: {fees_total*100:.2f}% of notional over the period")

    # average raw differential available (gross, ignoring min-ann gate + fees)
    diffs: List[float] = []
    for t in timeline:
        for c in coins:
            vr = {ex: slots[c][ex][t] for ex in slots[c] if t in slots[c][ex]}
            if len(vr) >= 2:
                diffs.append(max(vr.values()) - min(vr.values()))
    if diffs:
        avg_ann = (sum(diffs) / len(diffs)) * PERIODS_PER_DAY * 365 * 100
        print(f"avg raw venue spread available: {avg_ann:+.1f}%/yr annualized "
              f"(gross, before the {args.fee_bps:g}bps/switch cost)")

    print("\nBY MONTH (net of fees):")
    by_month: Dict[str, float] = defaultdict(float)
    for t, r in rets:
        by_month[datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m")] += r
    pos = 0
    for ym in sorted(by_month):
        v = by_month[ym]
        pos += 1 if v > 0 else 0
        print(f"  {ym}   {v*100:>+7.2f}%")
    n_months = len(by_month)

    print("\n=== read ===")
    if perf["ann"] > 0 and pos >= max(2, n_months * 0.7):
        print(f"the cross-venue differential paid +{perf['ann']*100:.1f}%/yr net of fees, positive in "
              f"{pos}/{n_months} months — a REAL, direction-free, same-asset spread that survives costs. "
              f"The first market-neutral edge that cleared the wall. Worth a careful live test.")
    elif perf["ann"] > 0:
        print(f"net-positive (+{perf['ann']*100:.1f}%/yr) but lumpy ({pos}/{n_months} months) — "
              f"regime-dependent: the venue spread widens in crowded markets and vanishes in calm ones. "
              f"Tradeable only when you wait for the spread, not always-on.")
    else:
        print(f"didn't clear costs net of fees ({perf['ann']*100:+.1f}%/yr) — in calm 2026 the venue "
              f"differential is thinner than the fees+margin to harvest it, same wall as the plain carry. "
              f"Honest result; the spread is real but too small right now.")
    print("\nNOTE: returns are on NOTIONAL. Real frictions this clean test omits: margin on TWO exchanges, "
          "the inter-venue perp basis isn't exactly zero, transfer/withdrawal risk to rebalance, and "
          "liquidation risk on the short leg. Treat a green here as 'worth a small real-money probe', not size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

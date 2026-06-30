"""funding_monitor.py — the carry-trade "ok signal" detector.

Unlike every directional signal we tested, this watches a number that's REAL and
readable before you risk anything: the perpetual funding rate. When it's
positive, longs pay shorts every funding interval — so a delta-neutral
long-spot / short-perp position COLLECTS it, regardless of which way price goes.

This reads current + recent funding for your coins, annualizes it, checks whether
it's been PERSISTENTLY positive (so it's worth paying the entry costs to put the
carry on), and flags which coins are worth it right now vs. "wait".

Public data, no keys. Run it any time:

    python funding_monitor.py
    python funding_monitor.py --min-ann 12 --min-pos 80 --roundtrip-bps 18
"""

from __future__ import annotations

import argparse
from typing import List, Optional

DEFAULT_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
                   "DOGE/USDT:USDT", "BNB/USDT:USDT", "ADA/USDT:USDT", "LINK/USDT:USDT"]
PERIODS_PER_DAY = 3  # Binance funding is every 8h for most pairs (some are 4h)


def _ann(rate_per_period: float) -> float:
    return rate_per_period * PERIODS_PER_DAY * 365 * 100  # % per year


def main() -> int:
    ap = argparse.ArgumentParser(description="Funding-rate carry 'ok signal' monitor")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--min-ann", type=float, default=10.0,
                    help="min annualized funding %% to call it worth collecting")
    ap.add_argument("--min-pos", type=float, default=80.0,
                    help="min %% of the last 7d the funding was positive (persistence)")
    ap.add_argument("--roundtrip-bps", type=float, default=18.0,
                    help="round-trip cost to put on+take off both legs (bps of notional)")
    args = ap.parse_args()

    try:
        import ccxt  # type: ignore[import-untyped]
    except ImportError:
        print("ccxt not installed — run:  .venv\\Scripts\\python.exe -m pip install ccxt")
        return 1

    ex = ccxt.binanceusdm({"enableRateLimit": True})
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    print("\n=== FUNDING MONITOR — carry-trade ok signal ===")
    print("positive funding = longs pay shorts = you COLLECT via long-spot / short-perp (delta-neutral)")
    print(f"worth-it rule: annualized >= {args.min_ann:g}% AND 7d positive >= {args.min_pos:g}% "
          f"(so it clears the ~{args.roundtrip_bps:g}bps entry cost)\n")
    print(f"{'coin':<7}{'now/8h':>10}{'ann%':>8}{'7d avg':>10}{'7d ann%':>9}{'%pos':>6}   signal")

    worth: List[str] = []
    for s in symbols:
        try:
            fr = ex.fetch_funding_rate(s)
            cur: float = fr.get("fundingRate") or 0.0
            hist = ex.fetch_funding_rate_history(s, limit=21)  # ~7 days at 3/day
            rates = [h["fundingRate"] for h in hist if h.get("fundingRate") is not None]
            avg = sum(rates) / len(rates) if rates else 0.0
            pos = 100.0 * sum(1 for r in rates if r > 0) / len(rates) if rates else 0.0
        except Exception as exc:  # noqa: BLE001
            print(f"{s.split('/')[0]:<7} fetch error: {str(exc)[:50]}")
            continue
        cur_ann, avg_ann = _ann(cur), _ann(avg)
        ok = avg_ann >= args.min_ann and pos >= args.min_pos and cur > 0
        if ok:
            worth.append(s.split("/")[0])
        sig = "COLLECT ✓" if ok else ("wait (low)" if cur_ann < args.min_ann else "wait (not persistent)")
        print(f"{s.split('/')[0]:<7}{cur*100:>+9.4f}%{cur_ann:>+7.1f}%{avg*100:>+9.4f}%"
              f"{avg_ann:>+8.1f}%{pos:>5.0f}%   {sig}")

    print("\n=== read ===")
    if worth:
        print(f"Worth collecting now: {', '.join(worth)} — funding is elevated AND persistent there. "
              f"A delta-neutral carry on these would pay the annualized rate above, minus costs.")
    else:
        print("Nothing clears the bar right now — funding is low / not persistent. The honest move "
              "is to WAIT. In carry trading, patience IS the edge; you only put it on when paid enough.")
    print("Reminder: real money on BOTH legs, keep a fat margin buffer (liquidation on the short is the "
          "#1 killer), and exit if funding flips negative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

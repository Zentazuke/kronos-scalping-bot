"""funding_backtest.py — backtest the delta-neutral funding carry on REAL history.

This is the first strategy in the whole project we can validate properly, because
funding-rate HISTORY is public (unlike order books). It simulates the carry trade
honestly: each rebalance, rotate capital into the top-N perps by recent funding,
hold them delta-neutral (long spot / short perp) so price direction cancels, and
collect the funding each interval — net of the fees you pay to rotate.

Strategy knobs:
  --top-n         how many coins to hold at once (rotation breadth)
  --rebalance-days  how often you re-rank and rotate
  --min-ann       only hold coins whose trailing funding annualizes above this
  --fee-bps       round-trip cost to open+close one coin's two legs

Returns are on NOTIONAL. Real capital-on-return is lower (you fund both legs +
keep a margin buffer so the short can't liquidate) — see the note at the end.
Public data, no keys.

    python funding_backtest.py --days 180 --top-n 5 --rebalance-days 1 --min-ann 8
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

DEFAULT_UNIVERSE = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "BNB/USDT:USDT", "ADA/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT", "DOT/USDT:USDT",
    "LTC/USDT:USDT", "TRX/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "SUI/USDT:USDT", "INJ/USDT:USDT", "TIA/USDT:USDT", "SEI/USDT:USDT",
]
PERIODS_PER_DAY = 3  # 8h funding


def fetch_funding_history(ex, symbol: str, days: int) -> List[Tuple[int, float]]:
    import time
    now = int(time.time() * 1000)
    since = now - days * 86_400_000
    out: List[Tuple[int, float]] = []
    cursor = since
    while cursor < now:
        batch = ex.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        if not batch:
            break
        for b in batch:
            r = b.get("fundingRate")
            if r is not None:
                out.append((int(b["timestamp"]), float(r)))
        if len(batch) < 1000:
            break
        cursor = int(batch[-1]["timestamp"]) + 1
    return out


def _perf_from_curve(rets: List[Tuple[int, float]]) -> Dict[str, float]:
    """rets = (ts, per-interval net return). Returns cumulative/annualized/dd."""
    if not rets:
        return {"n": 0, "total": 0.0, "ann": 0.0, "maxdd": 0.0, "days": 0.0}
    cum = peak = maxdd = 0.0
    for _t, r in rets:
        cum += r
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
    days = (rets[-1][0] - rets[0][0]) / 86_400_000 or 1.0
    ann = cum * (365.0 / days)
    return {"n": len(rets), "total": cum, "ann": ann, "maxdd": maxdd, "days": days}


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest the delta-neutral funding carry")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--rebalance-days", type=float, default=1.0)
    ap.add_argument("--min-ann", type=float, default=8.0, help="min trailing annualized funding %% to hold")
    ap.add_argument("--fee-bps", type=float, default=16.0, help="round-trip cost per coin (both legs in+out)")
    ap.add_argument("--lookback-days", type=float, default=3.0, help="trailing window to rank funding")
    ap.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE))
    args = ap.parse_args()

    try:
        import ccxt  # type: ignore[import-untyped]
    except ImportError:
        print("ccxt not installed — run:  .venv\\Scripts\\python.exe -m pip install ccxt")
        return 1
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    universe = [s.strip() for s in args.universe.split(",") if s.strip()]

    # fetch funding history per coin
    hist: Dict[str, Dict[int, float]] = {}
    for s in universe:
        try:
            data = fetch_funding_history(ex, s, args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"  {s.split('/')[0]:<6} fetch error: {str(exc)[:50]}")
            continue
        if data:
            hist[s] = dict(data)
    if not hist:
        print("no funding history fetched — check connectivity")
        return 1

    timeline = sorted({t for d in hist.values() for t in d})
    fee = args.fee_bps / 10000.0
    reb_ms = args.rebalance_days * 86_400_000
    look_ms = args.lookback_days * 86_400_000

    def trailing(coin: str, t: int) -> Optional[float]:
        rs = [r for ts, r in hist[coin].items() if t - look_ms <= ts < t]
        return sum(rs) / len(rs) if rs else None

    holdings: List[str] = []
    last_reb = None
    rets: List[Tuple[int, float]] = []
    deployed_intervals = 0
    fees_total = 0.0

    for t in timeline:
        if last_reb is None or (t - last_reb) >= reb_ms:
            ranked = sorted(
                ((c, trailing(c, t)) for c in hist),
                key=lambda kv: (kv[1] is not None, kv[1] or -9), reverse=True)
            picks = [c for c, sc in ranked
                     if sc is not None and sc * PERIODS_PER_DAY * 365 * 100 >= args.min_ann][:args.top_n]
            changes = set(holdings) ^ set(picks)
            fees_total += len(changes) * (fee / 2)  # each add or drop is one leg-pair open/close
            # charge the fee for this rebalance, spread onto this interval's return
            reb_fee = len(changes) * (fee / 2)
            holdings = picks
            last_reb = t
        else:
            reb_fee = 0.0
        held = [c for c in holdings if t in hist[c]]
        if held:
            gross = sum(hist[c][t] for c in held) / len(held)
            deployed_intervals += 1
        else:
            gross = 0.0
        rets.append((t, gross - reb_fee))

    # baselines
    btc = "BTC/USDT:USDT"
    btc_rets = [(t, hist[btc][t]) for t in timeline if btc in hist and t in hist[btc]] if btc in hist else []
    allcoin_rets = []
    for t in timeline:
        v = [hist[c][t] for c in hist if t in hist[c]]
        if v:
            allcoin_rets.append((t, sum(v) / len(v)))

    strat = _perf_from_curve(rets)
    btcp = _perf_from_curve(btc_rets)
    allp = _perf_from_curve(allcoin_rets)
    depl = 100.0 * deployed_intervals / len(rets) if rets else 0.0

    print(f"\n=== FUNDING CARRY BACKTEST — top {args.top_n} by funding, rebalance {args.rebalance_days:g}d, "
          f"min {args.min_ann:g}% ann, {args.fee_bps:g}bps/coin ===")
    print(f"{len(hist)} coins, ~{strat['days']:.0f} days of real funding history, "
          f"deployed {depl:.0f}% of the time\n")
    print(f"{'strategy':<26}{'total ret':>10}{'annualized':>12}{'max DD':>9}")
    print(f"{'rotate top-N (net fees)':<26}{strat['total']*100:>+9.2f}%{strat['ann']*100:>+11.1f}%{strat['maxdd']*100:>8.2f}%")
    print(f"{'hold ALL equal (gross)':<26}{allp['total']*100:>+9.2f}%{allp['ann']*100:>+11.1f}%{allp['maxdd']*100:>8.2f}%")
    print(f"{'hold BTC only (gross)':<26}{btcp['total']*100:>+9.2f}%{btcp['ann']*100:>+11.1f}%{btcp['maxdd']*100:>8.2f}%")
    print(f"\ntotal fees paid: {fees_total*100:.2f}% of notional over the period")

    # by month
    print("\nTOP-N STRATEGY BY MONTH (net of fees):")
    by_month: Dict[str, float] = defaultdict(float)
    for t, r in rets:
        by_month[datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m")] += r
    pos = 0
    for ym in sorted(by_month):
        v = by_month[ym]
        pos += 1 if v > 0 else 0
        print(f"  {ym}   {v*100:>+7.2f}%")

    print("\n=== read ===")
    n_months = len(by_month)
    if strat["ann"] > 0 and pos >= max(2, n_months * 0.7):
        print(f"the carry paid +{strat['ann']*100:.1f}%/yr (on notional) net of fees, positive in "
              f"{pos}/{n_months} months — a REAL, mechanical yield with no direction bet. "
              f"This is the first strategy that worked on real history.")
    elif strat["ann"] > 0:
        print(f"net-positive (+{strat['ann']*100:.1f}%/yr on notional) but lumpy "
              f"({pos}/{n_months} months) — regime-dependent (rich in crowded markets, flat otherwise).")
    else:
        print(f"the carry didn't pay net of fees over this window ({strat['ann']*100:+.1f}%/yr) — "
              f"funding was too low / your rotation churned too much. Try fewer rebalances or a higher --min-ann.")
    print("\nNOTE: returns are on NOTIONAL. Real return-on-capital is lower — you fund both legs and keep a "
          "margin buffer so the short can't liquidate (call it ~half). And this assumes clean fills; that's "
          "exactly where good execution earns its keep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

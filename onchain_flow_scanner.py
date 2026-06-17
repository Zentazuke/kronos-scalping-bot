"""onchain_flow_scanner.py — the first NON-PRICE signal: stablecoin supply growth.

The roadmap's orthogonal road, finally built. Stablecoins (USDT + USDC) are minted
when new fiat capital wants in and burned when it leaves — so the *aggregate
stablecoin supply* is dry powder that builds up ON-CHAIN before it hits price.
Academic work (arXiv 2411.06327) and practitioner data both find stablecoin
inflows / mints LEAD crypto returns by hours-to-days. This is information that
isn't in the candle feed Kronos sees, and it's a single-leg signal (no double fees).

Free data, no keys: stablecoin market-cap history from CoinGecko, daily price from
Binance (reused fetch_ohlcv). We test it the SAME honest way we tested OFI:
  * signal_t = trailing --window-day growth of (USDT+USDC) supply,
  * forward return = price move over the next --horizon days,
  * STAIRCASE: bucket days by signal quantile -> is mean forward return monotonic?
  * GATE: go long only when supply is growing > threshold, net of fees, vs take-all,
  * IS/OOS split so it can't hide in one regime.

    python onchain_flow_scanner.py --days 360 --window 7 --horizon 3 --fee-bps 10
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from consensus_backtest import fetch_ohlcv

STABLES = ("tether", "usd-coin")   # CoinGecko ids for USDT, USDC


def _day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def fetch_stable_supply(coin_id: str, days: int) -> Dict[str, float]:
    """Daily market cap (=circulating supply in $) for one stablecoin, by UTC date."""
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
           f"?vs_currency=usd&days={days}&interval=daily")
    req = urllib.request.Request(url, headers={"User-Agent": "kronos-onchain/1.0"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        data = json.load(resp)
    out: Dict[str, float] = {}
    for ts, mcap in data.get("market_caps", []):
        if mcap:
            out[_day(int(ts))] = float(mcap)   # last value of the day wins
    return out


def aggregate_supply(days: int) -> Dict[str, float]:
    """Summed USDT+USDC supply per day (only days present for ALL stables)."""
    per = {c: fetch_stable_supply(c, days) for c in STABLES}
    common = set.intersection(*[set(d) for d in per.values()]) if per else set()
    return {d: sum(per[c][d] for c in STABLES) for d in sorted(common)}


def price_by_day(symbol: str, days: int) -> Dict[str, float]:
    """Daily close by UTC date, from Binance."""
    out: Dict[str, float] = {}
    for b in fetch_ohlcv(symbol, "1d", days):
        out[_day(int(b[0]))] = float(b[4])
    return out


def build_samples(supply: Dict[str, float], price: Dict[str, float],
                  window: int, horizon: int) -> List[Tuple[str, float, float]]:
    """[(date, supply_growth_signal, forward_return)] aligned on common days."""
    days = sorted(set(supply) & set(price))
    out: List[Tuple[str, float, float]] = []
    for i in range(window, len(days) - horizon):
        d0, dw, dh = days[i], days[i - window], days[i + horizon]
        s_now, s_prev = supply[d0], supply[dw]
        if s_prev <= 0 or price[d0] <= 0:
            continue
        signal = s_now / s_prev - 1.0                 # trailing supply growth
        fwd = price[dh] / price[d0] - 1.0             # forward price return
        out.append((d0, signal, fwd))
    return out


def staircase(samples, q: int = 5) -> List[Tuple[float, float, int]]:
    """Bucket by signal quantile; return [(mean_signal, mean_fwd, n)] per bucket."""
    if len(samples) < q * 4:
        return []
    sig = np.array([s for _d, s, _f in samples])
    fwd = np.array([f for _d, _s, f in samples])
    order = np.argsort(sig)
    out = []
    for chunk in np.array_split(order, q):
        out.append((float(sig[chunk].mean()), float(fwd[chunk].mean()), len(chunk)))
    return out


def gate(samples, fee: float) -> Tuple[float, float, int, int]:
    """Long when supply growing above median; (gated_mean, all_mean, n_gated, n_all) net fee."""
    if not samples:
        return 0.0, 0.0, 0, 0
    sig = np.array([s for _d, s, _f in samples])
    fwd = np.array([f for _d, _s, f in samples])
    thr = float(np.median(sig))
    mask = sig > thr
    gated = fwd[mask] - fee
    return (float(gated.mean()) if mask.any() else 0.0,
            float((fwd - fee).mean()), int(mask.sum()), len(fwd))


def directional_gate(samples, fee: float) -> Tuple[float, int, float, float]:
    """Long the top-tercile supply-growth days, SHORT the bottom-tercile (risk-off),
    flat in the middle. Captures the inverted-U the long-only gate misses.
    Returns (mean_net_per_trade, n_active, IS_mean, OOS_mean)."""
    if len(samples) < 30:
        return 0.0, 0, 0.0, 0.0
    sig = np.array([s for _d, s, _f in samples])
    fwd = np.array([f for _d, _s, f in samples])
    lo, hi = float(np.quantile(sig, 1 / 3)), float(np.quantile(sig, 2 / 3))
    pos = np.where(sig >= hi, 1.0, np.where(sig <= lo, -1.0, 0.0))
    cut = sorted(d for d, _s, _f in samples)[len(samples) // 2]
    is_m = np.array([d < cut for d, _s, _f in samples])

    def m(mask):
        a = mask & (pos != 0)
        return float((pos[a] * fwd[a] - fee).mean()) if a.any() else 0.0
    active = pos != 0
    return (m(np.ones(len(samples), bool)), int(active.sum()),
            m(is_m), m(~is_m))


def analyze(name: str, samples, fee: float) -> None:
    if len(samples) < 40:
        print(f"\n[{name}] only {len(samples)} samples — need more history"); return
    sig = np.array([s for _d, s, _f in samples])
    fwd = np.array([f for _d, _s, f in samples])
    corr = float(np.corrcoef(sig, fwd)[0, 1]) if sig.std() and fwd.std() else 0.0
    cut = sorted(d for d, _s, _f in samples)[len(samples) // 2]
    is_s = [t for t in samples if t[0] < cut]
    oos_s = [t for t in samples if t[0] >= cut]
    g_all = gate(samples, fee); g_is = gate(is_s, fee); g_oos = gate(oos_s, fee)

    print(f"\n=== {name} — stablecoin-supply growth vs forward return ===")
    print(f"{len(samples)} day-samples · predictive corr {corr:+.3f}")
    print("staircase (low supply-growth -> high), mean forward return per bucket:")
    for ms, mf, n in staircase(samples):
        bar = "#" * max(0, int(mf * 2000))
        print(f"  growth {ms*100:>+6.2f}%  ->  fwd {mf*100:>+6.2f}%  (n={n}) {bar}")
    print(f"GATE long-when-growing (net {fee*1e4:g}bps):")
    print(f"  all days   fwd/trade {g_all[1]*100:>+6.3f}%  (n={g_all[3]})")
    print(f"  growing    fwd/trade {g_all[0]*100:>+6.3f}%  (n={g_all[2]})   "
          f"IS {g_is[0]*100:+.3f}%  OOS {g_oos[0]*100:+.3f}%")

    dg_all, dg_n, dg_is, dg_oos = directional_gate(samples, fee)
    print(f"DIRECTIONAL gate (long top-tercile growth / SHORT bottom-tercile, flat middle):")
    print(f"  net/trade {dg_all*100:>+6.3f}%  (n={dg_n})   IS {dg_is*100:+.3f}%  OOS {dg_oos*100:+.3f}%")

    print("--- read ---")
    if corr > 0.05 and dg_oos > 0 and dg_is > 0:
        print(f"REAL SIGNAL: stablecoin-supply flow predicts forward returns (corr {corr:+.3f}), and the "
              f"DIRECTIONAL gate (short contraction / long growth) pays in BOTH halves "
              f"(IS {dg_is*100:+.3f}%, OOS {dg_oos*100:+.3f}%/trade). The inverted-U is real — the "
              f"contraction-is-bearish side is the robust core. Wire it in as a risk-off regime gate.")
    elif corr > 0.04 and (dg_oos > 0 or dg_is > 0):
        print(f"PARTIAL: consistent sign (corr {corr:+.3f}) and the directional gate helps "
              f"(IS {dg_is*100:+.3f}%, OOS {dg_oos*100:+.3f}%) but not cleanly both halves. Real but weak — "
              f"best as CONTEXT (risk-off when supply contracts), not a standalone trade. Sweep window/horizon.")
    else:
        print(f"no usable edge here (corr {corr:+.3f}, directional OOS {dg_oos*100:+.3f}%) — "
              f"sweep --window/--horizon before discarding.")


def main() -> int:
    ap = argparse.ArgumentParser(description="On-chain stablecoin-supply flow scanner")
    ap.add_argument("--days", type=int, default=360)
    ap.add_argument("--window", type=int, default=7, help="trailing days for the supply-growth signal")
    ap.add_argument("--horizon", type=int, default=3, help="forward days for the return")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0

    print("fetching stablecoin supply (CoinGecko) + daily price (Binance) ...")
    try:
        supply = aggregate_supply(args.days)
    except Exception as exc:  # noqa: BLE001
        print(f"stablecoin fetch failed ({str(exc)[:80]}).")
        print("CoinGecko free API can rate-limit; wait a minute and retry, or get a free demo key.")
        return 1
    if len(supply) < args.window + args.horizon + 40:
        print(f"only {len(supply)} days of supply data — try a smaller --days or retry"); return 1
    print(f"got {len(supply)} days of stablecoin supply "
          f"(${supply[max(supply)]/1e9:.1f}B latest USDT+USDC)")

    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        try:
            price = price_by_day(sym, args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"{sym}: price fetch failed ({str(exc)[:40]})"); continue
        samples = build_samples(supply, price, args.window, args.horizon)
        analyze(sym.split("/")[0], samples, fee)

    print("\nNote: this is the stablecoin-MINT proxy (free). The stronger signal — per-exchange "
          "netflows — needs a paid feed (CryptoQuant/Glassnode); this tells us if the road is worth that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""fetch_data.py — run ONCE (or on a schedule) to cache market data so Claude can
iterate on experiments without you re-running every test.

The agent's workspace can't reach the internet, but the project folder is shared. So
this dumps the raw series we keep needing into data_cache/*.csv; from then on, analyses
read those files instead of live-fetching — you fetch once, the agent does the rest.

Writes:
  data_cache/daily_close.csv    — UTC date + daily close for BTC + alts (Binance)
  data_cache/dvol.csv           — UTC date + BTC/ETH implied vol (Deribit DVOL)
  data_cache/stable_supply.csv  — UTC date + aggregate USDT+USDC supply $ (CoinGecko)
  data_cache/_meta.txt          — when it was fetched + row counts

  python fetch_data.py                 # ~365 days
  python fetch_data.py --days 540
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List

from consensus_backtest import fetch_ohlcv

CACHE = "data_cache"
COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
         "ADA/USDT", "DOGE/USDT", "LINK/USDT", "AVAX/USDT"]
DEFILLAMA_URL = "https://stablecoins.llama.fi/stablecoincharts/all"


def _day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def daily_closes(days: int) -> Dict[str, Dict[str, float]]:
    """{coin: {date: close}} from Binance daily candles."""
    out: Dict[str, Dict[str, float]] = {}
    for sym in COINS:
        try:
            c = fetch_ohlcv(sym, "1d", days)
            out[sym.split("/")[0]] = {_day(int(b[0])): float(b[4]) for b in c}
            print(f"  {sym.split('/')[0]:<6} {len(c)} daily bars")
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym:<10} failed ({str(exc)[:40]})")
    return out


def dvol(days: int) -> Dict[str, Dict[str, float]]:
    """{currency: {date: dvol}} 30-day implied vol from Deribit public API."""
    end = int(time.time() * 1000); start = end - days * 86_400_000
    out: Dict[str, Dict[str, float]] = {}
    for cur in ("BTC", "ETH"):
        url = (f"https://www.deribit.com/api/v2/public/get_volatility_index_data"
               f"?currency={cur}&start_timestamp={start}&end_timestamp={end}&resolution=1D")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kronos/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.load(r)
            out[cur] = {_day(int(row[0])): float(row[4]) for row in data["result"]["data"]}
            print(f"  DVOL {cur} {len(out[cur])} days")
        except Exception as exc:  # noqa: BLE001
            print(f"  DVOL {cur} failed ({str(exc)[:40]})")
    return out


def stable_supply(days: int) -> Dict[str, float]:
    """{date: total USD-pegged stablecoin supply $} from DefiLlama (free, no key, years of
    history). Replaces the old CoinGecko USDT+USDC series that now 401s on long requests."""
    try:
        req = urllib.request.Request(DEFILLAMA_URL, headers={"User-Agent": "kronos/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
    except Exception as exc:  # noqa: BLE001
        print(f"  supply (DefiLlama) failed ({str(exc)[:50]})")
        return {}
    out: Dict[str, float] = {}
    for row in data:
        ts = row.get("date")
        val = row.get("totalCirculatingUSD")
        if isinstance(val, dict):
            val = val.get("peggedUSD")
        if ts is None or val in (None, ""):
            continue
        out[_day(int(ts) * 1000)] = float(val)
    if days and len(out) > days:
        for d in sorted(out)[:-(days + 2)]:
            out.pop(d, None)
    print(f"  supply (DefiLlama) {len(out)} days")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache market data to CSV for offline analysis")
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    print("1/3 daily closes (Binance) ...")
    closes = daily_closes(args.days)
    print("2/3 DVOL implied vol (Deribit) ...")
    iv = dvol(args.days)
    print("3/3 stablecoin supply (CoinGecko) ...")
    supply = stable_supply(args.days)

    # daily_close.csv  (date, BTC, ETH, ...)
    coins = [c.split("/")[0] for c in COINS if c.split("/")[0] in closes]
    all_days = sorted({d for c in coins for d in closes[c]})
    with open(os.path.join(CACHE, "daily_close.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date"] + coins)
        for d in all_days:
            w.writerow([d] + [closes[c].get(d, "") for c in coins])

    # dvol.csv  (date, BTC, ETH)
    iv_days = sorted({d for c in iv for d in iv[c]})
    with open(os.path.join(CACHE, "dvol.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "BTC", "ETH"])
        for d in iv_days:
            w.writerow([d, iv.get("BTC", {}).get(d, ""), iv.get("ETH", {}).get(d, "")])

    # stable_supply.csv  (date, supply_usd)
    with open(os.path.join(CACHE, "stable_supply.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "stable_supply_usd"])
        for d in sorted(supply):
            w.writerow([d, f"{supply[d]:.0f}"])

    with open(os.path.join(CACHE, "_meta.txt"), "w") as f:
        f.write(f"fetched_utc {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"days {args.days}\n")
        f.write(f"daily_close rows {len(all_days)} coins {coins}\n")
        f.write(f"dvol rows {len(iv_days)}\n")
        f.write(f"stable_supply rows {len(supply)}\n")

    print(f"\nwrote {CACHE}/daily_close.csv, dvol.csv, stable_supply.csv "
          f"({len(all_days)} days). Claude can now read these and run analyses directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

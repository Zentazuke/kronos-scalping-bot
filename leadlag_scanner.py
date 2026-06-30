"""leadlag_scanner.py — the BTC-residual catch-up scalp (a genuinely new road).

Lead-lag in crypto is real but the *raw* tick lag is ~15 seconds — an HFT game we
can't win. This tests the slower, retail-reachable cousin: the BTC-RESIDUAL catch-up.

Idea: in any short bar, an alt's move = (its beta x BTC's move) + its own residual.
BTC and alts move together, but each alt over/under-shoots BTC by some amount. When an
alt LAGS BTC's move (negative residual — it didn't keep up), does it CATCH UP next bar?
That's not chasing the 15s tick lag; it's trading the dispersion between an alt and its
BTC-implied move, which lives at the minutes scale where retail can actually act.

Honest method (per coin, causal — no lookahead):
  * trailing-window beta of alt returns on BTC returns,
  * residual[t] = alt_ret[t] - beta * btc_ret[t]   (how much the alt over/under-shot),
  * DIAGNOSTICS: corr(btc_ret[t], alt_ret[t+1])   = raw BTC lead (probably ~0, it's too fast)
                 corr(residual[t], alt_ret[t+1])  = catch-up/reversal (negative => laggard rebounds)
  * SCALP: when |residual| is in its top quantile, take the alt NEXT bar in the
    catch-up direction (-sign(residual)); exit one bar later; net of round-trip fees.
  * IS/OOS split so it can't hide in one stretch.

    python leadlag_scanner.py --timeframe 5m --days 45 --fee-bps 10
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np

from consensus_backtest import fetch_ohlcv

BASE = "BTC/USDT"
DEFAULT_ALTS = "ETH/USDT,SOL/USDT,XRP/USDT,BNB/USDT,ADA/USDT,DOGE/USDT,LINK/USDT,AVAX/USDT"


def returns_by_ts(symbol: str, timeframe: str, days: int) -> Dict[int, float]:
    """Per-bar close-to-close return keyed by bar timestamp."""
    c = fetch_ohlcv(symbol, timeframe, days)
    out: Dict[int, float] = {}
    for i in range(1, len(c)):
        p0, p1 = c[i - 1][4], c[i][4]
        if p0 > 0:
            out[int(c[i][0])] = p1 / p0 - 1.0
    return out


def analyze(coin: str, btc: Dict[int, float], alt: Dict[int, float], *,
            beta_window: int, fee: float, q: float) -> None:
    ts = sorted(set(btc) & set(alt))
    if len(ts) < beta_window + 200:
        print(f"[{coin}] only {len(ts)} aligned bars — need more"); return
    b = np.array([btc[t] for t in ts])
    a = np.array([alt[t] for t in ts])

    # causal trailing beta + residual
    resid = np.full(len(ts), np.nan)
    for i in range(beta_window, len(ts)):
        bw, aw = b[i - beta_window:i], a[i - beta_window:i]
        var = float(bw @ bw)
        beta = float(bw @ aw) / var if var > 0 else 1.0
        resid[i] = a[i] - beta * b[i]
    valid = ~np.isnan(resid)

    # diagnostics on the next-bar alt return
    nxt = np.roll(a, -1)
    use = valid.copy(); use[-1] = False                      # drop last (no next bar)
    raw_lead = float(np.corrcoef(b[use], nxt[use])[0, 1])     # BTC[t] -> alt[t+1]
    catchup = float(np.corrcoef(resid[use], nxt[use])[0, 1])  # residual[t] -> alt[t+1]

    # scalp: big |residual| -> trade next bar in catch-up direction (-sign(residual))
    rv = resid[use]; nv = nxt[use]
    thr = float(np.quantile(np.abs(rv), q))
    big = np.abs(rv) >= thr
    pos = -np.sign(rv[big])                                   # laggard (neg resid) -> long
    rets = pos * nv[big] - fee
    n = int(big.sum())
    half = n // 2
    is_r, oos_r = rets[:half], rets[half:]

    print(f"\n=== {coin} vs BTC — {len(ts)} bars ===")
    print(f"  raw BTC lead   corr(btc[t], alt[t+1])   = {raw_lead:+.4f}   (near 0 = lag too fast for us)")
    print(f"  catch-up       corr(resid[t], alt[t+1]) = {catchup:+.4f}   (negative = laggards rebound)")
    if n:
        print(f"  SCALP top-{(1-q)*100:.0f}% |residual|, catch-up dir, next bar, net {fee*1e4:g}bps:")
        print(f"    n={n}  win {100*(rets>0).mean():.1f}%  net/trade {rets.mean()*100:+.4f}%  "
              f"total {rets.sum()*100:+.1f}%")
        print(f"    IS net/trade {is_r.mean()*100:+.4f}%   OOS net/trade {oos_r.mean()*100:+.4f}%")
    return None if not n else (catchup, rets.mean(), oos_r.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="BTC-residual lead-lag catch-up scanner")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--beta-window", type=int, default=500, help="trailing bars for the BTC beta")
    ap.add_argument("--q", type=float, default=0.9, help="trade only the top (1-q) |residual| bars")
    ap.add_argument("--fee-bps", type=float, default=10.0, help="round-trip cost per scalp")
    ap.add_argument("--alts", default=DEFAULT_ALTS)
    args = ap.parse_args()
    fee = args.fee_bps / 10000.0

    print(f"fetching {args.timeframe} returns for BTC + alts ({args.days}d) ...")
    btc = returns_by_ts(BASE, args.timeframe, args.days)
    if len(btc) < args.beta_window + 200:
        print(f"only {len(btc)} BTC bars — increase --days"); return 1

    results = []
    for sym in [s.strip() for s in args.alts.split(",") if s.strip()]:
        try:
            alt = returns_by_ts(sym, args.timeframe, args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"{sym}: fetch failed ({str(exc)[:40]})"); continue
        r = analyze(sym.split("/")[0], btc, alt, beta_window=args.beta_window, fee=fee, q=args.q)
        if r:
            results.append(r)

    print("\n=== read ===")
    if not results:
        print("no usable results — increase --days"); return 0
    catchups = [c for c, _e, _o in results]
    oos = [o for _c, _e, o in results]
    neg_catch = sum(1 for c in catchups if c < -0.02)
    oos_pos = sum(1 for o in oos if o > 0)
    avg_oos = float(np.mean(oos))
    if neg_catch >= 0.6 * len(results) and oos_pos >= 0.6 * len(results) and avg_oos > 0:
        print(f"REAL: the catch-up correlation is negative on {neg_catch}/{len(results)} coins (laggards "
              f"rebound), and the scalp is OOS-positive on {oos_pos}/{len(results)} (avg OOS "
              f"{avg_oos*100:+.4f}%/trade). A retail-reachable lead-lag edge — worth a maker-fee / "
              f"forward-test look. Mind: 5m scalps are fee-sensitive; check at lower --fee-bps too.")
    elif neg_catch >= 0.6 * len(results):
        print(f"SIGNAL but COST-WALLED: laggards do rebound (catch-up corr negative on {neg_catch}/"
              f"{len(results)}), but net of {args.fee_bps:g}bps the scalp doesn't clear OOS "
              f"({oos_pos}/{len(results)} positive). Classic: real micro-edge, eaten by taker fees. "
              f"Re-run at --fee-bps 2 (maker) — if it clears there, it's a MAKER strategy.")
    else:
        print(f"no usable catch-up edge ({neg_catch}/{len(results)} coins show rebound, OOS+ "
              f"{oos_pos}/{len(results)}) — the residual doesn't predict next-bar cleanly at this "
              f"timeframe. Try --timeframe 1m or 15m, or a different --beta-window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

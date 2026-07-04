"""costs.py — THE single source of truth for trading frictions.

Why this exists (2026-07-03 audit): fee assumptions had drifted across scripts
(7.5 / 10 / 15 / 20 bps in different places), and the whole project's core lesson
is that the toll decides everything. Every backtest, forward test, and live
scorecard should import these numbers instead of hard-coding its own.

Also the "first cell of every new strategy notebook": run this BEFORE building
anything to see the edge a strategy must clear just to break even.

    python costs.py                    # breakeven table at the default cost
    python costs.py --rt-bps 15        # e.g. BNB-discount taker round trip
"""
from __future__ import annotations

import argparse

# ---- Binance spot fee schedule (default tier) --------------------------------
SPOT_TAKER_BPS = 10.0        # 0.10% per fill
SPOT_MAKER_BPS = 10.0        # 0.10% per fill (LIMIT_MAKER)
BNB_DISCOUNT = 0.25          # 25% off when paying fees in BNB
SLIPPAGE_BPS = 2.0           # per-fill haircut (queue position / adverse selection)


def taker_bps(bnb: bool = False) -> float:
    return SPOT_TAKER_BPS * (1 - BNB_DISCOUNT if bnb else 1.0)


def maker_bps(bnb: bool = False) -> float:
    return SPOT_MAKER_BPS * (1 - BNB_DISCOUNT if bnb else 1.0)


def round_trip_bps(entry: str = "taker", exit_: str = "taker",
                   bnb: bool = False, slippage: bool = True) -> float:
    """Total friction for one complete trade, in bps."""
    legs = {"taker": taker_bps(bnb), "maker": maker_bps(bnb)}
    total = legs[entry] + legs[exit_]
    if slippage:
        total += 2 * SLIPPAGE_BPS
    return total


def breakeven_winrate(rr: float, rt_bps: float, risk_bps: float) -> float:
    """Win rate needed for zero expectancy: trade risks `risk_bps`, wins pay
    rr*risk_bps, and every trade pays rt_bps of friction."""
    # w*rr*R - (1-w)*R - c = 0  ->  w = (R + c) / (R*(1+rr))
    return (risk_bps + rt_bps) / (risk_bps * (1 + rr))


def main() -> int:
    ap = argparse.ArgumentParser(description="Friction constants + breakeven table")
    ap.add_argument("--rt-bps", type=float, default=None,
                    help="override the round-trip cost (default: taker/taker, no BNB, +slip)")
    args = ap.parse_args()
    rt = args.rt_bps if args.rt_bps is not None else round_trip_bps()
    print("=== frictions (Binance spot) ===")
    print(f"  taker {SPOT_TAKER_BPS:g} bps/fill ({taker_bps(True):g} w/ BNB) · "
          f"maker {SPOT_MAKER_BPS:g} ({maker_bps(True):g} w/ BNB) · slip {SLIPPAGE_BPS:g}/fill")
    print(f"  round trips: taker/taker {round_trip_bps():g} · maker/maker {round_trip_bps('maker','maker'):g} · "
          f"taker/taker+BNB {round_trip_bps(bnb=True):g}\n")
    print(f"=== breakeven win rate at {rt:g} bps round-trip ===")
    print(f"  {'risk/trade':>12} | " + " | ".join(f"RR {r:.1f}" for r in (0.5, 1.0, 1.5, 2.0)))
    for risk in (25.0, 50.0, 100.0, 200.0, 400.0):
        cells = " | ".join(f"{breakeven_winrate(r, rt, risk)*100:5.1f}%" for r in (0.5, 1.0, 1.5, 2.0))
        print(f"  {risk:>9.0f}bps | {cells}")
    print("\nRead: if your strategy's plausible win rate is below the cell for its")
    print("risk/RR, it CANNOT make money here — don't build it. (This table is the")
    print("cost wall that killed the entire fast-scalp family; check it FIRST.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

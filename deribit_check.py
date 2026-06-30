"""deribit_check.py — read-only sanity check for the Deribit TESTNET API key.

Confirms two things the VRP harvest needs, with a key that CAN'T trade:
  1. the key authenticates  (Account=read)  -> we can see the account
  2. we can pull live BTC option-chain data (mark price + implied vol by strike)

Make the key on test.deribit.com with Account=read, everything else none.
Put the pair in .env:  DERIBIT_CLIENT_ID=...   DERIBIT_CLIENT_SECRET=...

    python deribit_check.py
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:  # noqa: BLE001
    pass

import ccxt  # type: ignore[import-untyped]


def main() -> int:
    cid, sec = os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET")
    if not (cid and sec):
        print("Missing DERIBIT_CLIENT_ID / DERIBIT_CLIENT_SECRET in .env"); return 1

    ex = ccxt.deribit({"apiKey": cid, "secret": sec, "enableRateLimit": True})
    ex.set_sandbox_mode(True)   # test.deribit.com — fake money, no real funds involved
    print("=== Deribit TESTNET — read-only check ===")

    # 1) AUTH — fetch balance (proves Account=read works)
    try:
        bal = ex.fetch_balance()
        nonzero = {k: v for k, v in (bal.get("total") or {}).items() if v}
        print("AUTH OK — testnet balances:", nonzero or "(empty — normal for a fresh testnet account)")
    except Exception as exc:  # noqa: BLE001
        print("AUTH FAILED:", str(exc)[:200])
        print("  -> check the client_id/secret in .env and that the key is a TESTNET key (test.deribit.com).")
        return 1

    # 2) MARKET DATA — load the BTC option chain (public, but proves the pipe works)
    try:
        ex.load_markets()
        opts = [m for m in ex.markets.values()
                if m.get("option") and m.get("base") == "BTC" and m.get("active")]
        if not opts:
            print("no active BTC options found (unexpected)"); return 1
        idx = float(ex.fetch_ticker("BTC-PERPETUAL")["last"])
        print(f"\nmarket data OK — {len(opts)} active BTC option instruments · BTC ~ ${idx:,.0f}")

        # nearest expiry at least ~5 days out
        now = datetime.now(timezone.utc).timestamp() * 1000
        exps = sorted({m["expiry"] for m in opts if m.get("expiry") and m["expiry"] > now + 5 * 86400_000})
        if not exps:
            print("(no expiry >5d out to sample)"); return 0
        exp = exps[0]
        exp_str = datetime.fromtimestamp(exp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        chain = [m for m in opts if m.get("expiry") == exp]
        # ATM-ish strike
        atm = min(chain, key=lambda m: abs((m.get("strike") or 0) - idx))["strike"]
        print(f"\nsampling the {exp_str} expiry, strike nearest ATM = ${atm:,.0f}:")
        for typ in ("call", "put"):
            row = next((m for m in chain if m.get("strike") == atm and m.get("optionType") == typ), None)
            if not row:
                continue
            try:
                t = ex.fetch_ticker(row["symbol"])
                info = t.get("info") or {}
                mark = t.get("mark") or info.get("mark_price")
                iv = info.get("mark_iv") or info.get("iv")
                print(f"  {typ.upper():<4} {row['symbol']:<28} mark {mark}  mark_IV {iv}%")
            except Exception as exc:  # noqa: BLE001
                print(f"  {typ}: ticker read failed ({str(exc)[:60]})")
        print("\nALL GOOD — the key reads the account and the live option chain (implied vol by strike).")
        print("That's everything the VRP backtest + paper-trade needs. Next: build the defined-risk structure.")
    except Exception as exc:  # noqa: BLE001
        print("market data error:", str(exc)[:200]); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

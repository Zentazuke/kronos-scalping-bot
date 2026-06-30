# 👉 PICK UP HERE — next session (paused 2026-06-29)

## Where we are: testing the Crypto-Analyst as a STANDALONE SWING strategy

We exhausted using the analyst as an intraday *gate* (both the 1D lean and the 4h
trend-label gate looked great — OOS Sharpe ~1.0 — but **both failed the placebo test**:
the lift was just trade-count reduction + one outlier month, p=0.18 and p=0.36).

Then we flipped to the right idea (Ricardo's): **trade the analyst's own conviction
signal directly** — long when strong-positive, short when strong-negative, flatten on
reversal. Slow, full-period, holds for days → the structural category that can clear costs.

### First swing result (1D, BTC+ETH only) — MIXED, the most alive result yet:
- **ETH**: OOS Sharpe **1.04**, beats buy&hold (0.05), placebo **3/50 (p≈0.06)** — borderline real.
- **BTC**: noise (full-sample negative, placebo 8/50).
- ⚠️ Three caveats: (1) borderline not decisive; (2) the passing coin (ETH) is the WRONG
  one — the model card said BTC had the 1D edge, not ETH → smells like luck; (3) ~139%
  drawdowns (no sizing yet).

## ✅ THE NEXT STEP: the breadth test (decisive)
A real edge works across MANY coins; a fluke is one coin. Run the swing on the whole basket.

**1. Export all 7 coins (slow, ~15-20 min — has the ensemble conviction):**
```
cd crypto-analyst
.venv\Scripts\python.exe scripts\export_lean.py --symbols BTC_USDT,ETH_USDT,SOL_USDT,XRP_USDT,DOGE_USDT,LINK_USDT,AVAX_USDT --days 1900 --out analyst_lean.csv
copy analyst_lean.csv "..\Trading Bot\data_cache\analyst_lean.csv"
```
**2. Run the swing across all 7 (Trading Bot):**
```
.venv\Scripts\python.exe analyst_swing_backtest.py --lean-csv data_cache\analyst_lean.csv --gate-tf 1D --symbols BTC_USDT,ETH_USDT,SOL_USDT,XRP_USDT,DOGE_USDT,LINK_USDT,AVAX_USDT --signal normalized --enter 0.3 --exit 0.1 --placebo 50
```
**3. Read the AGGREGATE line + verdict:**
- 5+/7 coins pass cleanly (OOS>0, beat B&H, placebo ≤10%) & mean OOS>0 → **real broad edge**,
  worth vol-targeting + forward-testing. This would be the FIRST thing to survive breadth+placebo.
- Only ETH passes → ETH was luck; record it and move on.

## Honesty bar (unchanged): judge on OOS + placebo + breadth, never the in-sample total.
No parameter-sweeping to chase a green. If breadth fails, it fails.

## Status of everything else
- Intraday-TSM: alive only as a LIVE testnet forward test (both backtest angles failed placebo).
- Live trader: fixed (executes the trial's committed decisions; one brain, one executor).
- Postmortem doc (Failed_Strategies_Postmortem.md/.pdf) is current through stablecoin-flow.
  STILL TO DO: fold in the analyst gate negative results (1D + 4h) once the swing verdict lands.
- Untested undisproved road: structural "be the house" (DeFi LP vaults, vol risk premium).

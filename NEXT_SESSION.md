# 👉 PICK UP HERE — next session (paused 2026-06-30 evening)

## TOMORROW'S BUILD: an INTELLIGENT GRID BOT

The grid bot was always the eventual goal (the analyst README even says "the grid bot
only comes after the analyst is a trusted instrument"). Now's the time.

**What a grid bot is:** place a ladder of buy orders below and sell orders above the
current price; profit from price *oscillating* through the rungs (buy low rung, sell the
next rung up, repeat). It's NON-directional — it harvests chop/range, not trend.

**Why it can work where everything else failed:** it doesn't predict direction (which we
proved is dead). It monetizes range-bound oscillation, which is a different thing.

**What makes it "intelligent" (and the one legit use we found for the analyst):**
  1. **Vol-adaptive spacing** — set rung spacing from ATR / realized vol, not a fixed %.
  2. **Regime gate = the analyst's real edge.** The analyst CAN'T call direction, but its
     regime classifier CAN tell range vs trend vs flash (that's the part with measured
     lift). So: run the grid ONLY in a RANGE regime; stand down (or stop-out) in
     trend/flash. This is the honest, validated use of the analyst.
  3. **Hard trend stop** — a grid's fatal flaw is a strong trend running out of the grid
     (you accumulate a losing bag on one side). Need a band/stop that exits when price
     breaks the range.

**THE HONESTY BAR (same as everything else):** grid bots look gorgeous in a backtest of a
ranging period and BLOW UP in a trend. So:
  - Backtest FIRST on real candles, across regimes (including 2021-22 + trends), with real
    fees + the trend-blowup modeled. No deploy until it survives a trending stretch.
  - Judge net of costs, OOS, with the worst trending month shown — not the pretty range months.
  - Only then → Binance testnet (same ccxt/sandbox infra as the intraday trader).

**Suggested first step:** build `grid_backtest.py` — simulate a vol-spaced grid on BTC/ETH
candles, gated by the analyst regime (range-only), with a trend stop, net of fees, reported
across regimes with the worst trend stretch called out. If it survives, THEN testnet.

## STATUS OF EVERYTHING ELSE
- **Intraday-TSM: running live on testnet** (cron 08:05 enter / 00:05 exit). The forward
  test is the scorecard (~+0.8%, 69% win, accumulating). Today was green (+$13.98). Let it
  run — judge over weeks, not days. Dashboard shows your-manual vs bot-auto closes.
- **VRP (variance risk premium): concluded.** Real premium, but the tradeable defined-risk
  condor loses to crypto's fat tails; the delta-hedged version works but is too complex/
  risky and we declined it. Fully written up in Failed_Strategies_Postmortem.md/.pdf.
- **Everything directional (Kronos, analyst, stablecoin, etc.): dead** — all in the postmortem.
- **Deribit testnet API:** read-only key works (deribit_check.py). Parked unless VRP revisited.

## THE THROUGHLINE (don't forget): no flattering number gets trusted until it survives
OOS + placebo + breadth + real-friction. That discipline is what's kept real money safe.

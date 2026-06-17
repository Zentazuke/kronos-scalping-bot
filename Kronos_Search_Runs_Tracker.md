# Kronos — Entry-Search Runs Tracker
Tracking each "green" across runs, so we judge **persistence** (does the same edge survive on fresh data) instead of celebrating whatever rule won this time. Single-rule greens have been unstable; the more trustworthy signal is which *ingredients* keep surviving.

---

## Run A — 2026-06-16 08:11 UTC
- Data: `observations.db`, **2,689** trades, **2-condition** combos, 2,326 searched, noise floor 1.059
- Take-all: Sharpe −0.085 (47% win)
- **Verdict: GREEN** — best `funding ≥ 7.6e-05 AND macd ≥ 2`, OOS Sharpe **0.066** / 39, PSR 66%
- Strongest holdout: `funding ≥ 6.9e-05 AND supertrend ≥ 2` → OOS **0.48** / 132 / 71% win
- Surviving ingredients: consensus×27, supertrend×10, macd×7, funding×6, di_align×4, cci×4, adx×3, boll×3, votes×3, donchian×3, stoch×2, attention×1, fear_greed×1, sent_aligned×1

## Run B — 2026-06-16 13:43 UTC
- Data: `observations.db`, **2,970** trades, **3-condition** combos, 46,672 searched, noise floor 3.061
- Take-all: Sharpe −0.081 (47% win)
- **Verdict: GREEN** — best `macd ≥ 1 AND rsi ≥ 54.42 AND stoch ≥ -2`, OOS Sharpe **0.267** / 99, PSR 99%
- Funding cluster this run: `funding ≥ 7.6e-05 AND macd ≥ 2 AND votes ≥ 2` → OOS **−0.03** / 16 / 50%; `conviction ≥ 0.7 AND funding ≥ 7.6e-05 AND supertrend ≥ 2/3` → OOS 0.16–0.23
- Surviving ingredients: votes×36, funding×30, supertrend×17, conviction×16, consensus×14, di_align×6, stoch×4, adx×4, macd×2, outlook_aligned×2, rsi×1, attention×1, donchian×1, sent_aligned×1, book_imb×1, rel_volume×1, cci×1, ls_ratio×1, boll×1

## Run C — 2026-06-16 23:23 UTC
- Data: `observations.db`, **3,680** trades, **3-condition** combos, 46,672 searched, noise floor 1.443
- Take-all: Sharpe −0.055 (48% win)
- **Verdict: GREEN** — best `conviction ≥ 0.667 AND stoch ≥ 2 AND votes ≥ 3`, in-sample Sharpe 2.42 / 38 / 97% win, OOS Sharpe **0.090** / 34 / 53% win, PSR 69%
- Surviving ingredients: stoch×5, votes×4, conviction×3, adx×1, outlook_aligned×1, consensus×1  *(far fewer survivors than Run B — the previously "strong" funding/supertrend cluster mostly dropped out)*

## Run D — 2026-06-17 11:51 UTC
- Data: `observations.db`, **4,674** trades, **2-condition** combos, 2,461 searched, noise floor 0.768
- Take-all: Sharpe −0.037 (49% win)
- **Verdict: GREEN banner** — best `conviction ≥ 0.9 AND supertrend ≥ 2`, in-sample Sharpe 1.25 / 81 / **86% win**, OOS Sharpe **0.092** / 62 / **53% win**, "edge-is-real 76%"
- **Honest read: same rule-roulette.** The in-sample→OOS *win-rate collapse is the tell*: 86% → 53% (a real edge doesn't shed 33 points out of sample). OOS column is ~half negative; the sibling `conviction ≥ 0.9 AND donchian ≥ 2` is OOS **−0.96 / 14% win** — one indicator away from the "winner". 53% OOS win at 2.5/2.5 geometry = coin-flip that loses after fees.
- **Already disproven directly:** `conviction_gate_check.py` put `conviction ≥ 0.9` through per-coin/per-month/net-of-fee testing earlier and it FAILED (BTC 15% win, per-coin miscalibrated, gating made it worse). Run D is in-sample mining re-surfacing a signal a clean test already killed.
- Surviving ingredients: conviction×5, supertrend×2, votes×2, macd×2, funding×1, stoch×1, consensus×1 — `conviction`+`supertrend` recur from Run A's neighborhood, but the OOS stays coin-flip.
- **Action taken:** locked `conviction ≥ 0.9 AND supertrend ≥ 2` into `rule_forward_check.py` (forward test on post-2026-06-17 observations only) to settle it the one honest way — forward, on data the search never saw. Verdict pending accumulation (rule fires rarely, ~1–2/wk).

---

## What the four runs together tell us (the honest read)

**The single best rule keeps changing and never replicates.** Four greens, four different winners:
- Run A → `funding + macd` (OOS 0.066) — collapsed to OOS −0.03 in Run B
- Run B → `macd + rsi + stoch` (OOS 0.267) — absent from Run C's top rules
- Run C → `conviction + stoch + votes` (OOS 0.090) — never appeared in A or B
- Run D → `conviction + supertrend` (OOS 0.092, but 86%→53% win IS→OOS) — and `conviction` was already disproven by the dedicated breadth/cost test

**Runs B and C are apples-to-apples** (both 3-condition, both ~46.7k combos), so the non-replication is clean: B's winner isn't in C, C's winner isn't in B. A real edge is *stable*; a different winner every run, with prior winners vanishing, is the fingerprint of **mining noise** from tens of thousands of combos.

**Even the ingredients reshuffle.** Run B's survivors were dominated by funding×30 / supertrend×17 / votes×36. Run C's are stoch×5 / votes×4 / conviction×3 — funding and supertrend largely *gone*. The only ingredient that recurs across all three with any consistency is **votes** (and weakly conviction/stoch), but the magnitudes swing wildly. So even the "trust ingredients not rules" fallback is shaky here.

**Three green banners, three different rules, none reproduced.** The deflation guard handles over-testing *within* a run; it can't handle us re-running and stopping at the first green *across* runs. Each green is ~30-trade-OOS noise at 50-ish% win. Don't trust the banner.

**Note for the backtest plan:** these rules use bot-computed live features (conviction = Kronos, votes = confluence) that can't be recomputed from historical candles — so there's no months-of-history backtest for them like we did for TA/reversion. The only real test is *forward in time*: keep harvesting and check whether the SAME rule greens on data the search never saw.

---

## Next step (Ricardo's call): forward-test in real time

The plan is to eventually test the surviving rules/ingredients **live on the bot (testnet)** — a true forward test. That's the gold-standard out-of-sample check: no lookback bias, no multiple-looks, no data-mining. It requires:
- committing to **one** rule/config (e.g. a funding + trend-confirmation gate) and leaving it fixed,
- letting it run long enough to accumulate enough trades to judge,
- comparing its live win rate / expectancy to the take-all baseline.

If a rule survives a clean forward test, *that's* a real edge. If it dissolves like the backtest greens, we've confirmed the directional search is noise — cheaply, on testnet.

## For now
Keep harvesting, untouched. Re-run the search on the same combo setting as data grows, and watch the **ingredients**, not the banner.

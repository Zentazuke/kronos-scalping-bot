# Strategy Roadmap — post-grid convergence (2026-07-14)
*External proposal reviewed against house evidence; annotated verdicts. The thesis is accepted: the analyst's proven intelligence is slow, directional, allocation-grade — so the roadmap is capital allocation, not another bot.*

| # | Idea | Verdict | Notes / house-evidence adjustments |
|---|---|---|---|
| 1 | **Optimize sleeve consumption** | ✅ BUILD NOW (backtest only) | `sleeve_consumption_study.py` — 4 pre-registered variants, see below. Does NOT touch the running shadow test; a winning variant becomes a *third* shadow sleeve after the Aug 1 read, never an edit to the live ones. |
| 2 | Slow cross-sectional rotation | 🟡 One pre-registered test, after #1 | **Survivorship trap:** today's top-30 universe backtested 5yr = testing only survivors. V1 must use a fixed basket of long-listed majors and say so. Weekly/monthly rebalance, long-only, turnover penalty, benchmarks incl. the sleeves. |
| 3 | Structural-yield allocator | 🟡 Research engine, fits the analyst | "Which premium is worth bearing" — extends carry_read. Compare stable yield / staking / cash-and-carry / funding, each minus venue+depeg+contract tail reserves. Non-market risk is the whole game; a 9% yield at a failable venue is not 9%. |
| 4 | DeFi LP as short-vol | 🟡 Simulation research only, later | Conceptually the VRP lesson transferred (LP ≈ selling options; IL = the tail). Big lift; only after 1–3. Same trap as the condor: fee income is visible, the tail is not. |
| 5 | Event flows (unlocks etc.) | 🟡 Analyst *context* first | Data sourcing is the blocker (reliable unlock calendars). ONE pre-registered event class if ever tested; no filter stacking. Own stablecoin-flow failure is the cautionary precedent. |
| 6 | Convex protection overlay | 🟡 Only VRP-gated | House data: implied > realized 74% of days → systematically buying protection = systematically overpaying. The permissible version: protection only when the analyst's own VRP verdict = **cheap**. Portfolio engineering, judged on DD/recovery, not return. |
| — | Everything in the "do not reopen" list | ✅ Agreed | Matches the graveyard. The "harvestability metric" stays parked — valid idea, high risk of becoming the grid family's séance. |

## Pre-registration — sleeve consumption study (locked 2026-07-14, before any results)

**Question:** can we preserve the sleeve signal while reducing drawdown and needless transitions? NOT "which of 300 combos has max Sharpe."

**Variants (only these; no post-hoc additions):**
- **Baseline:** current binary trend-rider and default-long (as running live).
- **A — confidence-scaled:** exposure = analyst regime confidence (0..1) instead of binary.
- **B — vol-targeted:** binary signal × min(1, target_vol / realized_vol), target 40% annualized, 30-day realized.
- **C — A × B combined.**
- **D — portfolio-level with no-trade band:** basket exposure = share of coins confirmed, equal-weight, rebalance only when target shifts > 15 pp.

**Judged on:** OOS (last 40%) per coin + equal-weight basket: Sharpe, maxDD, total, flip count — vs Baseline and vs buy&hold. **Success bar:** a variant must beat BASELINE OOS Sharpe with maxDD no worse, on the basket AND ≥4/7 coins. Ties or partial wins → keep the baseline (it's simpler and already live-validated). Costs: one taker+slip leg per exposure change, scaled by |Δexposure|.

**Sequence:** run study now (backtest) → sleeve shadow verdict ~Aug 1 → if sleeves confirm AND a variant won, promote variant to a third shadow sleeve for 2+ weeks before any executor work.

**RESULT (2026-07-14, real data, 7 coins, 4.5yr):**
- **TR-A (confidence-scaled trend-rider) PASSES the bar** — basket Sharpe 1.17 vs 1.06 with DD 19% vs 23%; 4/7 coins (the minimum qualifying breadth). It won by *risk reduction*, not return (+57% vs +62% total): the desired failure-resistant shape. Flips 111 vs 40 median, but each is a small |Δconf| adjustment so fee drag stays modest.
- B/C (vol-targeting) FAIL — 2.8k–14k micro-rebalances; churn ate the signal (compatibility doctrine, again). D (portfolio bands) FAIL. **Default-long: no variant beat binary — DL stays as-is.**
- **Action per pre-registration:** TR-A earns third-shadow-sleeve status *after* the ~Aug 1 sleeve read confirms the baselines — `trend_sleeve_forward.py` gets a conf-scaled TR column then, NOT before. No change to anything running.
- **Interpretation preserved:** TR-A is a *risk-management upgrade*, not new alpha. And the analyst discovery: **confidence is information about how much risk to carry, not permission to trade more often** — scale with it, never churn because of it (also recorded in the analyst's MODEL_CARD).

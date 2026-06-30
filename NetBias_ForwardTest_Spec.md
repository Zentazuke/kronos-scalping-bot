# Net-Bias Gate — Frozen Forward-Test Spec
The first signal that survived the honest wringer: trades taken when the TA board strongly agrees with the direction win far more (OOS ~75-80%) than trades taken against it (OOS ~17-25%), and the aligned zone clears fees (thin at 15 bps, clearly positive at 10 bps / BNB). This spec freezes the rule **before** more data arrives, so we can't move the goalposts.

---

## What we found (honest summary)
- Single pre-specified feature (`ta_consensus`, direction-aligned), bucketed, time hold-out — **no data-mining inflation**.
- Monotonic win-rate staircase, **holds out-of-sample through a worse regime** (the absolute win rate dropped but the separation widened).
- Net expectancy after fees is **thin but positive in the aligned zone**: top bucket ≈ +0.05%/trade at 15 bps, ≈ +0.10%/trade at 10 bps (BNB).
- Edge **plateaus** around moderate-strong alignment — extreme selectivity does NOT improve it (don't over-tune the threshold).
- Measured on the **offline labeler's 2.5×/2.5× ATR bracket** (R:R 1.0) — NOT the live bot's current 1.5×/2.5× (R:R 0.6).

## The rule to freeze (no changes once it starts)
- **Gate:** take a trade only when direction-aligned `ta_consensus ≥ 5`. (Captures the robust aligned zone; not the noisy extreme.)
- **Geometry:** trade the **2.5× / 2.5× ATR** bracket — the geometry the edge was measured at. (This means temporarily setting the live `TAKE_PROFIT_ATR_MULT` to 2.5 for the test, matching `SL`.)
- **Fees:** pay in **BNB** (gives the edge its margin). 10 bps round-trip assumption.
- **Symbols:** the existing 8. **Testnet only.**
- Everything else unchanged. The observation journal keeps recording **every** bar (gated or not), so we never stop learning.

## Implementation (one bounded change)
- Add a pre-trade gate in the decision path: if `aligned_consensus < 5` → skip the live trade (still record the observation).
- Set TP multiple to 2.5 for the duration of the test.
- Reversible: it's two config-level changes, revert by removing the gate and restoring TP=1.5.

## Pass / fail bar (decide before we look)
The **win-rate** separation is already statistically strong; the open question is whether **net expectancy stays positive** on fresh trades. So:
- **PASS:** over the forward window, gated live trades show net expectancy **> 0** (BNB fees) with a win rate in the **70%+** range, on a sample large enough to trust.
- **FAIL:** net expectancy drifts to **≤ 0**, or win rate falls well below ~70%.
- **Sample needed:** the edge is small (~+0.05-0.10%/trade), so it takes volume to separate from zero — target **~300-500 gated trades** (roughly 1-2 weeks at current rates) before judging. Don't call it early on 50 trades.

## What would falsify it (stay honest)
- Gated win rate collapses toward the take-all 49% → the staircase was regime-specific, not real.
- Net expectancy negative despite decent win rate → fees/slippage eat it; needs cheaper fills or different geometry.
- Live results diverge from the backtest because real fills differ from the labeler's clean bracket replay (phantom-fill risk on testnet — watch this).

## Caveats carried in
- Thin edge: this is a marginal, hard-won signal, not a money printer. Treat it as "first real foothold," not "deploy and scale."
- Threshold is mildly noisy across bucketings; `≥ 5` is a deliberate round choice, not an optimized one.
- Small OOS per-bucket samples — the forward test is what upgrades this from "promising" to "real."

## Decision points for Ricardo (before we build)
1. Threshold: `consensus ≥ 5` (recommended) vs a stricter `≥ 9` (thinner, safer-at-15bps but fewer trades).
2. Geometry: match the measurement at 2.5/2.5 (recommended) vs test at the current live 1.5/2.5 (different question).
3. Go live with the gate now, or keep harvesting and re-run `net_bias_check` on **future-only** data first (zero-touch forward test, slower but changes nothing).

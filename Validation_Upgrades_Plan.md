# Validation Upgrades — Plan Before We Build

*Purpose: decide whether and how three rigor upgrades actually help the data we're gathering right now (the observation journal), before writing a line of code. Drafted from the two research memos (`algorithm_research_strategy.md`, `algorithm_research_technical_analysis.md`), which largely validate the current build and point to exactly these gaps.*

---

## First, the thing we're optimizing for

We're filling `observations.db`: every directional bar across 8 symbols, labeled offline against mainnet brackets (which is already López de Prado's triple-barrier method). The property that matters for validation is that **this data is dense and heavily overlapping**:

- Bars are 5 minutes apart; each label's outcome window is the bracket horizon (up to ~4 hours / 48 bars).
- So a setup at bar *t* and the setup at bar *t+1* on the same coin share almost the same future path — their win/loss labels are strongly correlated.

That density is the whole point (it's how we get 10× the data), but it changes what "honest validation" has to look like. Every upgrade below is judged by one question: **does it help us read this specific data without fooling ourselves?**

---

## Upgrade 1 — Purged cross-validation + Deflated Sharpe Ratio
**Role: foundation. Time-sensitive — best built before observations cross ~100 labeled.**

**Why it's advantageous on this data, specifically:**

- **Overlap is leakage.** Our current walk-forward is a single expanding split with no purging. With dense overlapping setups, near-identical neighbours can land on opposite sides of the train/test boundary — the model effectively "remembers" the test answer, inflating the apparent edge and handing us a **false green light**. This is worse on the observation journal than it was on the sparse trade journal precisely *because* the observation data is so dense. *Purging* (drop training rows whose label window overlaps the test window) plus an *embargo* gap removes that leak.
- **Multiple-testing honesty.** As the data grows we'll run many configurations — logistic vs XGBoost, journal vs observations, different feature sets, maybe hyperparameters. Each trial is another chance to find spurious edge. The **Deflated Sharpe Ratio** discounts the best result by the number of trials and corrects for skew, kurtosis, and sample length, returning *the probability the edge is really positive*. Without it, "we tried six things and one looked good" masquerades as signal.

**Net payoff:** the first time observations cross ~100 labeled, the verdict is *trustworthy* instead of optimistic. That's the single thing we're waiting on, so making it honest is the highest-value move.

**What we'd measure:** the edge **with vs without** purging (the gap is the leakage we were previously fooling ourselves with), and a DSR/PBO probability that the edge is real given how many models we've tried.

**Effort / risk:** moderate, contained entirely to `learner.py`; testable on synthetic data; **no change to the live bot**.

---

## Upgrade 2 — Feature pruning
**Role: sharpens the model. Partly doable now; the rigorous half depends on Upgrade 1.**

**Why it's advantageous on this data, specifically:**

- We record 26 feature columns, and many are near-redundant (`adx` with `plus_di`/`minus_di`; `trend_1h`/`trend_4h`/`trend_1d`; `p_up` with `p_down`; `depth_imbalance` with `total_depth`). The TA memo is blunt: indicators collapse into a few latent factors, and stacking correlated ones "inflates apparent confidence without adding information." High dimensionality on limited samples = the model burns capacity fitting noise.
- **Fewer orthogonal features need less data to surface real signal** — directly useful while the observation count is still modest. This is help *now*, not later.
- **A first cut is free and leak-proof today:** an unsupervised correlation prune (drop pairwise |ρ| > ~0.75) needs no labels and can run on the existing `journal.db` (437 trades) immediately.
- The *supervised* importance pass (MDA — does shuffling a feature hurt the out-of-sample score?) must run **inside** purged CV or it introduces selection bias. Hence it waits on Upgrade 1.

**What we'd measure:** does a pruned ~10-feature model match or beat the full 26-feature model under purged CV? Same signal with fewer features = more robust and less overfit.

**Effort / risk:** a standalone analysis script that outputs a recommended feature set; the unsupervised cut is immediate, the supervised cut follows Upgrade 1.

---

## Upgrade 3 — Candlestick feature flags
**Role: defer. Low expected value and a real timing cost.**

**Why it barely helps this data:**

- New features only attach to **future** observations. Existing `journal.db` and everything already recorded won't have them — so they **reset the "enough data" clock by weeks** before they could contribute to any verdict.
- Prior support is weak (Marshall–Young–Rose: candlestick strategies "do not have value" in liquid markets; both memos say encode them as low-weight features only, never as rules).

**Verdict:** defer. If we ever deliberately expand the feature set, add them early so the clock starts — but don't expect near-term help, and keep them low-weight.

---

## Recommended sequence

1. **Build Upgrade 1 (purged CV + Deflated Sharpe) now** — it's the foundation and it's time-sensitive (ideally in place before observations cross ~100). Everything else trusts its output.
2. **Run the unsupervised correlation prune now** on `journal.db` — cheap, leak-proof, no dependency. Hold the supervised MDA prune until Upgrade 1 exists.
3. **Defer Upgrade 3** unless/until we choose to expand features.

---

## What the whole thing buys us

- A verdict we can **act on without self-deception** — the entire reason we're gathering 10× data.
- A **leaner model** that can find real signal sooner on the data we already have.
- A clear, evidence-graded answer to "is there an edge here?" — either a clean green light to go live (capped fractional Kelly, as planned) or a definitive *no*.

## Decision checkpoints (when to stop or pivot)

- If the purged edge **collapses toward zero** once leakage is removed, **and** the Deflated Sharpe says "not significant" even as the data grows → that's the honest no-edge. Pivot to the funding-arb sleeve.
- If the purged **and** deflated edge is convincingly positive across folds → promote the meta-gate from shadow to live with capped fractional Kelly, exactly as the roadmap says.

*Bottom line: none of this changes the destination — it makes the instrument readings trustworthy. Build Upgrade 1 first; it's the difference between a verdict and a guess.*

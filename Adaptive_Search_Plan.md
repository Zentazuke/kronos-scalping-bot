# Kronos — The Adaptive Search Engine (the actual "learning algorithm")

*Design plan. This is the system that tries many combinations, finds patterns, and learns which strategies work and why — instead of running one fixed config forever.*

---

## What you asked for

Not a bot that repeats one setup and hopes. A system that **experiments**: proposes many strategy combinations, tests each honestly, records everything, and **learns which ingredients drive an edge and which don't** — and can keep adapting as new data arrives. An automated researcher.

## The shape of it: Search → Validate → Understand → Adapt

1. **Search** — automatically generate strategy configurations (combinations of rules and parameters).
2. **Validate** — score each one with the leak-free, overfit-corrected machinery we already built.
3. **Understand** — analyse across all the results to find *why* some worked: which ingredients, in which conditions.
4. **Adapt** — promote the best validated config to live, and re-run as new data comes in. (The closed loop.)

## The one rule that makes or breaks the whole thing

**Trying many combinations and keeping the best is *also* the number-one way to fool yourself.** Try 5,000 configs and dozens will look brilliant by pure luck. This is the data-snooping trap, and it's why most "I backtested everything and found gold" systems blow up live.

The defense — already built — is non-negotiable:
- Every result is **discounted by how many things were tried** (Deflated Sharpe; the trial count is the penalty).
- The final pick is confirmed on a slice of time **the search never saw**.
- We prefer broad **plateaus** (many nearby configs all work → robust) over lone **spikes** (one lucky config → overfit).

Without this layer, "try as many combinations as possible and find patterns" is just automated self-deception. With it, it's genuine learning. We have it.

---

## What can vary (the search space)

The realistic, tractable version searches over the data we already collect — every recorded setup carries its full feature vector and can be re-labelled under different exit rules. So the knobs are:

- **Entry filters** — thresholds on the recorded signals: Kronos `p_up` cutoff, `adx` minimum, confluence votes required, RSI/trend conditions, order-book imbalance, etc. (Which signals to trust, and how strongly.)
- **Risk / reward** — TP and SL multiples (re-label observations at different brackets), time stop.
- **Feature set** — which of the 26 features the decision uses (this is where the "pruning" question finally gets answered *empirically*).
- **Regime conditioning** — only trade in certain ADX / volatility / trend states.
- **Direction logic** — trust Kronos, fade it, or require agreement with the gates.

What *can't* vary yet: anything the bot doesn't currently record. New indicators mean the bot starts logging them first, then they enter the search later. (Kronos itself stays frozen — what adapts is the strategy *around* it.)

---

## Components (what to build)

| Component | Status | Notes |
|-----------|--------|-------|
| Clean data + feature pipeline | ✅ have | observation journal + mainnet labeler |
| **Validator (purged CV + Deflated Sharpe)** | ✅ **have** | the hard part — done |
| Configurable backtester | 🔨 build | given a config, filter/re-label the observations → PnL series |
| Search loop | 🔨 build | random/grid first, then smarter (Bayesian/genetic) |
| Trial ledger | 🔨 build | record every config + full results (the memory; powers deflation *and* pattern-finding) |
| Pattern analysis | 🔨 build | which ingredients drive edge; plateau-vs-spike; sensitivity |
| Adaptation loop | 🔨 later | promote best validated config live + scheduled re-search |

The "Understand why" layer is the part most systems skip and the part you specifically want. Over the trial ledger it answers: *which config dimensions correlate with real (deflated) edge? Is the good region a stable plateau or a fluke? How does edge move as I turn each knob?* That's the system explaining itself in plain terms — "edge concentrates where R:R > 2 and ADX > 25, regardless of the RSI setting," for example.

---

## Build phases

**Phase 0 — Backtester.** Generalise the labeler into "given a config, replay it over the clean data and return a PnL series." This is the foundation everything else stands on.

**Phase 1 — Close the loop on a small grid.** Wire Search → Backtest → Validate, run a few hundred configs, fill the trial ledger. First real output: a ranked, deflation-corrected leaderboard.

**Phase 2 — Understand.** Build the pattern analysis over the ledger. First insight report: what actually drives edge (and the honest verdict on whether any survives the deflation).

**Phase 3 — Scale & confirm.** Bigger space, smarter search, and a held-out time period the search never touched, for final confirmation of anything promising.

**Phase 4 — Adapt.** The best validated config goes live; the search re-runs on a schedule so the live strategy updates itself as markets change. *This* is the self-adapting learning algorithm, end to end.

---

## The honest expectations (so we don't repeat the over-claim)

- This finds an edge that **exists** in the data. It **cannot create one that isn't there.** If the search comes back empty even after exploring thousands of honest combinations, that is itself a real, hard-won answer — not a failure of effort.
- It's an **offline research engine** — heavy compute, runs on the server, never risks money until Phase 4 (and even then, paper first).
- It's a **multi-week build**, not a weekend. But it reuses the hard part we already have.
- Expect most combinations to be noise. The job is to find the rare robust region, *if one exists*, without being fooled by the lucky ones.

---

## Decisions before we start

1. **Scope of the search.** Start narrow (tune the current directional strategy over the existing features), or wide (also include other families — mean-reversion, breakout)? Wider = more chance of finding something, but more compute and more overfit risk to manage.
2. **End goal.** A one-time research run that hands you the best validated config, or the full self-adapting live loop (Phase 4)?
3. **Data depth.** The search is only as honest as the data spans. We may want to let the observation journal grow across more market conditions first, or start now on what we have and expand.

My recommendation: **build Phase 0–2 now on the existing data and narrow scope** — that gets you a real, honest "what works and why" report fast and cheap, and proves the engine end-to-end before we scale or wire it live.

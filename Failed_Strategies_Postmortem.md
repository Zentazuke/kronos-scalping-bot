# Kronos Project — Failed-Strategies Postmortem
### Every directional/scalping approach we tried, why it died, with the numbers
*Compiled 2026-06-27. Testnet/backtest figures throughout — no real capital was risked.*

---

## TL;DR — one root cause killed almost everything

At the frequency we were trading, **the directional signal was always thinner than the cost of trading on it.** Round-trip cost on Binance is ~**15 bps with the BNB discount, ~20 bps without**. Every fast edge we found was worth a few bps per trade — *smaller than the toll*. So a strategy could be genuinely "right" 51% of the time and still bleed, because the 1% edge didn't cover the 0.15–0.20% it cost to collect it.

We proved this **at least five independent ways** (Kronos, meta-labeler, consensus TA, the entry-rule search, OFI). It is not a bug in our code — it's the nature of a liquid, efficient market versus a retail cost structure.

The scoreboard below is brutal but clean. **Nothing in the "directional prediction" or "market-timing" family survived the 5-year, multi-regime bar.** Two slower ideas lasted longest. Intraday-TSM's base config failed 5yr, and a more selective "careful bets" version failed too (pickier = *worse* out-of-sample). Stablecoin-flow has now also failed its 5-year run (**−8.4% vs buy-and-hold +40%**). Intraday-TSM is kept alive only as a *live* testnet forward test. The one family we never disproved — because we never built it — is the structural **"be the house"** set (DeFi LP vaults, vol risk premium): edges that collect a premium instead of predicting, and don't hit the cost wall.

---

## Master scorecard

| Strategy | What it was | Best honest result | Verdict |
|---|---|---|---|
| **Kronos** (core) | Kronos-small ML price predictor, 30× Monte-Carlo, edge gate ≥0.53 | Take-all **47–49% win**, Sharpe **−0.04 to −0.09** (negative before fees) | ❌ Dead — never cracked 50% |
| **Kelly / meta-labeler** | Logistic-regression P(win) filter over journal features | v1 **33.3%** holdout vs **61.9%** majority baseline; retrain ~**45%** vs **51%** | ❌ Anti-predictive; never promoted past shadow |
| **Entry-rule search** (rule-roulette) | Mined 2–3 condition combos from 2.7k–4.7k trades | 4 "greens," OOS Sharpe **0.07–0.27** on ~30–130 trades, **never replicated** | ❌ Data-mining noise |
| **Consensus TA board** | 7-indicator trend net-bias (MACD, Supertrend, Stoch, CCI, Boll, Donchian, OBV) | **48% win** at 1:1 over **5 months** | ❌ Lost money trading with the signal |
| **RSI-2 reversion** | Connors-style fade-the-extreme (counter-test to TA) | No breadth+persistence survivor net of fees | ❌ Didn't clear cost wall |
| **OFI** (order-flow imbalance) | Cont-style queue-change signal | The one *real* micro-signal, but **net-negative as taker**; maker = optimistic best-case, still adversely selected | ❌ Edge < spread |
| **Lead-lag** (BTC-residual catch-up) | Trade alt's under/over-shoot vs BTC-implied move | Raw BTC lead ≈ **0** (it's a ~15s HFT game); residual catch-up didn't survive OOS | ❌ Too fast / no edge |
| **Funding carry** | Delta-neutral long spot / short perp, collect funding | Headline research ~**19%/yr**; *our* net after real financing + fees + rotation = thin, tail-risky | ⚠️ Structural but compressing; not deployed |
| **Pairs / stat-arb** | Cointegration mean-reversion, market-neutral | Beat passive on paper in lit; *our* OOS weak — cross-sectional momentum confirmed **weak** | ⚠️ Marginal after costs |
| **Vol risk premium (VRP)** | Sell overpriced implied vol (short variance) on Deribit | Premium is **REAL** (proxy Sharpe ~1.8, implied > realized **74%** of days, +9.7 vol-pts); but the tradeable **defined-risk iron condor LOSES** (net Sharpe **−0.04 to −0.8** across strikes — crypto's fat-tailed endpoints breach the static structure). The version that *does* monetize it (delta-hedged short straddle) is operationally brutal + tail-risky. | ⚠️ The **one genuinely real edge** — but not cleanly harvestable by a small retail trader |
| **Crypto-Analyst** (as alpha) | Calibrated regime/ensemble used to gate or trade | Intraday gates "won" then **failed the placebo** (1D + 4h, p=0.18 / 0.36 — one-month mirages); standalone swing **failed breadth** (2 of 7 coins, ETH was luck) | ❌ Honest engine, but no tradeable directional edge — its own model card says so |
| **Intraday-TSM** | Morning→afternoon session momentum, daily hold | **+16%** on 1yr → **−17% / Sharpe 0.22 / −52% DD** on **5yr**; selective "careful-bets" version **worse** OOS | 🟡 NOT CLOSED (live only) — both backtests failed; alive only as a live forward test |
| **Stablecoin-flow** | On-chain stablecoin supply timing (non-price) | **Sharpe ~1.3** on 1yr → **−8.4% / Sharpe −0.17** on 5yr vs buy-hold **+40%** | ❌ Failed 5yr — regime luck (IS −0.61 → OOS +0.32) |

---

## The directional-prediction family (everything that died the same death)

### Kronos — the original engine
Kronos was a transformer price-predictor wrapped in a disciplined execution stack (regime sieves, confluence votes, ATR brackets, half-Kelly sizing). It was beautifully built. It just couldn't predict.

- **Take-all baseline** across the harvested trade population: **47–49% win rate**, Sharpe **−0.037 to −0.085**. That's a coin-flip that *loses* once you add the bracket geometry and fees.
- The model's confident calls were often flatly wrong — there were nights of **28–30 of 30 Monte-Carlo paths pointing the wrong way**.
- By direction/coin it was lopsided noise: **ADA shorts bled 7W/13L (35%)** while ADA longs were 7W/1L and BTC shorts 7W/2L — i.e., no stable directional edge, just regime accidents.

**Why it died:** at the 5-minute horizon, the predictable component of the next move is smaller than the spread+fee you pay to act on it.

### Kelly / the meta-labeler — the filter that couldn't filter
The meta-labeler was supposed to rescue Kronos by learning *which* trades to take.

- **v1 (101 trades):** holdout accuracy **33.3%** vs a **61.9%** predict-majority baseline. It was *worse than guessing "majority class"* — it had memorized one overnight regime.
- **Retrain:** ~**45%** vs the ~**51%** baseline. Still under water. It never once beat the naive baseline, so it correctly stayed in **shadow mode** and was **never promoted to veto.**

**Why it died:** you can't meta-label your way to an edge that isn't in the features. Garbage direction in, garbage filter out.

### The entry-rule search — the seductive one
This is the trap we kept circling back to: mine the harvested trades for a magic rule. We ran it four times (Runs A–D) with deflation guards (PSR, Deflated Sharpe).

- Every run produced a **GREEN banner** — and **a different winner each time**:
  - Run A → `funding + macd` (OOS Sharpe 0.066) → collapsed to −0.03 next run
  - Run B → `macd + rsi + stoch` (OOS 0.267) → absent in Run C
  - Run C → `conviction + stoch + votes` (OOS 0.090) → never seen in A/B
  - Run D → `conviction + supertrend` (OOS 0.092) — but **86% in-sample win → 53% out-of-sample.**
- That **86%→53% win-rate collapse is the fingerprint of overfitting.** A real edge doesn't shed 33 points out of sample. The sibling rule one indicator away was OOS **−0.96 / 14% win**.
- `conviction ≥ 0.9` had *already been killed* by a dedicated per-coin/per-fee test (**BTC 15% win**). The search just kept re-surfacing noise a clean test had buried.

**Why it died:** searching tens of thousands of combos and stopping at the first green finds luck, not edge. Different winner every run = pure mining noise. **This is the same "find something there by slicing per-coin" instinct we have to keep resisting.**

### Consensus TA + RSI-2 reversion — both sides of the same coin
- The **7-indicator trend board**, traded *with* its own strong-agreement signal over **5 months**: **48% win at a 1:1 bracket → net loss.** Direct evidence that at short horizons price doesn't trend, it chops.
- So we tested the **opposite** (RSI-2 fade-the-extreme). It also failed to produce a breadth-and-persistence survivor net of fees. Neither trend nor counter-trend has a retail-harvestable edge at this frequency.

### OFI and lead-lag — the "real but unreachable" signals
- **OFI** was the one micro-signal that genuinely carried information — but it's **net-negative as a taker** (you pay the spread chasing flow), and the maker version only fills the trades that *don't* run your way (adverse selection). Even the optimistic best-case sim couldn't flip it positive.
- **Lead-lag:** the raw BTC→alt lag is **~15 seconds** — an HFT game we can't win. The slower retail-reachable "residual catch-up" cousin showed correlation ≈ 0 and didn't survive IS/OOS.

---

## The structural / market-neutral family (researched, mostly set aside)

These don't predict direction — they collect a premium. More durable in principle, but each had a real blocker for *our* setup:

- **Funding carry (cash-and-carry):** headline research says ~**19%/yr with <2% drawdown** (one study: 115.9% over six months, worst-case −1.92%). But returns are **compressing as retail piles in**, it needs **two legs + margin management + borrow costs**, and the short leg carries liquidation tail-risk in a violent move. Net-of-everything for a small operator is far thinner than the headline. Promising as a *pivot*, never deployed.
- **Pairs / stat-arb:** cointegration mean-reversion beats passive on a risk-adjusted basis *in the literature*, but our own testing confirmed **cross-sectional momentum is weak** and the OOS edge was marginal after fees.
- **Vol risk premium:** a genuinely robust premium (implied vol > realized), but a **fat left tail** (collect small for months, lose big in a crash) and it requires an **options venue (Deribit)** we aren't set up on.

The reframe from the deep-dive holds: **stop being the gambler, become the house** — provide liquidity (DeFi LP vaults like HLP: ~22% CAGR, Sharpe 2.9–5.2), sell insurance (vol premium), or read information that hits the chain before the price (on-chain flow).

---

## The one genuinely real edge — and why we still couldn't harvest it

The variance risk premium (VRP) is the only thing in this entire project the data *refused to disprove*. On BTC, the implied volatility the options market charges (Deribit's DVOL) sits **systematically above** the volatility that actually shows up — implied beat realized **74% of days, by ~9.7 vol-points on average**. The proxy backtest (a daily-marked variance swap) showed **Sharpe ~1.8**, positive in both halves, surviving cost stress and beating ~96% of placebo shuffles. Unlike every directional signal, it didn't reshuffle — because it isn't a prediction. You're being *paid to bear tail risk*, like an insurer. That premium is real.

**But the tradeable version failed.** We modeled the thing you'd actually put on — a monthly **defined-risk iron condor** (sell a strangle, buy wings to cap the loss), priced on real DVOL with the real capped payoff. Across strikes (1σ, 1.5σ) and tenors (30d, 45d) it came out **net-negative: Sharpe −0.04 to −0.8**, with an 82–91% win rate masking a few −100% capped tails that wiped out all the small wins. The diagnostic that explains it: even priced at *realized* vol (no premium), the structure loses **~−5%/cycle** — because **crypto's endpoints are fatter-tailed than daily vol predicts**, so a static short structure held to expiry gets run over by jumps. The premium (+2.5%/cycle) couldn't cover the fat-tail drag.

**The reconciliation:** the ~1.8 proxy is a *continuously delta-hedged* variance swap — it harvests implied-minus-realized variance regardless of where price ends up. The condor is a *static endpoint bet* — no hedging — so the tail kills it. Same premium, completely different risk. The trade that actually monetizes the premium is a **delta-hedged short straddle** (rehedge daily) — which is the operationally-intensive, gap-risky institutional version, and we **chose not to pursue it** (a small trader selling vol they aren't fully confident in is exactly how blow-ups happen).

So VRP lands in its own honest category: **a real edge that a small retail trader can't cleanly harvest.** The simple, defined-risk, set-and-forget version loses to crypto's fat tails; the version that works requires daily hedging, real tail risk, and a conviction we didn't have. The premium is genuine — the harvest isn't simple.

---

## The two that lasted longest — both failed the 5-year bar

**Intraday-TSM** was the lone directional edge to clear the cost wall in testing: early-session return predicts late-session return, trade the high-vol days both directions, hold to day-close. It looked like the answer:

- **1-year backtest:** $5,000 → **$5,799 (+16%)**, ~+0.28%/trade gross.
- **5-year backtest (the real test):** $5,000 → **$4,139 (−17%, CAGR −3.8%/yr)**, **max drawdown −52%**, Sharpe **0.22**. The +16% was carried almost entirely by a single explosive month (Jan 2026 +20%). Across a full cycle including the 2021–22 bear, it bleeds.

A Sharpe of 0.22 is statistically indistinguishable from noise. **The one-year window was the small-sample mirage we'd warned about all along.** The hardening (vol-target + regime gate) genuinely works — it cut the worst month from −211% to −24% and max-DD from 511% to 92% — but sound risk machinery on top of a too-thin edge still nets ~nothing.

**Live-trader footnote:** when we put it on testnet it lost −$41 (27% win) while the paper trial showed +0.84% (69% win). The diagnostic proved this was a *wiring bug* (the live bot took LONGs the trial gated out), now fixed — but it's a clean live demonstration of the same lesson: paper always flatters, real costs and real execution only subtract.

**The "careful bets" idea was tested — and rejected.** The hope was that the failure lived in low-conviction days, so trading only the very biggest morning moves would concentrate a stronger edge. The selectivity ladder (top ⅓ → ¼ → 15% → 10% → 5%) said the opposite: OOS Sharpe got *worse* as we got pickier (0.11 → −0.00 → 0.05 → −0.42 → −0.26). The mechanism is clean — **the biggest morning moves are usually overreactions (news shocks, liquidation spikes) that mean-revert in the afternoon**, the opposite of the continuation the strategy bets on. So filtering *for* the biggest moves filters *for* the reversals. Selectivity is not the missing piece. Intraday-TSM is now kept alive **only as a live testnet forward test** — not because the backtests are promising, but to watch the real-money-shaped result accumulate.

**Stablecoin-flow also failed its 5-year test.** On one year it looked strong (Sharpe ~1.3) — but that was beating a *negative* buy-and-hold in a flat year. Over five years: **−8.4% total, Sharpe −0.17, while buy-and-hold returned +40% (Sharpe 0.40).** A market-timing overlay that lost money *while the market rose*. The IS/OOS split confirms regime luck, not signal: **IS −0.61 → OOS +0.32**, and a single outlier month (2024-11, +12.1%) carries most of the recent gain — the Jan-2026 pattern again. 30 of 60 months positive is an exact coin flip. The signal is real as *context* (supply contraction is mildly bearish) but not a standalone edge.

**Where that leaves us:** every directional and timing road has now cleared exactly nothing on the 5-year, multi-regime, out-of-sample bar. The one family never disproved — because never built — is the structural **"be the house"** set: DeFi liquidity-provider vaults (HLP-style) and the volatility risk premium. They collect a premium for providing liquidity or insurance instead of predicting direction, so they don't hit the cost wall that killed everything here. That's the honest next road, and it gets the same backtest-before-a-cent treatment as everything above.

---

## The Simons lens — why fast retail prediction is *structurally* impossible

We studied how Medallion got its returns. The famous "66%" is the **annual return, not the win rate** — their win rate was about **50.75%**. They made a fortune from a *tiny* per-trade edge (0.01–0.05%) multiplied by **~300,000 trades/day**, ~12.5× leverage, and **near-zero per-trade cost** (they're the ones being paid the spread).

Our per-trade cost (**15–20 bps**) is **10–20× bigger than the fast edges we were chasing.** So the entire "fast directional scalp" category is closed to us by arithmetic — not skill, not model quality. The only edges that can work for us are the **inverse**: slow, fatter-per-trade, and structural, so the edge clears our fat cost. Every survivor we found does exactly that; every failure was a fast edge drowned by cost.

---

## The honest meta-lesson

The single most reliable tell across all of this: **the in-sample → out-of-sample collapse.**

- Rule search: 86% → 53% win.
- Strategy search greens: 79–88% IS → 54–58% forward.
- Intraday-TSM: +16% on the cherry year → −17% over five.
- Stablecoin-flow: Sharpe 1.3 on the cherry year → −0.17 over five (IS −0.61 → OOS +0.32).

Every time we got excited, it was a small sample or an in-sample fit. Every time we demanded honest OOS / forward / multi-regime evidence, the edge evaporated. **The discipline of refusing to trust the flattering number is the most valuable thing this project produced** — it's what keeps real money safe. The failures weren't wasted; they're the map of where the edge *isn't*, drawn cheaply on testnet — and they point clearly at the one road left: stop predicting, and collect a structural premium instead.

---

## ADDENDUM 2026-07-04 — the audit-fixed backtest round (grid, fade, split-hour, allocator)

Run after the equity-carry/lookahead bugs were fixed (see `Audit_2026-07-03.md` — pre-fix grid numbers were inflated and are void). 5-year bar, real fees, IS/OOS + breadth throughout.

| Strategy | Best honest result | Verdict |
|---|---|---|
| **Grid, no gate** (the sweep's own IS-winner: step 1.0%, band 8) | OOS Sharpe **0.20**, **+5.5%** over ~2yr OOS, breadth 4/6 | ❌ Economically dead — thinner than funding carry on idle cash; not worth infra |
| **Grid + exit gate** | BTC **−63.8%**, ETH **−84.1%**, fees 35–49% of capital | ❌ A sell-the-bottom fee machine |
| **Grid + freeze gate** | ≈ no-gate minus a little, every config | ❌ Gate adds nothing |
| **Grid + hedge gate** (the freeze+perp-short idea) | BTC **−58%**, ETH **−67%** — churned to death across ~280 regime flips | ❌ Creative idea, honestly killed |
| **Morning-fade** (mirror of careful-bets) | OOS **−0.21%/trade**, breadth **2/7**; ride-mirror ALSO negative | ❌ No signal either direction at the extreme — the toll eats all of it |
| **Split-hour TSM** (is 08:00 special?) | Green OOS at 12/16 UTC but **negative IS at those same hours** | ❌ Noise (real edges show in both halves). No hour-hopping; weakens the TSM prior |
| **Analyst-regime allocator — trend-rider** | Beat B&H OOS **6/7** (median Sharpe **0.58 vs 0.27**, maxDD **34% vs 77%**), consistent IS too | 🟡 **PASS pre-registered bar → shadow forward test** (`trend_sleeve_forward.py`) |
| **Analyst-regime allocator — default-long** | Beat B&H OOS **7/7** (median Sharpe **0.82**), but IS weak and DD 62% | 🟡 Shadow forward test alongside trend-rider |

**The pattern, again:** everything that *trades a lot* (grid rungs, fades, gate churn) died of fees; the one thing that passed *trades ~2 flips a month* and leans on the single classifier output that was calibrated as RELIABLE in both halves (SIGNAL_CARD). Slow, fat, structural — same lesson as ever.

**Grid loose end (one run, then closed):** the sweep gated on the internal ER classifier; the original thesis named the *analyst* as the brain. Final check: `grid_backtest.py --days 1650 --regime-csv data_cache/analyst_lean.csv`. If that doesn't materially beat the no-gate anchor either, the grid file closes for good.

## Sources (internal)
- `Kronos_Search_Runs_Tracker.md` — Runs A–D, OOS Sharpes, win-rate collapse
- `CAPTAINS_LOG.md` — meta-labeler v1 33.3% vs 61.9%, ADA shorts 7W/13L
- `Creative_Edges_Deep_Dive.md` — cost-wall framing, structural premia, sources
- `Backup_Strategy_Options.md` — funding-arb / pairs research figures
- `consensus_backtest.py`, `reversion_backtest.py`, `ofi_maker_backtest.py`, `leadlag_scanner.py`, `funding_backtest.py`, `pairs_backtest.py`, `vol_premium_backtest.py`, `intraday_tsm_equity.py` — methods + verdicts

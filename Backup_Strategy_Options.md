# Kronos — Backup Strategy Options

*A research memo: what to try if the meta-learner never finds an edge.*
*Compiled June 2026. Research notes, not financial advice. Everything here assumes testnet/paper validation first.*

---

## The honest framing

The thing to internalize up front: **directional price prediction on crypto failing to find a retail edge is the normal result, not your mistake.** The literature is blunt about it — naive baselines frequently beat complex ML, markets are non-stationary, microstructure noise and wash-trading corrupt the very features models train on, and most apparent edge is overfitting that evaporates out-of-sample. Marcos López de Prado's well-known list is literally titled *"The 10 Reasons Most Machine Learning Funds Fail."*

So pivoting isn't admitting defeat. It's following the evidence toward the kinds of edge that are actually durable for a small operator. And there's a clear pattern in what survives: **the robust edges are structural and market-neutral — they earn from the mechanics of the market (funding payments, mean-reverting spreads) rather than from forecasting direction.** That matters because forecasting direction is exactly the thing that isn't working.

Your current build is *not* wasted either way. Clean observation data, honest walk-forward validation, a real-time dashboard, systemd deployment — that's good infrastructure that most of these alternatives can reuse.

---

## Options at a glance

| # | Strategy | Edge type | Achievability | Risk profile | Fits your stack? |
|---|----------|-----------|---------------|--------------|------------------|
| 1 | Funding-rate arbitrage | Structural / market-neutral | High | Low (if managed) | Strong — esp. with Hyperliquid |
| 2 | Pairs / stat-arb (cointegration) | Mean-reversion / market-neutral | High | Low–medium | Strong — reuses your data infra |
| 3 | Fix the ML (López de Prado rigor) | Predictive (corrected) | Medium | Medium | Direct — improves what you have |
| 4 | Cross-sectional momentum | Factor / longer-horizon | High | Medium | Good — uses your 8-coin universe |
| 5 | Market making | Structural (spread capture) | Low | Medium–high | Weak on testnet |
| 6 | Deep reinforcement learning | Predictive | Low | High (overfit) | Research only |

---

## Tier 1 — Structural, market-neutral (best fit)

### 1. Funding-rate arbitrage (cash-and-carry)

**The idea.** Hold spot long and short the perpetual future on the same coin (or the reverse when funding is negative). You're delta-neutral — direction doesn't matter — and you collect the funding payment that longs pay shorts (or vice versa). This is the closest thing to a "structural yield" in crypto.

**Why it's compelling.** 2025 studies put market-neutral funding arb around **~19% annualized with maximum drawdowns under 2%**, and one academic study across BTC/ETH/XRP/BNB/SOL reported up to 115.9% over six months with worst-case loss of 1.92%. Funding rates actually rose ~50% from 2024 (stabilizing near 0.015% per 8h). It's the strategy professional desks run with billions.

**Why it fits *you* specifically.** It sidesteps the unsolved problem (prediction) entirely, and it connects directly to your Hyperliquid/HYPE interest: Hyperliquid pays funding **hourly** (capped 4%/hr), exposes it all via API, and lets anyone spin up a **vault from 100 USDC**. Hummingbot already has a documented funding-arb + Hyperliquid-vault workflow.

**The honest caveats.** Returns are compressing as retail piles in ("the easiest trade retail can do"). You need two legs (spot + perp), which means margin management, borrow/financing costs, and real tail risk if the short leg gets liquidated in a violent move — the "market-neutral" label only holds if you actively manage margin and rebalance. Counterparty/exchange risk is real. Net yields after fees are thinner than the headline numbers.

**First step.** Paper-trade a single-coin BTC cash-and-carry: track funding accrual vs. financing + fees, and stress-test what a 20% gap move does to your margin. If the net is positive and the margin math survives, scale to a basket.

### 2. Pairs trading / statistical arbitrage (cointegration)

**The idea.** Find two coins whose prices move together over time (cointegrated — e.g., ETH/BTC, or two L1s). When the spread between them stretches abnormally wide, short the rich one and long the cheap one, betting the spread reverts. Market-neutral again.

**Why it's compelling.** Recent work (including copula-based extensions over 2019–2024 data on ten majors) shows cointegration pairs trading **consistently beats passive holding on a risk-adjusted basis while keeping low market exposure.** It's a decades-old equities technique with a solid theoretical base, and it reuses almost all of your existing data pipeline.

**The honest caveats.** Cointegration relationships break — especially across regime shifts — so you need ongoing re-testing (rolling cointegration tests) and a hard stop for when a pair decouples. Spreads are thin, so transaction costs and slippage can eat the whole edge; backtests *must* include realistic frictions or they lie to you.

**First step.** Run rolling cointegration tests (Engle-Granger / Johansen) across your 8-coin universe, pick the 2–3 most stable pairs, and backtest a simple z-score entry/exit with realistic fees. Your walk-forward discipline carries straight over.

---

## Tier 2 — Improve what you have / robust factors

### 3. Fix the ML the López de Prado way (don't abandon — correct)

Before giving up on the meta-learner, it's worth knowing that the difference between the ML funds that fail and the few that work is *methodology*, and you already have most of the pieces:

- **Triple-barrier labeling** — label each setup by which of take-profit / stop-loss / timeout hits first. You *already do this* with your bracket replay; formalizing it is nearly free.
- **Meta-labeling done properly** — separate the *side* decision (Kronos) from the *size/confidence* decision (the gate). That's literally your architecture; the concept is sound.
- **Purged + embargoed cross-validation** — your expanding-window walk-forward is good, but add *purging* (drop training samples whose label window overlaps the test set) and an *embargo* gap. Overlapping bars are exactly the correlation problem you raised about trades-per-symbol — this is the rigorous fix.
- **Sample-uniqueness weighting** — down-weight overlapping/concurrent samples so near-duplicates don't inflate confidence.
- **Fractional differentiation** — make features stationary without destroying memory.

This is the "make the current approach actually correct" path. Your instinct to gather more clean data is right; this adds the statistical rigor that prevents fooling yourself. If an edge exists in these features, this is what surfaces it; if it doesn't, this is what proves it cleanly.

### 4. Cross-sectional momentum

**The idea.** Each week, rank your coins by trailing ~30-day return, go long the top performers and (optionally) short the bottom, rebalance. No prediction model — just sorting.

**Why it's compelling.** Momentum is documented as a *"pervasive and persistent anomaly"* in crypto. Cross-sectional winners showed ~1.65% mean weekly return at an annualized Sharpe ~1.28 in the research. It's dead simple, robust, and uses the exact 8-coin universe you just expanded to.

**The honest caveats.** It's a days-to-weeks horizon, not scalping — a different cadence than your current bot, though it could run alongside as a separate sleeve. Momentum suffers sharp crashes at reversals; position sizing and a trend filter matter.

---

## Tier 3 — High effort, lower odds

### 5. Market making (Hummingbot)

Provide liquidity on both sides of the book, earn the spread plus any rebates/liquidity-mining. Structurally sound in theory, but the 2025–2026 consensus is sobering: *"mixed profitability… steep learning curve… arbitrage harder than it looks because liquidity, fees and slippage often wipe out the edge."* It's latency- and competition-sensitive, and — critically for you — **you can't realistically validate fills on testnet**, where the thin book gives fake fills (the exact phantom-fill problem you already hit). File under "maybe later, with real capital and careful tuning," not a quick backup.

### 6. Deep reinforcement learning

Tempting, but the honest research is mostly about *how not to fool yourself*: DRL agents "optimistically report increased profits in backtesting which may suffer from false positives due to overfitting," and the better 2025 papers are largely contributions to overfitting *detection*. Low signal-to-noise + non-stationarity is hostile to RL. Treat as a research curiosity, not a backup plan you'd stake on.

---

## Don't rebuild — borrow proven infrastructure

If you pivot, these open-source frameworks save months:

- **Hummingbot** (Apache-2.0, 50+ CEX/DEX connectors **including Hyperliquid**) — the standard for **market making and funding-rate arbitrage / vaults**. The natural home for Option 1.
- **Freqtrade** (25k+ stars, 30+ exchanges) — signal-driven spot/futures with strong backtesting and hyperparameter optimization. Good for rule-based strategies and Option 4.
- **Jesse** (MIT) — **the most honest backtester in open source (zero look-ahead bias).** Many bots bake in look-ahead so backtests look great and live disappoints. Use Jesse (or your own purged walk-forward) to *validate any new strategy before it touches live*.

A recurring warning across all of them: look-ahead bias and unrealistic fill/fee assumptions make backtests lie. Your existing walk-forward discipline already respects this — keep that bar high for anything new.

---

## Recommendation

1. **Keep the current system running** — it's collecting clean, diverse data and costs nothing to let ride. Let the observation-journal verdict come in honestly.
2. **Prototype funding-rate arbitrage next** — it's the best-fit pivot: market-neutral (doesn't need the prediction that isn't working), structurally sound, well-documented returns, and it lines up with your Hyperliquid interest. Start with one coin, paper-traded, with the margin/tail-risk math done explicitly.
3. **Keep pairs trading as the second market-neutral leg** — cheap to prototype on your existing data.
4. **If you stay with ML, only do it with purged CV + sample weighting** — that's the line between rigor and self-deception.
5. **Validate everything in Jesse / purged walk-forward with realistic costs before going live.**

The throughline: stop trying to predict *where* price goes, and start harvesting edges that exist in the market's *structure*. That's where small operators actually make money.

---

## Sources

- [The 10 Reasons Most Machine Learning Funds Fail — López de Prado (GARP)](https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf)
- [Cryptocurrency Price Forecasting Using ML (arXiv)](https://arxiv.org/pdf/2508.01419)
- [Deep learning for Bitcoin price direction — models compared (Financial Innovation)](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00643-1)
- [Funding Rate Arbitrage: Complete Guide 2025 (CoinCryptoRank)](https://coincryptorank.com/blog/funding-rate-arbitrage)
- [Perpetual Funding Rate Arbitrage Strategy 2025 (Gate Learn)](https://www.gate.com/learn/articles/perpetual-contract-funding-rate-arbitrage/2166)
- [The Two-Tiered Structure of Crypto Funding Rate Markets (MDPI)](https://www.mdpi.com/2227-7390/14/2/346)
- [Funding Rate Arbitrage and Creating Vaults on Hyperliquid (Hummingbot)](https://hummingbot.org/blog/funding-rate-arbitrage-and-creating-vaults-on-hyperliquid/)
- [Hyperliquid Funding — Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding)
- [Copula-based trading of cointegrated cryptocurrency pairs (Financial Innovation)](https://link.springer.com/article/10.1186/s40854-024-00702-7)
- [3 Statistical Arbitrage Strategies in Crypto (CoinAPI)](https://www.coinapi.io/blog/3-statistical-arbitrage-strategies-in-crypto)
- [Meta-Labeling (Wikipedia)](https://en.wikipedia.org/wiki/Meta-Labeling) · [Purged cross-validation (Wikipedia)](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [Does Meta-Labeling Add to Signal Efficacy? (Hudson & Thames)](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
- [Cross-sectional Momentum in Cryptocurrency Markets — Drogen, Hoffstein, Otte (SSRN)](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4337066_code2135545.pdf?abstractid=4322637&mirid=1)
- [Deep RL for Crypto Trading: Addressing Backtest Overfitting (arXiv)](https://arxiv.org/abs/2209.05559)
- [Best Freqtrade Alternatives in 2026, compared](https://alexbobes.com/crypto/best-freqtrade-alternatives/)
- [Hummingbot Review 2026 (Finestel)](https://finestel.com/blog/hummingbot-review/)

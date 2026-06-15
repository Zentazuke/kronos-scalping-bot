# Kronos — Master Roadmap

*One place for everything we've built, concluded, and still plan to do. Updated June 2026, after the clean-data verdict on the directional model.*

---

## The through-line (the rule that earned its keep)

**Validate the edge before building the bot.** We did not promote the meta-model until leak-free, multiple-testing-corrected data said so — and it said *no*, saving months of chasing an 81%-accuracy mirage. Every track below obeys the same rule: prove it on real data, net of costs, before committing effort or capital.

---

## Status at a glance

| Area | State |
|------|-------|
| Server deployment (Hetzner, systemd, Tailscale) | ✅ Done, running 24/7 |
| Strategy equity (10k + realized + unrealized) | ✅ Done |
| 8 symbols (BTC ADA ETH BNB SOL XRP DOGE LINK) | ✅ Done |
| Mainnet macro feed (250 daily candles) | ✅ Done |
| Dashboard: analytics, candlestick chart, mobile | ✅ Done |
| Dashboard: walk-forward verdict panel | ✅ Done |
| TA Signals matrix (10 indicators × symbol, 5m/15m/1h) | ✅ Done |
| Phantom-fill filter | ✅ Done |
| XGBoost option | ✅ Done |
| Observation journal + offline labeler (10× clean data) | ✅ Done, running |
| Nightly learning timer (label + walk-forward) | ✅ Done |
| Purged CV + Deflated Sharpe | ✅ Done |
| **Verdict: directional meta-model has no edge** | ✅ **Concluded** |
| Funding-arb feasibility study | ⏭ Next |
| Backups in reserve (pairs, momentum) | 🅿 Held |
| Loose ends (memo update, dashboard PSR/DSR, etc.) | 🅿 Optional |

---

## Track A — Directional meta-model · CONCLUDED

**Outcome:** definitively no tradeable edge. On clean, mainnet-labeled data (241 trades), purged cross-validation gave a filtered Sharpe of −0.07 to −0.14 (still negative), a Probabilistic Sharpe near 0–6%, and a **Deflated Sharpe of 0%**. The deeper finding: the directional *entries themselves* are net-losing on real price paths (take-all Sharpe −0.16) — testnet's phantom fills had been flattering a strategy that actually bleeds. Pruning or more data can't fix a negative-expectancy base strategy.

**Decision:** shelve, don't delete. The data and the validation machinery stay; the trading logic is paused, not removed.

**Optional cleanup (low priority, only if useful later):**
- Surface PSR/DSR on the dashboard walk-forward panel (data already in `walkforward.json`).
- Feature-pruning *diagnostic* (a correlation/importance map) — we agreed to weigh this carefully; now mostly academic since the entries, not the features, are the problem.
- Raise the labeler cadence (it runs daily at 06:00 UTC; observations pile up faster than that) — a quick timer change if we keep collecting.
- Kelly live sizing — **moot** (no edge to size).

---

## Track B — Funding-rate arbitrage · PRIMARY NEXT

The pivot the research kept pointing to: market-neutral, structural, and it doesn't need the direction prediction that's now proven not to work. Same three-gate discipline.

**Phase 1 — Feasibility study (no trading, no risk).**
Build `funding_study.py` (same pattern as the labeler): fetch real funding-rate history + spot/perp prices via ccxt on the server, simulate cash-and-carry (long spot / short perp) net of fees and financing, and report net APR, drawdown, and how often funding flips negative. *We read the verdict together — exactly like the walk-forward.*
- Venue: start on **Binance** (spot + perp on one exchange = clean cash-and-carry; keys already in place). Add **Hyperliquid** after (hourly funding; your HYPE interest), accepting its perps-only hedge complexity and the demonstrated vault tail risk (JELLY, March 2025).
- **Gate:** proceed only if net carry is convincingly positive after realistic costs.

**Phase 2 — Paper trade.** Run live-but-fake across at least one funding-sign flip; confirm the simulated edge survives real fills and timing.

**Phase 3 — Small live.** Money you can lose, margin/liquidation math explicit, hard kill-switch outside the strategy. Scale only on a real track record.

---

## Track C — Backups in reserve (if funding arb fails Phase 1)

Held, validated-but-sober, ready if needed:
- **Pairs trading / stat-arb** — market-neutral mean reversion on cointegrated coins; reuses the existing data pipeline. Watch for cointegration breaking at regime shifts.
- **Cross-sectional momentum** — rank the 8 coins by trailing return, long winners / short losers, weekly. Simple, robust, but decayed post-2021 and carries crash risk.
- **Market making / RL** — explicitly deprioritized: market making can't be validated on testnet's phantom fills; RL is overfit-prone. Reserve only.

Full pros/cons live in `Backup_Strategy_Options.md` (which still leads with the rosy first-pass numbers — **a loose end is to revise it to the validated, sober ranges**).

---

## Infrastructure that transfers to everything

None of this is thrown away by the pivot:
- Hetzner server + systemd services + Tailscale access.
- The dashboard (charts, panels, TA signals) — re-pointable at any strategy.
- The data pipeline (ccxt fetching, candle cache, journals).
- **The validation machinery (purged CV + Deflated Sharpe)** — the single most valuable asset; it works on *any* strategy's returns and is what lets you trust a verdict.
- The open-source options if we ever stop hand-rolling: **Hummingbot** (funding arb + Hyperliquid vaults out of the box), **Jesse** (the most honest backtester), **Freqtrade**, **NautilusTrader**.

---

## Recommended sequence

1. **Build & run the funding-arb feasibility study (Phase 1).** ← the active next step
2. Read the verdict together; **gate** to paper trading only if net carry is real.
3. In parallel, keep the observation journal filling (free) and, if we want, tidy the loose ends (revise the backup memo, surface PSR/DSR).
4. If funding arb passes → paper → small live. If it fails feasibility → pairs trading study next, same discipline.

**The mindset that got us here:** every step is a hypothesis tested on real data before it earns the next step. That's why a "negative" result this week is actually a win — you know, instead of guess.

# Crypto-Analyst × Intraday-TSM — Integration Plan
*2026-06-29 · Drop the external sentiment app; use the analyst's own context.*

---

## 1. What the analyst actually is (and where its edge really lives)

The crypto-analyst is a well-built, **honest** analytics engine — and crucially, it reaches the *same* conclusion this project did. From its own README and auto-generated model card:

- **Causal, TA-Lib-verified indicators** (ATR/EMA/RSI/MACD/Bollinger/ADX/realized-vol), a **regime classifier** (trend/range/breakout/flash/low-vol with confidence + reasons), **microstructure** (spread, depth, OI, funding, basis), **on-chain**, and **crowd positioning** (long/short ratio, funding-crowding).
- A **calibrated directional ensemble** (`ensemble.py`) that weights each signal by its *measured* out-of-sample edge — anti-predictive signals get negative weight, useless ones get ~zero. Every number is reported with a Wilson 95% CI and only called an `EDGE` when the interval excludes the base rate.
- Its honest headline: *"short-horizon single-asset direction is near the predictability floor; the transparent rule classifier wins OOS; neither ML nor funding features beat it. Funding is kept as risk-context, not a directional predictor."* **This is exactly our cost-wall finding, independently reproduced.**

**Where the analyst has a real, calibrated edge** (from MODEL_CARD.md — the part that matters most):

| Signal | 1h | 4h | 1D |
|---|---|---|---|
| BTC ensemble lean | 1.05× · no edge | **1.09× EDGE** | **1.24× EDGE** (66% vs 53%) |
| ETH ensemble lean | 0.99× · no edge | **1.17× EDGE** (64% vs 54%) | 1.09× · no edge |
| BTC/ETH trend label | mixed | **1.30–1.58× EDGE** | partial |
| Funding-crowding contrarian | no edge | — | — |
| On-chain divergence (ETH) | — | — | **1.11× EDGE** |

The takeaway that drives the whole design: **the analyst's edge is on the 4h/1D regime and ensemble, not on 1h.** Intraday-TSM is a *daily-horizon* bet (decide 08:00 UTC, hold to 00:00). So the analyst's higher-timeframe lean is the *right altitude* to inform it.

---

## 2. The integration thesis — and why it's different from what just failed

We just proved that making intraday-TSM **more selective by morning-move size** doesn't help (pickier = worse OOS, because the biggest moves are overreactions that revert). That failed because it sliced on the *same* information the strategy already uses.

This integration is different in kind: the analyst supplies an **orthogonal, independently-calibrated, higher-timeframe signal** — the 1D/4h regime and ensemble lean. The hypothesis:

> Intraday-TSM should only press its momentum bet when the **higher-timeframe regime agrees**. Buy the morning-up day when the 1D lean is bullish / trend_up; sell the morning-down day when it's bearish / trend_down; **stand down when the analyst flags `flash` or `low-vol`** (momentum unreliable) or the two disagree.

Two thin-but-*real*, *orthogonal* edges (intraday session momentum + higher-timeframe regime) combining is genuine diversification — the one move that can lift a marginal edge. It might work where selectivity didn't. **It also might not** — and we hold the same bar: it has to prove itself on the 5-year out-of-sample backtest before it touches the live trader.

---

## 3. What we consume — and what we drop

**Consume (all self-contained in the analyst, no external deps):**

| Analyst output | Endpoint | Use in intraday |
|---|---|---|
| Calibrated 1D + 4h **lean** (bullish/bearish/neutral + OOS calibration) | `GET /ensemble/{sym}?timeframe=1d` | **Directional-agreement gate** |
| **Regime** label + confidence | `GET /regime/{sym}?timeframe=4h` | **Stand-down gate** (skip flash / low-vol) |
| Microstructure: funding, spread, OI, basis | `GET /microstructure/{sym}` | **Risk-context** (skip extreme funding / blown spread) |
| On-chain divergence (ETH has EDGE) | `GET /onchain/{sym}` | Optional secondary gate |
| Crowd positioning (long/short ratio, funding-crowding) | native | The analyst's **own sentiment substitute** |

**Drop — the external sentiment app:** Launch with **`run-analyst.bat`, not `run-all.bat`** (run-all also spins up the separate sentiment engine). The bot integration never calls `/sentiment` (that endpoint proxies the external engine and returns `available:false` gracefully when it's off). The analyst's **native crowd-positioning + funding-crowding + on-chain** *are* its market-sentiment read — and they're more robust than news/social sentiment for trading anyway. The model card confirms the cost of dropping it is ~nil: funding-crowding shows **no edge**, on-chain only a small edge for ETH. Nothing of measured value is lost.

---

## 4. Architecture — backtest first, live only if it passes

Two projects, two venvs, two data stores. Keep them **loosely coupled** so neither entangles the other:

```
crypto-analyst  ──(causal lean/regime, replayed over history)──>  analyst_lean.csv
                                                                        │
Trading Bot:  intraday_tsm_analyst.py  ◀── reads CSV ──────────────────┘
              (gates intraday trades on the analyst lean, reports OOS)

   ── if and only if OOS improves ──>

crypto-analyst API (run-analyst.bat, localhost:8000)
                                          ▲
Trading Bot live:  intraday_tsm_forward / _live  ── HTTP GET /ensemble,/regime ──┘
              (gate today's decision on the live lean before placing the order)
```

**Why CSV for the backtest, HTTP for live** — mirrors the pattern that already works here (`fetch_data.py` caches series the sandbox can't fetch). The analyst's ensemble/regime are causal and walk-forward by construction, so a one-shot exporter can dump a clean **daily lean series over 5 years** that the intraday backtest reads — no live calls, no lookahead.

---

## 5. The gate logic (precise, low-parameter)

For each intraday decision day `d`, after the existing vol-gate fires a candidate LONG/SHORT:

1. **Directional agreement** — take the trade only if the analyst's as-of-`d` **1D lean** agrees with the morning direction (LONG needs lean ≠ bearish; SHORT needs lean ≠ bullish). Optionally require the 4h lean to agree too (stricter).
2. **Regime stand-down** — skip the day if the analyst's 4h/1D regime is `flash` or `low_vol` (momentum unreliable there).
3. **Risk-context veto** — skip if funding is in a crowded extreme or spread is blown out (the analyst's microstructure read).
4. Keep the existing **vol-target sizing** and the on-chain risk-off gate (now on DefiLlama data).

**Market-regime simplification (recommended first cut):** the analyst is only calibrated on **BTC + ETH** so far, but crypto is highly correlated. Use **BTC's 1D ensemble lean as a single market-regime gate for the whole basket** (ETH/SOL/XRP/DOGE/LINK/AVAX). One gate, one parameter set, minimal overfitting surface — and it directly tests "does higher-timeframe market regime improve intraday." If that helps, *then* consider per-coin analyst calibration.

---

## 6. Phased plan

**Phase 0 — confirm the data bridge (½ day).**
Run `run-analyst.bat`; hit `GET /ensemble/BTC_USDT?timeframe=1d` and `/regime`. Confirm shapes, `lean`, `score`, `calibration{n,acc,base,lift}`. No code yet — just verify the contract.

**Phase 1 — export the historical lean (1 day).**
Add `scripts/export_lean.py` to the analyst: replay its causal ensemble + regime over 5 years of BTC (and ETH) 1D/4h history, write `data/analyst_lean.csv` → `(date, symbol, tf, lean, score, regime, confidence)`. Reuse `ensemble_read`'s walk-forward weights so it's honest. Copy the CSV into the Trading Bot's `data_cache/`.

**Phase 2 — the backtest (1–2 days, the decisive step).**
Build `intraday_tsm_analyst.py` in Trading Bot: same harness as `intraday_tsm_strategy.py`, but add the analyst gate (read `analyst_lean.csv`, align by date). Report the variant ladder — **base intraday / + analyst-agreement / + regime-standdown / + both** — with the **5-year OOS column**, exactly like every other test. Decision rule: **proceed only if OOS Sharpe improves AND holds in both halves.** If not, we've cleanly learned the analyst doesn't rescue intraday, and we stop (cheap, honest).

**Phase 3 — live wiring (only if Phase 2 passes).**
Point the forward logger / live trader at the analyst API: at 08:00 UTC decision time, `GET /ensemble` + `/regime` for the market gate, apply the same logic, then place (or skip) the order. Fail-open if the analyst is unreachable (log + fall back to the ungated decision, never crash). Add an "analyst lean" line to the dashboard's intraday panel.

**Phase 4 — forward-test & fold into the postmortem.**
Let the gated version run live alongside the ungated one; compare realized P&L. Update `Failed_Strategies_Postmortem.md` with the verdict either way.

---

## 7. The honesty bar (non-negotiable, same as everything else)

- **The analyst's edge is small** (lift 1.1–1.3×) and intraday's is thin. Combining them *can* help via orthogonality, but that's a hypothesis, not a result. The 5-year OOS backtest is the only judge.
- **More gates = more overfitting surface.** Mitigate by using the analyst's **pre-calibrated** weights (never re-tuned to flatter the intraday backtest) and the single BTC market-gate first cut.
- **Kill criteria:** if Phase 2 OOS doesn't cleanly improve in both halves, the integration is rejected and recorded as such. No sweeping knobs until a green appears.
- **Symbol coverage:** the analyst is calibrated on BTC/ETH; SOL/XRP/DOGE/LINK/AVAX would need their own calibration before per-coin gating — which is why the BTC market-gate is the honest first test.

---

## 8. One-paragraph summary

Use the crypto-analyst as a **higher-timeframe regime filter** on intraday-TSM, not a new predictor: gate each intraday momentum bet on the analyst's calibrated 1D/4h **lean** (and stand down in flash/low-vol regimes), starting with **BTC's 1D ensemble as a single market gate** for the whole basket. Couple them by a **CSV export for the backtest** and the **analyst's HTTP API for live**. **Drop the external sentiment engine entirely** — run `run-analyst.bat`, never call `/sentiment`; the analyst's own crowd-positioning, funding, and on-chain context cover market sentiment, and the model card shows those external sentiment signals carry ~no measured edge anyway. Prove it on the 5-year out-of-sample backtest **before** a single live order, exactly as we've done with everything else.

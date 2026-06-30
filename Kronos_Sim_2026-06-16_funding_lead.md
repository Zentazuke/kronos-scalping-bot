# Saved simulation — first green verdict (funding lead)
**Run:** 2026-06-16 08:11 UTC · `observations.db` · 2,689 decided trades · 2-condition combos
**Baseline:** take-all Sharpe −0.085 (47% win) · 2,326 combos searched · noise floor 3.268

**Verdict (dashboard):** *best rule BEATS the noise floor AND holds out-of-sample (OOS Sharpe 0.066 over 39, edge-is-real 66%) — worth a closer look* ✅ first green

---

## Leaderboard (in-sample | out-of-sample)

| rule | Sharpe | n | win | OOS Sharpe | OOS n | OOS win |
|---|---|---|---|---|---|---|
| funding ≥ 7.6e-05 AND macd ≥ 2 | 7.96 | 25 | 100% | 0.07 | 39 | 54% |
| di_align ≥ −0.75 AND funding ≥ 7.6e-05 | 7.23 | 29 | 100% | 0.30 | 83 | 64% |
| funding ≥ 7.6e-05 AND supertrend ≥ 2 | 7.19 | 33 | 100% | 0.40 | 78 | 68% |
| di_align ≥ 8.30 AND funding ≥ 7.6e-05 | 7.05 | 26 | 100% | 0.37 | 66 | 67% |
| consensus ≥ −3 AND macd ≥ 2 | 4.30 | 35 | 100% | 0.04 | 208 | 51% |
| consensus ≥ 1 AND macd ≥ 2 | 4.25 | 33 | 100% | 0.06 | 204 | 53% |
| consensus ≥ 6 AND macd ≥ 1 | 4.16 | 27 | 100% | 0.02 | 215 | 50% |
| consensus ≥ 6 AND macd ≥ 2 | 4.16 | 27 | 100% | 0.11 | 179 | 54% |
| adx ≥ 23.15 AND macd ≥ 2 | 3.94 | 48 | 100% | 0.01 | 150 | 49% |
| funding ≥ 6.9e-05 AND supertrend ≥ 3 | 3.90 | 28 | 100% | 0.37 | 105 | 68% |
| adx ≥ 23.15 AND supertrend ≥ 3 | 3.86 | 51 | 100% | 0.28 | 208 | 62% |
| **funding ≥ 6.9e-05 AND supertrend ≥ 2** | 3.78 | 41 | 100% | **0.48** | **132** | **71%** |

**Surviving ingredients:** consensus×27, supertrend×10, macd×7, funding×6, di_align×4, cci×4, adx×3, boll×3, votes×3, donchian×3, stoch×2, attention×1, fear_greed×1, sent_aligned×1

---

## Honest read

**What's genuinely encouraging**
- First green verdict and first time *any* rules survived the holdout — and it only appeared after the data grew (2,028 → 2,689). Patience surfaced it.
- A coherent cluster keeps recurring: **funding rate + a trend confirmation** (supertrend / macd / di_align). Strongest is `funding ≥ 6.9e-05 AND supertrend ≥ 2` — OOS Sharpe **0.48**, **71% win on 132 holdout trades**. That's not nothing.
- Funding is interesting *because* it's a **structural / crowding signal**, not a price prediction. A real edge is more plausible there than in pure direction-guessing — and it rhymes with the funding/basis backup idea.

**What to be careful about**
- Ignore the 100% in-sample win / Sharpe ~8 columns entirely — that's the search carving a perfect in-sample pocket from few trades (25–51). The **OOS columns are the truth**, and they're modest (Sharpe 0.07–0.48).
- The *headline* rule (funding+macd) has an OOS edge of basically zero (0.066) and "edge-is-real" only **66%** — a 1-in-3 chance it's noise. The verdict is technically green but on a thin margin.
- **Multiple-looks caveat:** this is the first green after many re-runs as data accumulated. The deflation handles multiple testing *within* a run, not the fact that we keep checking and stopping at the first green. A first green can be luck of repeated looking.

**The disciplined next step — do NOT change the bot yet**
This is "worth a closer look," not "deploy." The real test of signal vs. fluke is **persistence**: does the funding + trend cluster show the same OOS strength on the *next* batch of fresh data? Keep Kronos running untouched, re-run in ~a week, and see if `funding ≥ ~7e-05 AND supertrend/macd` holds up again. If it does, that's a genuine lead worth building a focused, isolated test around — and notably one centered on a *structural* signal.

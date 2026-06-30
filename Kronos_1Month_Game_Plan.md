# Kronos — One-Month Game Plan
### Data → Research → Honest Iteration

The whole month runs on one philosophy: **let the bot harvest data, test ideas with the honest search tools we built, and change the live strategy only when a finding earns it.** No tuning on thin data, no acting on in-sample numbers, one bounded change at a time.

---

## The rules we don't break

- **Change nothing live until a finding goes _green_** — beats the deflated noise floor *and* holds on the out-of-sample hold-out. Anything else = no change.
- **One change at a time**, bounded and reversible (a config constant, a single gate), so we can actually attribute the effect.
- **Keep harvesting while we trade.** The observation journal records *every* directional bar regardless of what we trade, so narrowing live trades never starves the learning.
- **Re-validate every change** on fresh data: did the edge the sim predicted actually show up live? If not, revert — it's just config.
- **The search needs volume.** A feature only enters the search once ≥25 observations carry it; the search won't even run until ~75 decided observations exist. Milestones below are anchored to *counts*, not just dates — adjust the calendar to the real numbers.

---

## Week 1 — Foundation & instrumentation

Goal: everything recording cleanly, accumulation automated, a baseline captured.

- Confirm the deploy is in: `ta_consensus` column migrated, sentiment recording, geometry sweep button live. (`journalctl -u kronos-bot | grep "observations migrated"`)
- Confirm the daily scheduled label + search job is running. **Add the geometry sweep to that same daily job** so its history builds on its own.
- Capture a **baseline snapshot**: export the DBs, write down today's numbers — # decided observations, clean win rate, sentiment coverage (rows), TA/consensus coverage (rows).
- Otherwise: hands off. The bot harvests on the current brackets (TP 1.5 / SL 2.5, unchanged).

**Week-1 checkpoint:** sentiment rows climbing, TA + consensus rows climbing, daily history files growing.

---

## Week 2 — Accumulate & first honest look

Goal: cross the threshold where features start entering the search.

- Keep harvesting. No trading changes.
- Once decided observations pass ~75, do the **first real run** of both searches. *Expect* "not enough data / within the noise floor" — that's the honest baseline, not a failure.
- Track coverage: are sentiment and consensus past the 25-row bar to *enter* the search yet? If not, that's the week's bottleneck to watch.
- Export mid-week and eyeball the history trend.

**Week-2 checkpoint:** the searches run end-to-end; we can see which features have enough rows to be in play.

---

## Week 3 — First iteration window

Goal: with ~2–3 weeks of data, the searches finally have power — look for a real edge.

- Run both searches, geometry sweep in coarse-to-fine mode.
- **Decision gate — did anything go green?**
  - *Geometry green* → plan the bracket change (the two ATR constants in `execution.py`), deploy, watch. Highest-leverage lever, so it's first in line.
  - *Entry filter green* → plan it as a pre-trade gate (e.g. "consensus ≥ X AND ADX ≥ Y"), keeping the journal recording everything.
  - *Nothing green* → **no change.** Keep accumulating; note what's trending toward the floor.
- If we make a change, make **only one**, and write down the edge it predicted so we can check it next week.

**Week-3 checkpoint:** either one deliberate, logged change — or a clear "still noise, keep going."

---

## Week 4 — Validate, iterate, decide

Goal: confirm or reject, and make a month-end call.

- If a change was made in Week 3: measure whether the predicted edge **actually showed up live** on the fresh week of data. Keep it only if it did.
- Re-run both searches on the now-larger dataset.
- **Month-end verdict — pick the honest branch:**
  - *A real, surviving edge exists* → keep iterating on it next month (tighten it, test interactions like TA-consensus × sentiment).
  - *Everything is still within the noise floor after a month of real data* → that is itself the answer. Stop torturing the search and seriously evaluate the **backup strategies** we researched (funding / basis being the most grounded). Decide deliberately, not emotionally.

---

## Track every week (the scoreboard)

| Metric | Why it matters |
|---|---|
| # decided observations | Statistical power — drives when searches can speak |
| Clean win rate | The honest baseline (no phantom fills) |
| Feature coverage (sentiment / TA / consensus rows) | Which signals are even testable yet |
| Search verdict + noise-floor trend | Is anything moving toward a real edge? |
| Geometry sweep: best vs live bracket | Is the 0.6 reward:risk actually costing us? |

---

## Parallel research track (light, no build)

Keep the backup-strategy research *warm* but don't build anything: quietly confirm whether funding/basis is even executable on your venue and what the realistic net edge is. This is the safety net if the month-end verdict is "no directional edge." It stays a sketch until the data tells us we need it.

---

## What we are explicitly NOT doing this month

- Not tuning the live bot on thin data.
- Not making more than one change at a time.
- Not acting on a good-looking in-sample number without the hold-out.
- Not extending the search machinery further (coarse-to-fine on entry filters, combined TA+sentiment consensus, etc.) until the current tools have run on real data and earned the next step.

---

*Bottom line: this month is about earning the right to change the strategy, not changing it. By month-end you'll either have a real, validated edge to build on — or a clear, honest signal to pivot. Both are wins.*

# CAPTAIN'S LOG — Kronos Scalping Bot

Handoff document for continuing this project in any session/workspace.
Read this top to bottom before touching code. Last updated: 2026-07-04.
**Start with §8 (the audit sessions) — it is the current state of play.**
§1–7 describe the retired Kronos-scalper era, kept as history.

---

## 1. What this is

Institutional-grade algorithmic scalping bot for **BTC/USDT + ADA/USDT, 5-minute
timeframe, Binance Spot TESTNET** (Stage 1 pilot — no real capital). Built in
phases, each phase one module, each module with an embedded unittest suite.

Project root: `C:\Users\Ricardo\Desktop\Work\PROJECTOS\Trading Bot`
(NOT the WANDERING PIXELS folder — that is an unrelated project.)

**Current test count: 89 passing** — run with:
```
python -m unittest gatekeeper predictor execution journal backtest learner main
```

## 2. Module map (build order = dependency order)

| Module | Phase | Role |
|---|---|---|
| `feed.py` | 1 | CCXT Pro websockets, deque(512) buffers, confirmed-bar dispatch, REST reseed, `adjustForTimeDifference` enabled |
| `gatekeeper.py` | 2 | Decimal indicators: Wilder ATR/ADX(+DI/−DI)/RSI(14), book imbalance; 4 regime sieves (ADX>25, ATR>SMA20(ATR), volume, L2 freshness) + `confluence()` 2-of-3 directional votes (DI / RSI 70-30 / top-5 book lean) |
| `predictor.py` | 3 | Kronos-small (NeoQuasar, HF-cached; repo cloned at `./Kronos`), 30 × sample_count=1 Monte Carlo, edge gate ≥0.53, dead band 0.48–0.52, STRAT_NEUTRAL safe-state on ANY fault |
| `execution.py` | 4 | Sandbox-locked router. Sieves in order: state machine → exchange-truth reconcile → 0.05% slippage → half-Kelly × min(1, ATR_sma/ATR) (5% equity hard cap) → 5% of top-3 L2 depth → quantize → bracket TP 1.5×ATR / SL 2.5×ATR. Spot brackets = ONE OCO list (`privatePostOrderListOco`, LIMIT_MAKER + STOP_LOSS_LIMIT with 0.5% buffer through the stop). `reduceOnly` ONLY on derivatives — spot rejects it (-1104) |
| `main.py` | 5 | Supervisor: .env load → emergency_lock refusal → self-healing bot.lock → sandbox verify → pipeline A (drawdown 3%/day kill switch) → A½ (settle outcomes) → B (regime) → C (inference) → C½ (confluence veto) → D (route) → D½ (journal) |
| `visualizer.py` | 6 | rich Live 4-quadrant Mission Control TUI; drop-oldest queue; encoding fallback for cp1252 (NEVER emit bare `→ ✓ ✗ █` without the glyph probe) |
| `journal.py` | 7 | SQLite trade journal (Decimal as TEXT), OutcomeMonitor polls TP/SL order pair per bar, computes PnL, feeds Kelly tracker; `replay_into()` restores Kelly state on boot |
| `backtest.py` | 8 | Offline replay through the REAL gatekeeper + bracket simulator. No lookahead (entry = next bar open), both-legs-hit = LOSS, DI as direction proxy (no Kronos in backtest). `--walk-forward` for parameter validation |
| `learner.py` | 9 | Meta-labeling: pure-numpy logistic regression over journal features → P(win). Modes via META_FILTER_MODE: off / **shadow (default)** / veto. Dormant under 100 decided trades. Train: `python learner.py train` |

## 3. Non-negotiable contracts (from the original blueprint)

1. **No hardcoded secrets ever** — config only via `os.getenv()` / `.env`.
2. **Decimal domain boundary** — floats live only inside pandas/torch and at the
   CCXT wire seam. Every decision/money number is
   `Decimal(str(x)).quantize(…, ROUND_HALF_EVEN)`.
3. **Sandbox verify before any order routing** (`USE_SANDBOX=True` mandatory;
   ExecutionRouter constructor refuses without it).
4. **Boot refused if `emergency_lock.lock` exists** (kill switch fired; only a
   human removes it).
5. **Single instance** — `bot.lock` with owner PID; stale locks (dead PID)
   self-reclaim, live PID refuses.
6. Every module: `mypy --strict`-style annotations + embedded unittest suite
   with injected fakes, zero network in tests.

## 4. Hard-won gotchas (do not relearn these the painful way)

- **Windows `os.kill(pid, 0)` KILLS the process** (CPython maps it to
  TerminateProcess). Liveness probe uses ctypes OpenProcess — see
  `_pid_is_alive` in main.py.
- **Binance spot ≠ futures**: `reduceOnly` → -1104; plain STOP_LOSS market
  forbidden on majors → use STOP_LOSS_LIMIT; two separate exit orders
  double-lock spot balance → OCO is mandatory; cancel-all with nothing open →
  -2011 OrderNotFound (benign, tolerated).
- **Windows clock drifts ahead of Binance** → -1021 InvalidNonce. Fixed:
  ccxt `adjustForTimeDifference` + resync-and-retry-once in
  `_fetch_total_equity`. If it recurs, also `w32tm /resync` as admin.
- **cp1252 console** cannot encode `█ ─ · ✓ ✗ →` — visualizer probes encoding
  and falls back to ASCII. Any new glyph must go through the same probe.
- **PowerShell 5.1 wraps gh/git stderr in fake errors** — "NativeCommandError"
  with exit 0/255 is usually noise; read the actual output text.
- Spot SHORTs only work while the testnet wallet holds base inventory
  (they sell inventory, buy back cheaper).
- Embedded test fakes: `_FakeExchange` (futures-ish) vs `_FakeSpotExchange`
  (OCO path) in execution.py; supervisor fakes in main.py accept
  `confluence_ok`, order_states for the OutcomeMonitor, etc.
- **Cowork mount sync can serve stale/truncated views of recently edited
  files to the sandbox shell** (Windows-side files stay correct). Workaround:
  rebuild from `git show HEAD:<file>` + re-apply patches, or transfer under a
  new filename; verify with `ast.parse` before running anything.

## 5. Environment (`.env` — real file is NOT in git; see `.env.example`)

```
EXCHANGE_ID=binance          USE_SANDBOX=True (mandatory)
EXCHANGE_API_KEY/SECRET      (Binance Spot Testnet keys, 64-char)
RISK_TOTAL_DRAWDOWN_LIMIT=0.03   ORDER_TIMEOUT_S=2.0 (testnet; ~0.2 live)
KRONOS_REPO_PATH=./Kronos    CONFLUENCE_MIN_VOTES=2
META_FILTER_MODE=shadow      META_MIN_PWIN=0.5
Variant farm: VARIANT=prod|relaxed|harvester, REGIME_ENFORCE=true|false,
CONFLUENCE_ENFORCE=true|false, FIXED_TRADE_NOTIONAL=<quote amount, unset=Kelly>
Tuning knobs: REGIME_MIN_ADX=25, EDGE_THRESHOLD=0.53,
DEAD_BAND_LOW=0.48, DEAD_BAND_HIGH=0.52
Optional: JOURNAL_DB, HEADLESS=true, CONFLUENCE_MIN_VOTES=0 to disable veto
```

Python 3.14 at `C:\Python314`, packages in user-site
(`pandas ccxt torch transformers rich einops python-dotenv numpy`).
If `python main.py` says ModuleNotFoundError: packages were installed via
`python -m pip install …` — must be the SAME interpreter the terminal uses.

## 6. How to run things

```
python main.py                         # live testnet bot + Mission Control TUI
                                       # (HEADLESS=true for log-only mode; logs → bot.log)
python -m unittest <module>            # any single suite
python backtest.py BTC/USDT --days 14  # offline replay (public data, no keys)
python backtest.py BTC/USDT --days 30 --walk-forward
python learner.py train                # trains meta model from journal.db
                                       # (refuses < 100 decided trades)
python learner.py train --db A/journal.db --db C/journal.db  # pooled variants
```

## 6¼. MASTER PLAN v2 (written 2026-06-12 evening — supersedes v1 below)

**PHASE 1 — industrialize the mill (this week)**
1. [USER, in progress] Server: Netcup VPS (Hetzner CX22 was out of stock),
   DEPLOY.md runbook, carry journal.db over, Tailscale for dashboard.
   ONE bot per account — stop the PC bot before starting the server one.
2. [CLAUDE, small build] SYMBOLS env knob + add ETH/USDT, SOL/USDT
   (2-3x data velocity, CPU permitting) + trainer upgrades: walk-forward
   CV instead of single 80/20 holdout, L1 to expose dead features.
3. [USER, any night, on the PC while server trades]
   python calibrate.py BTC/USDT --days 14 --stride 3      (overnight)
   python backtest.py BTC/USDT --days 90 --walk-forward   (fee-aware now)
   python backtest.py ADA/USDT --days 90 --walk-forward

**PHASE 2 — learning-rate multipliers (next full build session)**
4. [CLAUDE, the big one] Observation journal: record the 21-feature vector
   for EVERY evaluated bar (not just routed trades) + offline phantom
   labelling via the backtest bracket simulator -> ~10x training samples
   per day. Real trades remain the gold standard; phantoms are clearly
   flagged in a separate table/column.
5. [CLAUDE, rides along] MFE/MAE columns in OutcomeMonitor (max
   favorable/adverse excursion) -> every real trade re-labelable against
   any hypothetical bracket offline.

**PHASE 3 — first real decisions (data-gated)**
6. Retrain meta v2 at ~250+ decided (or phantom-augmented earlier);
   judge by walk-forward CV stability, not one holdout number.
7. Calibration verdict -> probability shrinkage / dead-band / edge changes.
8. 90d walk-forward verdicts -> confirm or adjust TP/SL/ADX.

**PHASE 4 — the variant farm (blocked on USER: 2 extra testnet accounts)**
9. Three bots on the server (systemd each, separate accounts/folders):
   A prod = enforced gates + Kelly + single position (the CONTROL GROUP);
   B relaxed = enforced but loose (ADX>20, votes=1, edge 0.50 + dead band
   down); C harvester = current config. CPU check with 3x Kronos.
10. M4 promotion: meta shadow record vs unfiltered baseline over ~100+
    trades; veto power only on a win. Pooled training across variants.

**PHASE 5 — frontier (strictly evidence-gated)**
11. XGBoost/LightGBM meta v3 at ~1000+ samples (architecture is already
    model-agnostic; ~50-line learner change).
12. Kronos fine-tune (Phase 10) ONLY if calibration proves the model is
    the bottleneck.
13. Sentiment engine integration — USER's parallel project, his timeline;
    shadow hook already live, journal-first when it matures.
14. Named chart patterns / sub-minute momentum — parked unless evidence
    demands.

**BEFORE ANYTHING RESEMBLING REAL MONEY (unchanged, non-negotiable):**
restore RISK_TOTAL_DRAWDOWN_LIMIT=0.03, MAX_OPEN_TRADES_PER_SYMBOL=1,
Kelly sizing, enforce flags true, spread/fee-aware edge PROVEN positive,
IP-whitelisted keys. The harvester .env is a data rig, never production.

## 6½. MASTER PLAN — everything left, in execution order (written 2026-06-12 ~01:30)

**M0 — user's morning chores (no Claude needed):**
- One bot restart (`start_all.bat`) — loads the sibling-flatten fix
  (running process predates it; dormant bug until a bracket fails).
- Optional: market-buy ~162.3 ADA on testnet to square the UNKNOWN #15–19
  inventory. Optional: ADA balance check so concurrent shorts never starve.

**M1 — partially done 2026-06-12 ~midday:** meta v1 trained at 101 decided
trades: holdout accuracy 33.3% vs 61.9% predict-majority baseline — v1 is
ANTI-predictive (memorized one overnight regime, holdout n≈20, older rows
lack Phase A features). Correctly stays in shadow; its scores now journal
as meta_p_win for the M4 comparison. Retrain at ~250–300 decided across
more market conditions. ADA SHORTS are the bleeding pattern so far
(7W/13L vs BTC shorts 7W/2L, ADA longs 7W/1L as of trade ~#37).

**M1 — read the harvest (evening 2026-06-12):**
- Scheduled task `harvester-24h-report` fires ~18:15 UTC: volume, win rates
  per direction, KRONOS CALIBRATION buckets (the key question), whether the
  disabled gates would have helped, stop slippage.
- If decided trades ≥ 100: train meta v1 (`python learner.py train`), check
  holdout vs predict-majority baseline. KEEP shadow mode. Journal now stores
  meta_p_win per trade → its shadow record accumulates for M4.

**M2 — DONE 2026-06-12 afternoon (suite 111):** feed.py gained a
TradeFlowMonitor: watch_trades worker per symbol (rolling 20k-print window),
level-1 OFI accumulated from every book update (Cont 2014 — _ofi_event),
`feed.trade_flow(symbol)` is READ-AND-RESET, called once per bar at step B¾.
Gatekeeper: microprice gap (Stoikov depth-weighted mid vs mid, bps) +
multi-timeframe context resampled from the in-memory 512-bar window
(trend_1h = close vs 1h SMA10, trend_4h = close vs 4h SMA5, rsi_1h Wilder,
day_range_pos over last 288 bars). 8 new journal columns (auto-migrate):
trade_imbalance, ofi_rel (OFI / candle volume), mvwap_gap_bps,
microprice_gap_bps, trend_1h, trend_4h, rsi_1h, day_range_pos.
Learner FEATURE_NAMES now 17 (v2). IMPORTANT: extending features
deliberately invalidates the saved v1 model (load() rejects feature-name
mismatch) -> meta filter DORMANT until v2 trains at ~250 decided. v1 was
anti-predictive (33%), so nothing of value was lost. feed.py also gained
its first embedded test suite (pure math; add `feed` to the unittest list).

**M2 (original spec, kept for reference):**
- feed.py: `watch_trades` consumer per symbol; rolling 1m/5m windows of
  aggressive buy/sell volume → CVD + trade imbalance; book-snapshot diffs
  accumulated between bars → true OFI (Cont); micro-VWAP from trades;
  microprice gap from the existing snapshot.
- New journal columns (auto-migrate) + meta features v2 + tests.
- Journal-first: NO new gates. Watch CPU (harvester runs Kronos every bar).
- Build AFTER meta v1 is trained so v1's feature set stays stable.

**M3 — BUILT 2026-06-12 (tool ready, run pending):** `calibrate.py` —
replays the real engine over history (public data), buckets predicted p(up)
vs realized next-bar direction, Brier + skill vs climatology, confident-call
hit rate, and fits the best shrinkage lambda (p' = 0.5 + λ(p−0.5)) with a
plain verdict. USER RUNS OVERNIGHT: `python calibrate.py BTC/USDT --days 14
--stride 3` (hours on CPU; stride trades resolution for speed).
ALSO BUILT same session: backtest --fee-bps (default 10 = 0.1% taker per
side, both legs, conservative) threaded through simulate + walk_forward;
USER_DATA_WS (default on) — watch_orders listener settles brackets seconds
after fill, settle path serialized with an asyncio.Lock so websocket and
per-bar poll can never double-record into Kelly; polling stays
authoritative. Suite 111 -> 121 (feed 6, gatekeeper 20, calibrate 7,
backtest 7, main 21...). Walk-forward rerun pending: user runs
`python backtest.py BTC/USDT --days 90 --walk-forward` and same for
ADA/USDT (now fee-aware by default).

**M3 (original spec, kept for reference):**
- New tool `calibrate.py`: run Kronos over N days of historical bars
  (user's PC, ~overnight on CPU), score p_up/p_down vs realized outcomes,
  emit a reliability curve. Decide: probability shrinkage (pseudo-counts
  toward 0.5), dead-band widening, or (only if truly broken) Phase 10
  fine-tune. Yesterday's 28–30/30-paths-wrong calls are the motivation.

**M4 — meta promotion decision (after ~100 trades WITH shadow scores):**
- Compare shadow-filtered expectancy vs unfiltered on the same trades.
  Promote META_FILTER_MODE=veto ONLY if shadow wins. Else retrain with
  Phase B features and repeat.

**M5 — variant farm proper (blocked on user creating 2 testnet accounts):**
- Folder per variant, .env per captain's-log NEXT section. Main account
  returns to prod settings (or stays harvester until M4 resolves).

**M6 — robustness backlog (before anything resembling real money):**
- Fee/spread model in backtest + journal (taker 0.1% would eat these
  margins; testnet hides it). REQUIRED before any live decision.
- Walk-forward rerun: --days 90, plus ADA/USDT.
- User-data websocket instead of per-bar TP/SL polling (faster settles).
- Restore RISK_TOTAL_DRAWDOWN_LIMIT=0.03, MAX_OPEN_TRADES_PER_SYMBOL=1,
  Kelly sizing, enforce flags true — the harvester .env is a DATA rig,
  never a production configuration.

## 7. State of play (as of 2026-06-11)

- Bot has run overnight on testnet; first pilot night's bugs all fixed (spot
  OCO, clock drift, flatten params). Dashboard shows confluence votes,
  DI/RSI/book rows, live bracket distances, session tally, W/L + Kelly line.
- `journal.db` exists with real journaled trades (gitignored, stays local).
- Kelly sizing now learns from real outcomes and survives restarts.
- Meta filter is in shadow mode, dormant until 100 decided trades.

### RESOLVED — GitHub push (2026-06-11)
- Private repo live at https://github.com/Zentazuke/kronos-scalping-bot
  (branch `main`).
- AUTO-PUSH: a PAT lives in `.env` as `GITHUB_TOKEN` (gitignored, never
  staged, never echoed into logs or commits). Push with a one-off URL so the
  token never lands in `.git/config`:
  `git push https://x-access-token:<GITHUB_TOKEN>@github.com/Zentazuke/kronos-scalping-bot.git HEAD:main`
- Cowork-sandbox note: if the mount serves a stale/corrupt `.git` view,
  copy `.git` to /tmp, fix its config, commit there with
  `--git-dir/--work-tree`, push, then copy objects + refs back to the
  mount's `.git` (objects are immutable, refs are single-line files).
- Never stage `.env` (gitignore covers it).

### BUILT — multi-variant data farm plumbing (2026-06-11)
All four pieces implemented + tested (12 new tests, suite now 89):
- `journal.py`: `variant` column (legacy DBs auto-migrate, old rows = prod);
  VARIANT env / constructor arg; `open_trades()`, `performance()`, and
  `replay_into()` are variant-scoped (harvester's record can never shrink
  prod's Kelly); `closed_trades()` returns all variants by default for the
  meta-labeler, `variant=` kwarg to scope.
- `main.py`: REGIME_ENFORCE / CONFLUENCE_ENFORCE (default true). When false
  the gate is computed+journaled but doesn't block (harvester mode).
  `sufficient_data=False` is NEVER bypassable. Bad flag value refuses boot.
- `execution.py`: FIXED_TRADE_NOTIONAL (quote currency) replaces
  Kelly × equity when set; liquidity cap/quantize/min-size sieves still
  apply; rejects non-positive/non-finite values at boot.
- `learner.py`: `train` accepts repeated `--db` and pools journals
  (pooling across variants is CORRECT for the meta-labeler); missing DB
  files are skipped with a warning, not fatal.
- `main.py` also gained tuning knobs: REGIME_MIN_ADX, EDGE_THRESHOLD,
  DEAD_BAND_LOW, DEAD_BAND_HIGH, TP_ATR_MULT, SL_ATR_MULT, SLIPPAGE_LIMIT
  (constructor-validated).

### BUILT — concurrent positions + venue-minimum sizing (2026-06-11 late)
User wants maximum trade throughput for training data. execution.py now has:
- `MAX_OPEN_TRADES_PER_SYMBOL` (default 1 = original strict state machine +
  zero-exposure reconcile; N>1 = cap; 0 = unlimited). In concurrent mode the
  journal's variant-scoped open-trade count (passed by the supervisor on
  every route call) is the limiter; only a mid-flight PENDING_ENTRY blocks.
  After placement the state returns to IDLE instead of holding ACTIVE.
- `FIXED_TRADE_NOTIONAL=min` — sizes at the venue minimum (max of min-amount
  and min-notional floors, +10% headroom so quantization can't round below).
- OutcomeMonitor already handles N open trades per symbol natively.
- CAUTION: concurrent mode skips the zero-exposure reconcile by design.
  NEVER enable it on a real-money account; it exists for the data farm.
- Suite: 93 tests.

### BUILT — Phase A microstructure (2026-06-11 night, suite 98)
Journal-first integration of scalping microstructure (user provided the
indicator list; principle: COMPUTE + JOURNAL as meta features, never bolt on
new entry gates — only promote to gates with evidence; spread is the one
exception because it is execution COST, not signal):
- gatekeeper.RegimeReport grew: spread_bps, relative_volume (candle/avg),
  depth_imbalance and total_depth within ±0.25% of mid (`_microstructure`).
- journal: 4 new TEXT columns, auto-migrating; plumbed through main.py.
- learner FEATURE_NAMES now 9 (spread_bps, rel_volume, depth_align).
  Safe because no trained model exists yet; MetaModel.load refuses
  feature-name mismatches anyway.
- execution: MAX_SPREAD_BPS env → ABORT_SPREAD sieve (disabled when unset;
  .env runs 25 bps).
PHASE B (next) — spec updated after web research 2026-06-12:
1. CVD/trade-flow imbalance (aggressive buy − sell volume, rolling windows)
   via a watchTrades consumer in feed.py — THE industry scalping signal.
2. True OFI per Cont et al.: queue-size CHANGES between book snapshots
   (not static depth) — strongest short-horizon predictor in the academic
   literature; accumulate in feed between bars.
3. Micro-VWAP from the trade stream.
4. Microprice (depth-weighted mid: (bidSz*ask+askSz*bid)/(bidSz+askSz));
   gap vs mid is a cheap directional feature from the existing snapshot.
All journal-first (meta features), no new gates without evidence.
DEFERRED: sub-minute momentum/vol burst (bot is bar-driven),
liquidations/funding/OI (spot has none), Tier-3 sentiment (noise at 5m).

### ALSO — dashboards (2026-06-11)
- dashboard_server.py + dashboard.html: real-time localhost dashboard
  (10s polling, LAN-accessible, read-only; equity heartbeat line in
  main.py `_drawdown_check` feeds wallet cards). Decision-chain modal:
  click any trade → 6-step pipeline view from journaled context.
- Cowork artifact + Vercel `dashboard` branch exist but are PAUSED
  (hourly refresh task disabled); localhost is the maintained one.

### FIXED — two live harvester bugs (2026-06-11 ~20:30, suite 101)
1. **-1100 OCO rejection (BTC only):** `quantity`/prices were serialized as
   Python floats; below 1e-4 float repr is scientific ("9e-05") and Binance's
   regex rejects it. Venue-minimum BTC sizes (~0.00009) hit it on EVERY
   bracket; the filled entry was emergency-flattened each time. Fix:
   `f"{decimal:f}"` fixed-point strings in `_place_spot_oco`. ADA (~32.4)
   never triggered it — that asymmetry was the diagnostic clue.
2. **Sibling-bracket massacre in concurrent mode:** a 6th ADA OCO failed at
   the venue (Binance spot caps stop-type orders at 5/symbol), and
   `_emergency_flatten`'s `cancel_all_orders(symbol)` then cancelled the five
   HEALTHY sibling OCOs → monitor recorded 5× UNKNOWN, wallet left short
   ~162 ADA unprotected. Fix: concurrent mode flattens only the naked amount
   and never sweeps the symbol's orders; single mode keeps the sweep.
   `.env` now runs MAX_OPEN_TRADES_PER_SYMBOL=4 (stay under the venue cap).
   NOTE: trades #15–19 (UNKNOWN) left real unhedged short exposure —
   testnet only, user advised to check/flatten ADA balance manually.
ALSO: bot.log timestamps are LOCAL (UTC+1 for user); journal/TUI are UTC.
Cowork mount sync can lag bot.log by HOURS — diagnose live issues from
user-pasted PowerShell output, not the mounted file.

### ACTIVE — main a

---

## 8. LOG ENTRY — 2026-07-03/04, the audit sessions (CURRENT STATE OF PLAY)

Two days that changed the project's footing. Full detail in `Audit_2026-07-03.md`
and the postmortem's 2026-07-04 addendum; this is the handoff summary.

### What was found and fixed
- **grid_backtest.py had a critical bug**: every grid rebuild silently reset cash
  to initial capital — realized losses were ERASED, flattering exactly the gated
  modes under test. Also same-bar regime lookahead, and later an unbounded
  level-loop that ate 100GB of pagefile (glitch-bar ATR → negative floor). All
  fixed, all regression-tested. **Any grid result from before 2026-07-04 is void.**
- **TSM cost model undercounted by ~half** (trial charged 10bps vs the real ~20bps
  two-taker round trip; live P&L ignored fees entirely). Fixed both;
  `restate_forward_nets.py` recomputes the stored forward history to the honest
  basis. Fees now live in ONE place: `costs.py`.
- **Dashboard**: manual-close P&L ignored fees (biasing the you-vs-bot scoreboard
  pro-manual), /close was CSRF-able, search endpoints could stack subprocesses,
  journal-path closes could double-fire. All fixed.

### The 2026-07-04 verdict round (honest backtests, post-fix)
- Grid: DEAD economically (best config = no gate at all, OOS Sharpe 0.20; the
  hedge-gate idea also died — churn). One loose end: rerun gated by the ANALYST
  regime csv; if that doesn't beat no-gate, the grid file closes forever.
- Morning-fade: FAIL, graveyarded. Split-hour study: noise (no hour-hopping).
- **Analyst-regime allocator: PASSED the pre-registered bar** — trend-rider beat
  buy&hold OOS on 6/7 coins (Sharpe 0.58 vs 0.27, maxDD 34% vs 77%), default-long
  7/7. The analyst's regime signal (its one RELIABLE calibrated output) is the
  project's best result to date. Reward: a shadow forward test, nothing more.

### What is RUNNING now (all on the Ubuntu server unless noted)
- `crypto-analyst.service` — analyst API + scheduler, localhost:8000, 7-coin
  basket, recording candles/micro/funding/alerts/VRP + (new) sentiment/on-chain
  metric history + (new) options surface / derivatives crowd / stablecoin feeds.
- `trend_sleeve_forward.py` — cron 00:10 UTC, shadow-logs both allocator sleeves
  vs buy&hold from the analyst's live regime reads. First day committed 7/7.
- Intraday-TSM live testnet (unchanged cadence) + `compare_live_vs_trial.py
  --alert` reconcile cron 01:10 UTC — divergence now alerts instead of waiting
  to be noticed.
- CryptoAnalyst is on GitHub (public) with CI green on clean Linux.

### The judgment rules (do not renegotiate mid-test)
Sleeves and TSM are judged over WEEKS via their forward reports, vs buy&hold,
on the restated fee basis. No live-config changes while evidence accrues. New
metric layers stay DECORATION until `/metric-calibration` cards them RELIABLE
on accrued history (needs ~4-6 weeks of scheduler uptime).

### Next session menu
1. Read the sleeve forward report (~2-3 weeks) and the grid loose-end output.
2. Analyst P1.3: frontend evidence panel (/explain) + metric explorer — the
   remaining UI work. P4.2 Docker + P3 composite score (data-gated) after.
3. If sleeves confirm live: design the testnet executor (same one-brain pattern
   as TSM: execute the trial's committed decisions, never recompute).

# CAPTAIN'S LOG — Kronos Scalping Bot

Handoff document for continuing this project in any session/workspace.
Read this top to bottom before touching code. Last updated: 2026-06-11.

---

## 1. What this is

Institutional-grade algorithmic scalping bot for **BTC/USDT + ADA/USDT, 5-minute
timeframe, Binance Spot TESTNET** (Stage 1 pilot — no real capital). Built in
phases, each phase one module, each module with an embedded unittest suite.

Project root: `C:\Users\Ricardo\Desktop\Work\PROJECTOS\Trading Bot`
(NOT the WANDERING PIXELS folder — that is an unrelated project.)

**Current test count: 77 passing** — run with:
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

## 5. Environment (`.env` — real file is NOT in git; see `.env.example`)

```
EXCHANGE_ID=binance          USE_SANDBOX=True (mandatory)
EXCHANGE_API_KEY/SECRET      (Binance Spot Testnet keys, 64-char)
RISK_TOTAL_DRAWDOWN_LIMIT=0.03   ORDER_TIMEOUT_S=2.0 (testnet; ~0.2 live)
KRONOS_REPO_PATH=./Kronos    CONFLUENCE_MIN_VOTES=2
META_FILTER_MODE=shadow      META_MIN_PWIN=0.5
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
```

## 7. State of play (as of 2026-06-11)

- Bot has run overnight on testnet; first pilot night's bugs all fixed (spot
  OCO, clock drift, flatten params). Dashboard shows confluence votes,
  DI/RSI/book rows, live bracket distances, session tally, W/L + Kelly line.
- `journal.db` exists with real journaled trades (gitignored, stays local).
- Kelly sizing now learns from real outcomes and survives restarts.
- Meta filter is in shadow mode, dormant until 100 decided trades.

### UNRESOLVED — GitHub push (was mid-flight when session ended)
- Local repo initialized, single commit `5a2d68c` (9 modules + .gitignore +
  .env.example; **.env/journal/logs/Kronos correctly excluded**).
- `gh` CLI v2.93 installed via winget. Device-flow auth COMPLETED once
  ("Authentication complete") but a fresh shell then said "not logged in" —
  token likely stored under the login shell's credential context only.
- NEXT STEP: run `gh auth status`; if logged out, `gh auth login` again
  (same browser device flow), then:
  `gh repo create kronos-scalping-bot --private --source . --push`
- The repo MUST be private. Never stage `.env` (gitignore already covers it).

### AGREED NEXT BUILD — multi-variant data farm (user approved)
Three parallel bots on **separate testnet accounts + separate folders**
(shared wallet is a hard blocker: reconcile blocks on any exposure; one bot's
kill switch would flatten the sibling's brackets; drawdown breaker reads
shared equity):
- **A: prod** — exactly as-is.
- **B: relaxed** — ADX>20, votes=1, edge 0.50. Same tiny sizing.
- **C: free-for-all harvester** — gates computed+journaled but NOT enforced,
  every non-dead-band signal, tiny fixed notional. Purpose: kill the
  journal's selection bias (vetoed setups are censored data today).

Plumbing required: `variant` column + VARIANT env in journal.py;
REGIME_ENFORCE / CONFLUENCE_ENFORCE flags in main.py; FIXED_TRADE_NOTIONAL
sizing mode in execution.py; `learner.py train` accepting multiple `--db`
(pooling across variants is CORRECT for the meta-labeler);
`replay_into()` filtered to OWN variant (harvester's bad win rate must never
shrink prod sizing). Watch CPU: harvester runs Kronos every bar.
User must create 2 extra Binance testnet accounts/keys first.

### Backlog (discussed, not yet committed to)
- Promote meta filter shadow→veto only after its shadow record beats the
  unfiltered baseline over ~100 trades.
- Walk-forward says whether TP1.5/SL2.5 (breakeven 62.5% win rate) should flip.
- Phase 10 (Kronos fine-tune): deferred until evidence shows the model is the
  bottleneck. Offline, versioned, validated only.
- Possible: spread/fee model; user-data websocket instead of per-bar polling.

## 8. Working style that produced this codebase (keep it)

- One phase = one module = complete production code + embedded tests in the
  same file; no stubs, no placeholders, run tests after every phase.
- Decimal for decisions, float for models; conversions only at boundaries.
- Fail-safe defaults everywhere: unverified state → refuse to trade;
  inference fault → NEUTRAL; missing evidence → failed confluence vote.
- Offline tools (backtest/learner) never share runtime state with the live
  bot and never need API keys.
- The dashboard is an observer — it can never block or influence a decision
  (drop-oldest queue, publish() never raises).

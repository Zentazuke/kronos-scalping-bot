# Sentiment ↔ Bot Integration Plan

*Goal: get the engine's novel alt-data signals (news sentiment, Fear & Greed, crowd positioning, funding) recorded **per setup** in the bot's observation journal, so the search engine can test them as ingredients — without the bot ever depending on the engine.*

---

## Where we are

- **Bot side:** already has `sentiment_shadow_client.py` (fail-safe, stdlib) and calls the engine **fire-and-forget** in `main.py` (~line 1026). So today the *engine* journals an opinion, but the *bot never captures it* — the signals don't reach `observations.db`, so the search can't see them. **That's the gap.**
- **Engine side (built in the other conversation):** exposes `/snapshot` (sentiment), `/context` (Fear & Greed), `/positioning` (crowd L/S), `/derivatives` (funding, OI), `/outlook` (composite). Will run on the **server**, light mode, at `127.0.0.1:8787`.

The fix is to **capture** the signals at decision time and write them as observation feature columns. Then they flow into `strategy_search.py` like any other ingredient and the honest search judges them.

---

## The contract (the one thing both sides build to)

A single endpoint on the engine so the bot makes **one** fast local call per setup:

`GET /signals/{symbol_key}` → flat JSON, every field optional (`null` = unavailable, **never a fake zero**):

```json
{
  "symbol": "BTC/USDT",
  "ts": 1718900000.0,
  "sentiment_score":    0.12,   // news/text sentiment, ~[-1, 1]
  "sentiment_velocity": 0.03,
  "attention_spike":    1.4,
  "fear_greed":         38,     // 0..100, market-wide
  "long_short_ratio":   1.8,    // crowd positioning
  "funding_rate":       0.00012,
  "open_interest":      1.2e9,
  "outlook_1h":         -0.2    // composite short-horizon score
}
```

Why a combined endpoint: one call instead of four, a stable interface, and the bot stays ignorant of how the engine computes anything. *(Fallback if `/signals` isn't ready: the bot can call `/snapshot` + `/context` + `/positioning` + `/derivatives` and assemble it — but the single endpoint is the clean path.)*

---

## Bot-side changes (build here)

1. **`SentimentShadowClient.signals(symbol) -> dict`** — `GET /signals`, fail-safe: returns all-`None` on any timeout/error, never raises, never blocks beyond a short timeout (~0.8s). Mirrors the existing `evaluate()` safety.
2. **`ObservationJournal`** — add optional columns: `sent_score, sent_velocity, attention_spike, fear_greed, long_short_ratio, funding_rate, open_interest, outlook_1h`. Additive — existing rows just carry `NULL`.
3. **`main.py` observation hook** — at the record point (right where we already record the setup), call `client.signals(symbol)` and pass the values into `record(...)` as new kwargs, wrapped fail-safe so it can never disturb a trading bar.
4. **`strategy_search.py`** — add the new columns to `load_setups` so they become searchable ingredients (mostly direction-agnostic; `sent_score`/`outlook_1h` can be made direction-aware like `conviction`).

All four are **fail-safe and dormant until the engine answers** — we can build them now, and the moment the engine is live they start filling in. Until then, observations simply carry `NULL` sentiment columns (harmless).

---

## Engine-side requirements (other conversation)

- Run on the **server**, light mode (lexicon, **no** Bluesky firehose, **no** CryptoBERT for now), reachable at `127.0.0.1:8787`.
- Expose `GET /signals/{symbol_key}` per the contract above.
- Cover the bot's symbols via `SENTIMENT_SYMBOLS` (the 8: BTC ADA ETH BNB SOL XRP DOGE LINK) — or accept that uncovered symbols return `null` signals (the search handles per-symbol missing features fine).
- Keep the DB **off the synced folder** (server local disk) so the corruption issue stays gone.

---

## Coexistence & safety (already validated)

- **Footprint:** measured 2.6 GB available on the box (bot only 0.59 GB). Light sentiment (~0.4 GB) fits with ~2 GB to spare. Add a swap file before any heavy text model.
- **Fail-safe:** engine down/slow → `signals()` returns `None`s → the bot trades exactly as now; only the sentiment columns are blank for those bars.
- **CPU:** drop Bluesky; the engine's polls are light and won't fight the bot's bursty Kronos inference.

---

## Sequence

1. **Agree the contract** (this doc) — so both sides meet at `/signals`.
2. **Engine:** add `/signals`, deploy on the server, light mode (other conversation).
3. **Bot:** build the four changes above (can start now; dormant until the engine answers).
4. **Accumulate** ~2 weeks of *aligned* data (each setup tagged with its live sentiment).
5. **Run `strategy_search.py`** — sentiment ingredients are now in the pool, and the noise-floor + holdout judge them like everything else. We find out, honestly, whether sentiment adds an edge.

*The discipline holds: we don't trust the engine's prediction — we feed its raw signals to the search and let the machine that refuses to lie tell us if they're worth anything.*

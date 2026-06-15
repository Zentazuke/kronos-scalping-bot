# Add this route to the Sentiment_Engine repo's sentiment_engine/api.py
# (next to the other @app.get routes). It bundles the existing endpoints into
# the flat shape the trading bot's GET /signals/{symbol} expects.
#
# Field map (engine -> contract):
#   snapshot.average_sentiment   -> sentiment_score
#   snapshot.sentiment_velocity  -> sentiment_velocity
#   snapshot.attention_spike     -> attention_spike
#   context.fear_greed_value     -> fear_greed
#   positioning.global_account_ratio -> long_short_ratio
#   derivatives.funding_rate     -> funding_rate
#   derivatives.open_interest_usd-> open_interest
#   outlook.horizons["1h"].score -> outlook_1h
#
# Fail-safe: every source is wrapped; a missing/erroring source yields null.
# The bot only reads numeric fields and treats null as "unavailable".

@app.get("/signals/{symbol_key}")
def signals(symbol_key: str) -> Dict[str, Any]:
    """Flat bag of headline alt-data signals for one symbol, for downstream
    consumers (the trading bot's observation journal). Every field optional;
    null = unavailable. Best-effort per source — never raises."""
    import time as _t
    out: Dict[str, Any] = {
        "symbol": symbol_key.replace("-", "/"),
        "ts": _t.time(),
        "sentiment_score": None,
        "sentiment_velocity": None,
        "attention_spike": None,
        "fear_greed": None,
        "long_short_ratio": None,
        "funding_rate": None,
        "open_interest": None,
        "outlook_1h": None,
    }
    try:
        s = snapshot(symbol_key)
        out["sentiment_score"] = s.get("average_sentiment")
        out["sentiment_velocity"] = s.get("sentiment_velocity")
        out["attention_spike"] = s.get("attention_spike")
    except Exception:
        pass
    try:
        out["fear_greed"] = ((context() or {}).get("context") or {}).get("fear_greed_value")
    except Exception:
        pass
    try:
        pos = (positioning(symbol_key) or {}).get("positioning") or {}
        out["long_short_ratio"] = pos.get("global_account_ratio")
    except Exception:
        pass
    try:
        d = (derivatives(symbol_key) or {}).get("derivatives") or {}
        out["funding_rate"] = d.get("funding_rate")
        out["open_interest"] = d.get("open_interest_usd")
    except Exception:
        pass
    try:
        hz = (outlook(symbol_key) or {}).get("horizons") or {}
        h1 = hz.get("1h")
        out["outlook_1h"] = h1.get("score") if isinstance(h1, dict) else h1
    except Exception:
        pass
    return out

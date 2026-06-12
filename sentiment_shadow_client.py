"""Shadow-mode client for the sentiment engine.

This file is deliberately self-contained (stdlib only, no imports from the
sentiment_engine package) so it can be COPIED into your trading bot project
as-is. The bot must never gain a hard dependency on the engine.

Guarantees, in order of importance:
1. NEVER raises. Any failure returns a safe neutral result.
2. NEVER blocks longer than `timeout` seconds (default 1.5s).
3. NEVER trades, sizes positions, or generates signals. It only asks the
   engine for an opinion about a signal the bot already produced.

Shadow-mode usage (bot ignores the answer, engine journals everything):

    from sentiment_shadow_client import SentimentShadowClient

    client = SentimentShadowClient()  # http://127.0.0.1:8787

    # ... inside your signal handler, right after a LONG/SHORT signal fires:
    client.evaluate_async(symbol="BTC/USDT", direction="STRAT_LONG",
                          bot_confidence=0.7, trigger_price=63376.0)
    # then proceed EXACTLY as before; do not read the result in shadow mode.

Later, veto-mode (ONLY after shadow data proves value) is the same call but
blocking and reading the action:

    result = client.evaluate(symbol="BTC/USDT", direction="STRAT_LONG",
                             bot_confidence=0.7, trigger_price=63376.0)
    if result["action"] == "veto" and result["safe_to_use"]:
        skip_trade()   # your decision, not the engine's
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional

__all__ = ["SentimentShadowClient"]

logger = logging.getLogger("sentiment_shadow_client")

_VALID_DIRECTIONS = ("STRAT_LONG", "STRAT_SHORT")


def _neutral_result(symbol: str, bot_confidence: float, reason: str) -> Dict[str, Any]:
    """The safe answer used whenever the engine cannot be consulted."""
    return {
        "symbol": symbol,
        "action": "neutral",
        "sentiment_score": 0.0,
        "confidence_modifier": 1.0,
        "final_confidence_estimate": bot_confidence,
        "data_quality": 0.0,
        "reason": f"Neutral (client fallback): {reason}",
        "safe_to_use": False,
        "snapshot": None,
        "microstructure": None,
    }


class SentimentShadowClient:
    """Fail-safe HTTP client for the standalone sentiment engine."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8787",
        timeout: float = 1.5,
        max_workers: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sentiment-shadow")
        self._lock = threading.Lock()
        self.stats: Dict[str, int] = {"sent": 0, "ok": 0, "failed": 0}

    # --- public API --------------------------------------------------------

    def evaluate(
        self,
        *,
        symbol: str,
        direction: str,
        bot_confidence: float,
        trigger_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Blocking evaluation. Returns a dict; on ANY failure returns neutral.

        Use only when you intend to read the result (veto mode). For shadow
        mode prefer evaluate_async().
        """
        if direction not in _VALID_DIRECTIONS:
            return _neutral_result(symbol, bot_confidence, f"invalid direction {direction!r}")
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "direction": direction,
            "bot_confidence": max(0.0, min(1.0, float(bot_confidence))),
            "timestamp": time.time(),
        }
        if trigger_price is not None and trigger_price > 0:
            payload["trigger_price"] = float(trigger_price)
        try:
            request = urllib.request.Request(
                f"{self.base_url}/evaluate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            with self._lock:
                self.stats["sent"] += 1
                self.stats["ok"] += 1
            return result
        except Exception as exc:  # noqa: BLE001 - the bot must never break
            with self._lock:
                self.stats["sent"] += 1
                self.stats["failed"] += 1
            logger.warning("sentiment engine unreachable (%s); neutral fallback", type(exc).__name__)
            return _neutral_result(symbol, bot_confidence, f"engine unreachable ({type(exc).__name__})")

    def evaluate_async(
        self,
        *,
        symbol: str,
        direction: str,
        bot_confidence: float,
        trigger_price: Optional[float] = None,
    ) -> Future:
        """Fire-and-forget evaluation for shadow mode.

        Returns immediately (sub-millisecond). The engine journals the
        evaluation and its price outcomes; the bot does not wait and should
        not act on the result.
        """
        try:
            return self._executor.submit(
                self.evaluate,
                symbol=symbol,
                direction=direction,
                bot_confidence=bot_confidence,
                trigger_price=trigger_price,
            )
        except Exception as exc:  # noqa: BLE001 - e.g. executor shut down
            logger.warning("shadow submit failed (%s)", type(exc).__name__)
            future: Future = Future()
            future.set_result(_neutral_result(symbol, bot_confidence, "submit failed"))
            return future

    def health(self) -> Optional[Dict[str, Any]]:
        """Engine health, or None if unreachable. Never raises."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        self._executor.shutdown(wait=False)

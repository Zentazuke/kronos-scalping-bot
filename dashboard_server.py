"""dashboard_server.py — real-time local dashboard for the Kronos bot.

Run alongside the bot from the project folder:

    python dashboard_server.py            # default port 8765
    python dashboard_server.py 9000       # custom port

Then open http://localhost:8765 on this PC, or http://<this-PC's-LAN-IP>:8765
from a phone on the same Wi-Fi (the startup banner prints both URLs).

Read-only: opens journal.db in SQLite read-only mode and tails bot.log.
It can never write, lock, or otherwise influence the trading runtime.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Final, List
from urllib.parse import parse_qs, urlparse

BASE_DIR: Final[Path] = Path(__file__).resolve().parent

# Load .env (USE_SANDBOX, EXCHANGE_API_KEY/SECRET, ...) so the manual-close feature
# works no matter how the dashboard is launched — not just when the shell sourced it.
try:  # optional; the read-only dashboard runs fine without it
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except Exception:  # noqa: BLE001
    pass

DB_PATH: Final[Path] = BASE_DIR / "journal.db"
LOG_PATH: Final[Path] = BASE_DIR / "bot.log"
HTML_PATH: Final[Path] = BASE_DIR / "dashboard.html"
TSM_DB_PATH: Final[Path] = BASE_DIR / "tsm_forward.db"
LOG_TAIL_BYTES: Final[int] = 600_000

try:  # optional TA-signals engine; the dashboard must run even if it's absent
    from ta_signals import compute_signals as _compute_signals
except Exception:  # noqa: BLE001
    _compute_signals = None  # type: ignore[assignment]

try:  # optional auto-forward-test registry; dashboard runs fine without it
    import forward_rules as _forward_rules_mod
except Exception:  # noqa: BLE001
    _forward_rules_mod = None  # type: ignore[assignment]

_LINE = re.compile(r"^([\d-]+ [\d:]+),\d+ (\w+)\s+([\w.]+) — (.*)$")
_KEEP = re.compile(
    r"bracket live|closed (WIN|LOSS|SCRATCH|UNKNOWN)|journaled|ABORT|BLOCKED"
    r"|veto|slippage abort|emergency|flatten|naked position|supervisor boot"
    r"|kill.?switch|drawdown|BOOT REFUSED|REGIME_ENFORCE|CONFLUENCE_ENFORCE"
    r"|FIXED_TRADE_NOTIONAL|MAX_OPEN_TRADES|bracket placement failed"
    r"|insufficient",
    re.I,
)


def _trades() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=2.0)
    try:
        conn.row_factory = sqlite3.Row
        return [
            dict(r)
            # SELECT * so newly migrated feature columns (Phase A/B and
            # beyond) reach the dashboard without further server changes.
            for r in conn.execute("SELECT * FROM trades ORDER BY id")
        ]
    finally:
        conn.close()


def _events() -> List[Dict[str, str]]:
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open("rb") as fh:
        fh.seek(max(0, LOG_PATH.stat().st_size - LOG_TAIL_BYTES))
        text = fh.read().decode("utf-8", errors="replace")
    events: List[Dict[str, str]] = []
    for line in text.splitlines():
        match = _LINE.match(line)
        if match and _KEEP.search(match.group(4)):
            events.append(
                {"ts": match.group(1), "level": match.group(2),
                 "msg": match.group(4)[:240]}
            )
    return events[-150:]


_SYM_MSG = re.compile(r"([A-Z]+/[A-Z]+): (.*)")
_DECISION_PATTERNS = (
    ("regime_reject", re.compile(r"regime rejected — no inference this bar \((.*)\)")),
    ("regime_bypass", re.compile(r"REGIME_ENFORCE=false")),
    ("signal", re.compile(r"STRAT_\w+ \| paths")),
    ("neutral", re.compile(r"STRAT_NEUTRAL — standing down")),
    ("conf_veto", re.compile(r"confluence veto on")),
    ("conf_bypass", re.compile(r"CONFLUENCE_ENFORCE=false")),
    ("meta", re.compile(r"META (SHADOW|VETO)")),
    ("outcome", re.compile(r"routing outcome ")),
    ("bracket", re.compile(r"bracket live — ")),
)


def _decisions(text: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Last decision-pipeline line of each kind, per symbol (log tail)."""
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for line in text.splitlines()[-600:]:
        match = _LINE.match(line)
        if not match:
            continue
        ts, _lvl, _mod, msg = match.group(1), match.group(2), match.group(3), match.group(4)
        sym_match = _SYM_MSG.match(msg)
        if not sym_match:
            continue
        sym, rest = sym_match.group(1), sym_match.group(2)
        for key, pattern in _DECISION_PATTERNS:
            if pattern.search(rest):
                out.setdefault(sym, {})[key] = {"ts": ts, "text": rest[:220]}
    return out


_OFI_GATE = re.compile(
    r"([A-Z0-9]+/[A-Z0-9]+): OFI gate — aligned (\S+) [<>]=? [\d.]+, "
    r"(ROUTED|live trade SKIPPED).*?\[routed=(\d+) gated=(\d+)\]"
)
_OFI_TS = re.compile(r"^([\d-]+ [\d:]+)")


def _ofi_gate(text: str) -> Dict[str, object]:
    """Recent OFI-gate decisions + running routed/gated tally from the log tail
    (the live forward-test view). Empty dict when the gate isn't active."""
    recent: List[Dict[str, object]] = []
    routed = gated = 0
    for line in text.splitlines()[-2000:]:
        m = _OFI_GATE.search(line)
        if not m:
            continue
        sym, aligned, decision, r, g = m.groups()
        routed, gated = int(r), int(g)
        tsm = _OFI_TS.match(line)
        recent.append({
            "ts": tsm.group(1) if tsm else "",
            "symbol": sym, "aligned": aligned, "routed": decision == "ROUTED",
        })
    if not recent:
        return {}
    return {"active": True, "routed": routed, "gated": gated, "recent": recent[-15:][::-1]}


def _tsm_forward() -> Dict[str, Any]:
    """Read-only summary of the intraday-TSM shadow forward test (tsm_forward.db):
    headline net/trade + win%, per-coin breakdown, a cumulative-net curve, and the
    pending (committed, not-yet-matured) trades. Empty when the DB isn't there yet."""
    if not TSM_DB_PATH.exists():
        return {"present": False}
    try:
        conn = sqlite3.connect(f"file:{TSM_DB_PATH}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT decision_day, symbol, direction, status, morning_ret, net_ret "
            "FROM forward_trades ORDER BY decision_day, symbol")]
        conn.close()
    except sqlite3.Error as exc:
        return {"present": True, "error": str(exc)}

    settled = [r for r in rows if r["status"] == "SETTLED"
               and r["direction"] != "FLAT" and r["net_ret"] is not None]
    pending = [r for r in rows if r["status"] == "PENDING"]
    flat = [r for r in rows if r["direction"] == "FLAT"]
    days = sorted({r["decision_day"] for r in rows})

    nets = [float(r["net_ret"]) for r in settled]
    n = len(nets)
    total = sum(nets)
    win = (sum(1 for x in nets if x > 0) / n) if n else 0.0
    exp = (total / n) if n else 0.0

    by_coin_map: Dict[str, List[float]] = {}
    for r in settled:
        by_coin_map.setdefault(r["symbol"].split("/")[0], []).append(float(r["net_ret"]))
    by_coin = sorted(
        ({"coin": c, "n": len(a), "win": sum(1 for x in a if x > 0) / len(a),
          "exp": sum(a) / len(a), "total": sum(a)} for c, a in by_coin_map.items()),
        key=lambda d: -d["total"])

    cum = 0.0
    curve: List[Dict[str, Any]] = []
    for r in sorted(settled, key=lambda r: (r["decision_day"], r["symbol"])):
        cum += float(r["net_ret"])
        curve.append({"day": r["decision_day"], "coin": r["symbol"].split("/")[0],
                      "dir": r["direction"], "net": float(r["net_ret"]), "cum": cum})

    pend = [{"day": r["decision_day"], "coin": r["symbol"].split("/")[0],
             "dir": r["direction"], "morning": r["morning_ret"]} for r in pending]
    pend.sort(key=lambda d: (d["day"], d["coin"]))

    return {"present": True, "error": "",
            "span": [days[0], days[-1]] if days else [],
            "n_days": len(days), "n_settled": n, "n_pending": len(pending), "n_flat": len(flat),
            "win": win, "exp": exp, "total": total,
            "by_coin": by_coin, "curve": curve, "pending": pend}


def _fwd_rules() -> Dict[str, Any]:
    """Auto-enrolled green search rules + their live forward results (read-only)."""
    if _forward_rules_mod is None:
        return {"present": False, "rules": []}
    try:
        return _forward_rules_mod.summary(str(_OBS_DB_PATH), str(BASE_DIR / "forward_rules.db"))
    except Exception as exc:  # noqa: BLE001
        return {"present": True, "error": str(exc), "rules": []}


# --------------------------------------------------------------------------- #
# Manual position control (testnet ONLY) — live P&L + close-only flatten.
# This is the one place the dashboard is allowed to act on the exchange. It is
# hard-guarded to sandbox mode and can only REDUCE/close, never open.
# --------------------------------------------------------------------------- #
_TRADE_EX: Dict[str, Any] = {"ex": None, "tried": False, "err": ""}
_POS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}


def _trade_exchange():
    """Lazily build a sandbox ccxt client for reading positions + closing. Returns
    None (with a reason) unless USE_SANDBOX=true and API keys are present."""
    if _TRADE_EX["tried"]:
        return _TRADE_EX["ex"]
    _TRADE_EX["tried"] = True
    if os.getenv("USE_SANDBOX", "").strip().lower() != "true":
        _TRADE_EX["err"] = "USE_SANDBOX is not 'true' — manual close disabled"; return None
    key, sec = os.getenv("EXCHANGE_API_KEY"), os.getenv("EXCHANGE_API_SECRET")
    if not (key and sec):
        _TRADE_EX["err"] = "no API keys in env — manual close disabled"; return None
    try:
        import ccxt  # type: ignore[import-untyped]
        klass = getattr(ccxt, os.getenv("EXCHANGE_ID", "binance"))
        ex = klass({"enableRateLimit": True, "apiKey": key, "secret": sec,
                    "options": {"defaultType": "future", "adjustForTimeDifference": True}})
        sm = getattr(ex, "set_sandbox_mode", None)
        if callable(sm):
            sm(True)
        if not (getattr(ex, "isSandboxModeEnabled", False) or getattr(ex, "sandboxMode", False)):
            _TRADE_EX["err"] = "sandbox mode could not be verified — refusing"; return None
        _TRADE_EX["ex"] = ex
        return ex
    except Exception as exc:  # noqa: BLE001
        _TRADE_EX["err"] = f"client init failed: {str(exc)[:80]}"; return None


def _positions(ttl: float = 5.0) -> Dict[str, Any]:
    """Open positions with live unrealized P&L. Cached ttl seconds (the dashboard
    polls often; fetch_positions is rate-limited)."""
    now = time.time()
    if _POS_CACHE["data"] is not None and now - _POS_CACHE["ts"] < ttl:
        return _POS_CACHE["data"]
    ex = _trade_exchange()
    if ex is None:
        out = {"ok": False, "err": _TRADE_EX["err"], "list": []}
        _POS_CACHE.update(ts=now, data=out); return out
    try:
        poss = ex.fetch_positions()
    except Exception as exc:  # noqa: BLE001
        out = {"ok": False, "err": str(exc)[:100], "list": []}
        _POS_CACHE.update(ts=now, data=out); return out
    rows: List[Dict[str, Any]] = []
    for p in poss:
        try:
            size = abs(float(p.get("contracts") or 0))
        except (TypeError, ValueError):
            size = 0.0
        if not size:
            continue
        entry = float(p.get("entryPrice") or 0) or 0.0
        mark = float(p.get("markPrice") or 0) or 0.0
        side = str(p.get("side") or "").lower()
        try:
            upnl = float(p.get("unrealizedPnl"))
        except (TypeError, ValueError):
            upnl = None
        pct = ((mark / entry - 1) * 100 * (1 if side == "long" else -1)) if entry and mark else None
        rows.append({"symbol": p.get("symbol"), "side": side, "size": size,
                     "entry": entry, "mark": mark, "upnl": upnl, "pct": pct,
                     "notional": (mark * size) if mark else None})
    out = {"ok": True, "err": "", "list": rows}
    _POS_CACHE.update(ts=now, data=out)
    return out


def _close_position(symbol: str) -> Dict[str, Any]:
    """Cancel a symbol's resting orders, then market-close its position (reduceOnly).
    Mirrors the bot's emergency_flatten. Sandbox-only, close-only."""
    ex = _trade_exchange()
    if ex is None:
        return {"ok": False, "err": _TRADE_EX["err"] or "trading not configured"}
    try:
        try:
            ca = getattr(ex, "cancel_all_orders", None)
            if callable(ca):
                ca(symbol)
        except Exception:  # noqa: BLE001 — nothing to cancel is fine
            pass
        closed = []
        for p in ex.fetch_positions([symbol]):
            try:
                size = abs(float(p.get("contracts") or 0))
            except (TypeError, ValueError):
                size = 0.0
            if not size:
                continue
            side = str(p.get("side") or "long").lower()
            exit_side = "sell" if side == "long" else "buy"
            try:
                spot = bool((ex.market(symbol) or {}).get("spot"))
            except Exception:  # noqa: BLE001
                spot = False
            params = {} if spot else {"reduceOnly": True}
            ex.create_order(symbol, "market", exit_side, size, None, params)
            closed.append({"side": exit_side, "size": size})
        _POS_CACHE["ts"] = 0.0  # force refresh next poll
        if not closed:
            return {"ok": True, "msg": f"no open position on {symbol}"}
        return {"ok": True, "msg": f"closed {symbol}", "closed": closed}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "err": str(exc)[:140]}


_EQUITY = re.compile(
    r"^([\d-]+ [\d:]+),\d+ .*EQUITY ([\d.]+) baseline ([\d.]+)", re.M
)


def _equity() -> Dict[str, str] | None:
    """Latest equity heartbeat from bot.log (None until the bot logs one)."""
    if not LOG_PATH.exists():
        return None
    with LOG_PATH.open("rb") as fh:
        fh.seek(max(0, LOG_PATH.stat().st_size - LOG_TAIL_BYTES))
        text = fh.read().decode("utf-8", errors="replace")
    matches = _EQUITY.findall(text)
    if not matches:
        return None
    ts, equity, baseline = matches[-1]
    return {"ts": ts, "equity": equity, "baseline": baseline}


_SYMBOLS: Final[List[str]] = [
    s.strip()
    for s in os.getenv(
        "SYMBOLS",
        "BTC/USDT,ADA/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,LINK/USDT",
    ).split(",")
    if s.strip()
]
_KLINES_HOST: Final[str] = os.getenv("KLINES_HOST", "https://testnet.binance.vision")
_CANDLE_TTL: Final[int] = 60  # seconds between background candle refreshes
_candle_cache: Dict[str, List[List[float]]] = {}  # 5m, "BTC/USDT" -> [[t,o,h,l,c,v], ...] (price chart + 5m TA)
# Extra timeframes for the TA Signals board only (the price chart stays on 5m).
_TA_EXTRA_TFS: Final[Tuple[str, ...]] = ("15m", "1h")
_ta_extra: Dict[str, Dict[str, List[List[float]]]] = {tf: {} for tf in _TA_EXTRA_TFS}

# Sentiment engine — read its /signals per symbol so the dashboard can show
# the live feed and whether the engine is reachable. Same box as the bot.
_SENTIMENT_URL: Final[str] = os.getenv("SENTIMENT_ENGINE_URL", "http://127.0.0.1:8787").rstrip("/")
_sentiment_cache: Dict[str, Dict[str, Any]] = {}
_sentiment_state: Dict[str, Any] = {"ok": False, "ts": ""}


def _fetch_sentiment(symbol: str) -> None:
    """Pull one symbol's current signals from the engine into the cache. On any
    failure the previous values are kept and the engine is flagged unreachable."""
    key: str = symbol.replace("/", "-")
    try:
        with urllib.request.urlopen(f"{_SENTIMENT_URL}/signals/{key}", timeout=2) as resp:  # noqa: S310
            data: Any = json.loads(resp.read().decode("utf-8"))
        _sentiment_cache[symbol] = data
        _sentiment_state["ok"] = True
        _sentiment_state["ts"] = time.strftime("%H:%M:%S")
    except Exception:  # noqa: BLE001 — engine is optional; keep stale, mark not-ok
        _sentiment_state["ok"] = False


def _fetch_klines(symbol: str, interval: str, limit: int = 288) -> List[List[float]]:
    """One REST pull of OHLCV bars. Volume (k[5]) is carried for OBV."""
    pair: str = symbol.replace("/", "")
    url: str = f"{_KLINES_HOST}/api/v3/klines?symbol={pair}&interval={interval}&limit={limit}"
    with urllib.request.urlopen(url, timeout=4) as resp:  # noqa: S310 — fixed host
        raw: Any = json.loads(resp.read().decode("utf-8"))
    return [
        [int(k[0]) // 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
        for k in raw
    ]


def _fetch_candles(symbol: str) -> None:
    """Refresh one symbol's 5m bars (price chart + 5m TA) and the extra TA
    timeframes. On any failure the previous series is kept — the dashboard must
    never break on network."""
    try:
        _candle_cache[symbol] = _fetch_klines(symbol, "5m", 288)
    except Exception:  # noqa: BLE001 — keep stale series, never raise
        pass
    for tf in _TA_EXTRA_TFS:
        try:
            _ta_extra[tf][symbol] = _fetch_klines(symbol, tf, 288)
        except Exception:  # noqa: BLE001
            pass


def _candle_loop() -> None:
    """Background daemon: refresh all symbols every TTL so the request handler
    never blocks on the network."""
    while True:
        for sym in _SYMBOLS:
            _fetch_candles(sym)
            _fetch_sentiment(sym)
        time.sleep(_CANDLE_TTL)


_WALKFORWARD_PATH: Final[Path] = Path(os.getenv("WALKFORWARD_JSON", "walkforward.json"))
_OBS_DB_PATH: Final[Path] = Path(os.getenv("OBSERVATIONS_DB", "observations.db"))


def _walkforward() -> Dict[str, Any]:
    """Latest walk-forward verdicts written by `learner.py walkforward --json`.
    Empty dict if the file isn't there yet — the panel just shows 'no runs'."""
    if not _WALKFORWARD_PATH.exists():
        return {}
    try:
        return json.loads(_WALKFORWARD_PATH.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def _obs_stats() -> Dict[str, int]:
    """Observation-journal counts: how much learning data has accrued and how
    much of it the offline labeler has resolved. Read-only, never raises up."""
    if not _OBS_DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(f"file:{_OBS_DB_PATH}?mode=ro", uri=True)
        try:
            total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            labeled = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status IN ('WIN','LOSS','SCRATCH')"
            ).fetchone()[0]
            wins = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status='WIN'"
            ).fetchone()[0]
        finally:
            conn.close()
        return {"total": int(total), "labeled": int(labeled), "wins": int(wins)}
    except sqlite3.Error:
        return {}


def _ta_signals() -> Dict[str, Any]:
    """Per-symbol technical-analysis signal board, computed per timeframe from
    the live candle caches. Read-only situational display — never wired into
    trading. Shape: {tf: {symbol: result}}. Returns {} if the engine is absent."""
    if _compute_signals is None:
        return {}
    out: Dict[str, Any] = {"5m": {}, "15m": {}, "1h": {}}

    def _fill(tf: str, cache: Dict[str, List[List[float]]]) -> None:
        for sym, candles in cache.items():
            try:
                res = _compute_signals(candles, tf)
            except Exception:  # noqa: BLE001 — one bad symbol must not break the board
                res = None
            if res is not None:
                out[tf][sym] = res

    _fill("5m", _candle_cache)
    for tf in _TA_EXTRA_TFS:
        _fill(tf, _ta_extra[tf])
    return out


_SEARCH_DBS: Final[Dict[str, str]] = {"observations": "observations.db", "journal": "journal.db"}


def _run_search(path: str) -> bytes:
    """Run strategy_search.py on demand and return the latest run record as JSON
    (the dashboard's Run-Simulation button). Constrained for safety: only the two
    known journals, 1-3 conditions, no user-controlled paths. Never raises."""
    q = parse_qs(urlparse(path).query)
    db_file = _SEARCH_DBS.get((q.get("db", ["observations"])[0]).lower())
    try:
        cond = max(1, min(3, int(q.get("cond", ["2"])[0])))
    except ValueError:
        cond = 2
    if db_file is None:
        return json.dumps({"error": "unknown dataset"}).encode("utf-8")
    if not (BASE_DIR / db_file).exists():
        return json.dumps({"error": f"{db_file} not found on the server yet"}).encode("utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(BASE_DIR / "strategy_search.py"),
             "--db", db_file, "--max-conditions", str(cond), "--top", "12"],
            cwd=str(BASE_DIR), capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "search timed out — try fewer conditions"}).encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"could not start search: {exc}"}).encode("utf-8")
    if proc.returncode != 0:
        detail = ((proc.stdout or "") + (proc.stderr or "")).strip()[-300:]
        return json.dumps(
            {"error": "not enough clean data yet, or no rule qualified", "detail": detail}
        ).encode("utf-8")
    hist = BASE_DIR / "search_history.jsonl"
    try:
        last = hist.read_text("utf-8").strip().split("\n")[-1]
        record = json.loads(last)
        # Auto-enroll a green rule into the forward-test registry — so the user
        # never has to vet a green by hand again; the rule starts accruing
        # out-of-sample evidence the moment it appears.
        if _forward_rules_mod is not None:
            try:
                enrolled = _forward_rules_mod.auto_from_history(
                    str(hist), str(BASE_DIR / "forward_rules.db"))
                if enrolled:
                    record["auto_enrolled"] = enrolled
            except Exception:  # noqa: BLE001 — never let enrollment break the search view
                pass
        return json.dumps(record, default=str).encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"ran, but could not read result: {exc}"}).encode("utf-8")


def _run_geometry(path: str) -> bytes:
    """Run geometry_search.py on demand and return its latest run record as JSON
    (the dashboard's Sweep-TP/SL button). Re-replays every observation against
    mainnet candles at a grid of bracket geometries, so it fetches candles and is
    heavier than the entry-filter search. observations.db only. Never raises."""
    q = parse_qs(urlparse(path).query)
    try:
        window = max(12, min(96, int(q.get("window", ["48"])[0])))
    except ValueError:
        window = 48
    db_file = "observations.db"
    if not (BASE_DIR / db_file).exists():
        return json.dumps({"error": f"{db_file} not found on the server yet"}).encode("utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(BASE_DIR / "geometry_search.py"),
             "--db", db_file, "--window", str(window), "--json"],
            cwd=str(BASE_DIR), capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "sweep timed out — too many observations / candles"}).encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"could not start sweep: {exc}"}).encode("utf-8")
    if proc.returncode != 0:
        detail = ((proc.stdout or "") + (proc.stderr or "")).strip()[-300:]
        return json.dumps(
            {"error": "not enough resolved observations yet to sweep geometry", "detail": detail}
        ).encode("utf-8")
    # geometry_search prints the run record as JSON on stdout (--json); fall back
    # to the persisted history if stdout was empty for any reason.
    out = (proc.stdout or "").strip()
    if out:
        try:
            return json.dumps(json.loads(out.split("\n")[-1]), default=str).encode("utf-8")
        except Exception:  # noqa: BLE001
            pass
    hist = BASE_DIR / "geometry_history.jsonl"
    try:
        last = hist.read_text("utf-8").strip().split("\n")[-1]
        return json.dumps(json.loads(last), default=str).encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"ran, but could not read result: {exc}"}).encode("utf-8")


def _payload() -> bytes:
    out: Dict[str, Any] = {
        "source": "db",
        "trades": [],
        "events": [],
        "equity": None,
        "decisions": {},
        "candles": {},
        "walkforward": {},
        "observations": {},
        "ta_signals": {},
        "tsm_forward": {},
        "forward_rules": {},
        "positions": {},
        "sentiment": {},
        "sentiment_ok": False,
        "sentiment_ts": "",
        "error": "",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for attempt in range(3):
        try:
            out["trades"] = _trades()
            break
        except sqlite3.Error as exc:  # locked mid-write: brief retry
            out["error"] = str(exc)
            time.sleep(0.3)
    try:
        out["events"] = _events()
        out["equity"] = _equity()
        if LOG_PATH.exists():
            with LOG_PATH.open("rb") as fh:
                fh.seek(max(0, LOG_PATH.stat().st_size - LOG_TAIL_BYTES))
                _logtext = fh.read().decode("utf-8", errors="replace")
            out["decisions"] = _decisions(_logtext)
            out["ofi_gate"] = _ofi_gate(_logtext)
    except OSError as exc:
        out["error"] += f" | log: {exc}"
    out["candles"] = dict(_candle_cache)
    out["walkforward"] = _walkforward()
    out["observations"] = _obs_stats()
    out["ta_signals"] = _ta_signals()
    out["tsm_forward"] = _tsm_forward()
    out["forward_rules"] = _fwd_rules()
    out["positions"] = _positions()
    out["sentiment"] = dict(_sentiment_cache)
    out["sentiment_ok"] = _sentiment_state["ok"]
    out["sentiment_ts"] = _sentiment_state["ts"]
    return json.dumps(out, default=str).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        if self.path.split("?")[0] == "/data":
            body = _payload()
            content_type = "application/json"
        elif self.path.split("?")[0] in ("/", "/index.html"):
            body = HTML_PATH.read_bytes()
            content_type = "text/html; charset=utf-8"
        elif self.path.split("?")[0] == "/run-search":
            body = _run_search(self.path)
            content_type = "application/json"
        elif self.path.split("?")[0] == "/run-geometry":
            body = _run_geometry(self.path)
            content_type = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        if self.path.split("?")[0] != "/close":
            self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            q = parse_qs(raw) or parse_qs(urlparse(self.path).query)
            symbol = (q.get("symbol", [""])[0]).strip()
            if symbol == "__ALL__":
                res = {"ok": True, "results": [_close_position(p["symbol"])
                                               for p in _positions().get("list", [])]}
            elif symbol:
                res = _close_position(symbol)
            else:
                res = {"ok": False, "err": "no symbol given"}
        except Exception as exc:  # noqa: BLE001
            res = {"ok": False, "err": str(exc)[:120]}
        body = json.dumps(res).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # keep the console quiet


def _lan_ip() -> str:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = str(probe.getsockname()[0])
        probe.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    if not HTML_PATH.exists():
        print(f"dashboard.html not found next to this script ({HTML_PATH})")
        return 1
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    print("Kronos dashboard (read-only) serving:")
    print(f"  this PC : http://localhost:{port}")
    print(f"  phone   : http://{_lan_ip()}:{port}   (same Wi-Fi)")
    print("Ctrl+C to stop. The bot is not affected either way.")
    threading.Thread(target=_candle_loop, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

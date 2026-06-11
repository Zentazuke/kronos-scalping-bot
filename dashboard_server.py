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
import re
import socket
import sqlite3
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Final, List

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DB_PATH: Final[Path] = BASE_DIR / "journal.db"
LOG_PATH: Final[Path] = BASE_DIR / "bot.log"
HTML_PATH: Final[Path] = BASE_DIR / "dashboard.html"
LOG_TAIL_BYTES: Final[int] = 600_000

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
            for r in conn.execute(
                "SELECT id, ts_open, ts_close, symbol, direction, amount,"
                " entry_price, tp_price, sl_price, exit_price, pnl, status,"
                " variant, confluence_votes, p_up, p_down, adx, rsi,"
                " atr, atr_sma, plus_di, minus_di, book_imbalance, meta_p_win"
                " FROM trades ORDER BY id"
            )
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


def _payload() -> bytes:
    out: Dict[str, Any] = {
        "source": "db",
        "trades": [],
        "events": [],
        "equity": None,
        "decisions": {},
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
                out["decisions"] = _decisions(
                    fh.read().decode("utf-8", errors="replace")
                )
    except OSError as exc:
        out["error"] += f" | log: {exc}"
    return json.dumps(out, default=str).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        if self.path.split("?")[0] == "/data":
            body = _payload()
            content_type = "application/json"
        elif self.path.split("?")[0] in ("/", "/index.html"):
            body = HTML_PATH.read_bytes()
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

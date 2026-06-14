"""ta_signals.py — a small, dependency-free technical-analysis signal engine.

Ten classic indicators, one per signal idea, drawn from the project's technical-
analysis research memo (which is emphatic that these are *weak, noisy features*,
not standalone trading rules — this board is a read-only situational display,
never wired into execution). Each indicator maps the latest bar to a direction
(long / short / neutral) and a strength of 1-3, rendered as arrows on the page.

Consensus parameters (textbook defaults, cross-checked June 2026):
  EMA 20/50 · MACD 12/26/9 · ADX/DMI 14 (trend >25, strong >40) · RSI 14 (50
  centre; 30/70) · Stochastic 14/3/3 (20/80) · CCI 20 (+/-100) · Bollinger 20/2s
  · Supertrend ATR 10 x3 · Donchian 20 · OBV 14-bar slope.

Input: candles as [[time_s, open, high, low, close, volume], ...] ascending.
Pure Python, no numpy/pandas, so it runs inside the stdlib dashboard server.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

LONG = "long"
SHORT = "short"
NEUTRAL = "neutral"

MIN_BARS = 60  # need warm-up for the slowest indicator (EMA50 / Donchian20)


# --------------------------------------------------------------------------- #
# Math helpers (all operate on plain lists, return the series or a scalar)     #
# --------------------------------------------------------------------------- #
def _ema(xs: List[float], n: int) -> List[float]:
    k = 2.0 / (n + 1)
    out: List[float] = []
    e = xs[0]
    for x in xs:
        e = x * k + e * (1 - k)
        out.append(e)
    return out


def _sma_last(xs: List[float], n: int) -> float:
    return sum(xs[-n:]) / n


def _std_last(xs: List[float], n: int) -> float:
    window = xs[-n:]
    m = sum(window) / n
    return (sum((x - m) ** 2 for x in window) / n) ** 0.5


def _rsi(c: List[float], n: int = 14) -> float:
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    ag, al = gains / n, losses / n
    for i in range(n + 1, len(c)):
        d = c[i] - c[i - 1]
        ag = (ag * (n - 1) + max(d, 0.0)) / n
        al = (al * (n - 1) + max(-d, 0.0)) / n
    if al == 0:
        return 100.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def _true_ranges(h: List[float], l: List[float], c: List[float]) -> List[float]:
    tr = [h[0] - l[0]]
    for i in range(1, len(c)):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return tr


def _atr(h: List[float], l: List[float], c: List[float], n: int = 14) -> List[float]:
    tr = _true_ranges(h, l, c)
    out = [tr[0]] * len(tr)
    a = sum(tr[:n]) / n
    for i in range(len(tr)):
        if i < n:
            out[i] = a
        else:
            a = (a * (n - 1) + tr[i]) / n
            out[i] = a
    return out


def _adx(h: List[float], l: List[float], c: List[float], n: int = 14) -> Tuple[float, float, float]:
    plus_dm: List[float] = [0.0]
    minus_dm: List[float] = [0.0]
    for i in range(1, len(c)):
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        plus_dm.append(up if (up > dn and up > 0) else 0.0)
        minus_dm.append(dn if (dn > up and dn > 0) else 0.0)
    tr = _true_ranges(h, l, c)
    # Wilder smoothing
    atr = sum(tr[1:n + 1])
    pdm = sum(plus_dm[1:n + 1])
    mdm = sum(minus_dm[1:n + 1])
    dxs: List[float] = []
    for i in range(n + 1, len(c)):
        atr = atr - atr / n + tr[i]
        pdm = pdm - pdm / n + plus_dm[i]
        mdm = mdm - mdm / n + minus_dm[i]
        pdi = 100 * pdm / atr if atr else 0.0
        mdi = 100 * mdm / atr if atr else 0.0
        denom = pdi + mdi
        dxs.append(100 * abs(pdi - mdi) / denom if denom else 0.0)
    if not dxs:
        return 0.0, 0.0, 0.0
    pdi = 100 * pdm / atr if atr else 0.0
    mdi = 100 * mdm / atr if atr else 0.0
    adx = sum(dxs[:n]) / n if len(dxs) >= n else sum(dxs) / len(dxs)
    for i in range(n, len(dxs)):
        adx = (adx * (n - 1) + dxs[i]) / n
    return pdi, mdi, adx


def _stochastic(h, l, c, k: int = 14, d: int = 3, smooth: int = 3) -> Tuple[float, float]:
    raw: List[float] = []
    for i in range(k - 1, len(c)):
        hh = max(h[i - k + 1:i + 1])
        ll = min(l[i - k + 1:i + 1])
        raw.append(100 * (c[i] - ll) / (hh - ll) if hh != ll else 50.0)
    kline = [sum(raw[i - smooth + 1:i + 1]) / smooth for i in range(smooth - 1, len(raw))]
    dline = [sum(kline[i - d + 1:i + 1]) / d for i in range(d - 1, len(kline))]
    return kline[-1], dline[-1]


def _cci(h, l, c, n: int = 20) -> float:
    tp = [(h[i] + l[i] + c[i]) / 3 for i in range(len(c))]
    sma = _sma_last(tp, n)
    mean_dev = sum(abs(x - sma) for x in tp[-n:]) / n
    if mean_dev == 0:
        return 0.0
    return (tp[-1] - sma) / (0.015 * mean_dev)


def _supertrend(h, l, c, period: int = 10, mult: float = 3.0) -> Tuple[int, float, float]:
    atr = _atr(h, l, c, period)
    fu = fl = 0.0
    direction = 1
    st = 0.0
    for i in range(len(c)):
        hl2 = (h[i] + l[i]) / 2
        bu = hl2 + mult * atr[i]
        bl = hl2 - mult * atr[i]
        if i == 0:
            fu, fl, direction, st = bu, bl, 1, bl
            continue
        fu = bu if (bu < fu or c[i - 1] > fu) else fu
        fl = bl if (bl > fl or c[i - 1] < fl) else fl
        if c[i] > fu:
            direction = 1
        elif c[i] < fl:
            direction = -1
        st = fl if direction == 1 else fu
    return direction, st, atr[-1]


def _obv(c: List[float], v: List[float]) -> List[float]:
    out = [0.0]
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            out.append(out[-1] + v[i])
        elif c[i] < c[i - 1]:
            out.append(out[-1] - v[i])
        else:
            out.append(out[-1])
    return out


# --------------------------------------------------------------------------- #
# Indicator -> signal mappers. Each returns (direction, strength 1-3, detail)  #
# --------------------------------------------------------------------------- #
def _sig(d: str, s: int, detail: str):
    return d, s, detail


def _ema_signal(o, h, l, c, v):
    e20, e50, p = _ema(c, 20)[-1], _ema(c, 50)[-1], c[-1]
    diff = (e20 - e50) / e50 if e50 else 0.0
    if abs(diff) < 0.0005:
        return _sig(NEUTRAL, 0, "EMA20 ≈ EMA50")
    d = LONG if diff > 0 else SHORT
    aligned = (p > e20) if d == LONG else (p < e20)
    mag = abs(diff)
    s = 3 if (mag > 0.01 and aligned) else 2 if (mag > 0.003 or aligned) else 1
    return _sig(d, s, f"EMA20 {'>' if diff>0 else '<'} EMA50 ({mag*100:.1f}%)")


def _macd_signal(o, h, l, c, v):
    e12, e26 = _ema(c, 12), _ema(c, 26)
    macd = [a - b for a, b in zip(e12, e26)]
    signal = _ema(macd, 9)
    hist = macd[-1] - signal[-1]
    p = c[-1]
    if abs(hist) / p < 5e-5:
        return _sig(NEUTRAL, 0, "MACD ≈ signal")
    d = LONG if hist > 0 else SHORT
    agree = (macd[-1] > 0) == (hist > 0)
    mag = abs(hist) / p
    s = 3 if (agree and mag > 0.001) else 2 if agree else 1
    return _sig(d, s, f"hist {hist:+.4f}{' (line agrees)' if agree else ''}")


def _adx_signal(o, h, l, c, v):
    pdi, mdi, adx = _adx(h, l, c, 14)
    if adx < 20:
        return _sig(NEUTRAL, 0, f"ADX {adx:.0f} (no trend)")
    d = LONG if pdi > mdi else SHORT
    s = 3 if adx > 40 else 2 if adx > 25 else 1
    return _sig(d, s, f"ADX {adx:.0f}, +DI{'>' if pdi>mdi else '<'}-DI")


def _rsi_signal(o, h, l, c, v):
    r = _rsi(c, 14)
    if 45 < r < 55:
        return _sig(NEUTRAL, 0, f"RSI {r:.0f}")
    d = LONG if r > 50 else SHORT
    dist = abs(r - 50)
    s = 3 if dist > 20 else 2 if dist > 10 else 1
    return _sig(d, s, f"RSI {r:.0f}")


def _stoch_signal(o, h, l, c, v):
    k, dd = _stochastic(h, l, c)
    gap = k - dd
    if abs(gap) < 0.5 and 40 < k < 60:
        return _sig(NEUTRAL, 0, f"%K {k:.0f}")
    d = LONG if gap > 0 else SHORT
    if (d == LONG and k < 25) or (d == SHORT and k > 75):
        s = 3
    elif abs(gap) > 3:
        s = 2
    else:
        s = 1
    return _sig(d, s, f"%K {k:.0f}/%D {dd:.0f}")


def _cci_signal(o, h, l, c, v):
    x = _cci(h, l, c, 20)
    if abs(x) < 40:
        return _sig(NEUTRAL, 0, f"CCI {x:.0f}")
    d = LONG if x > 0 else SHORT
    s = 3 if abs(x) > 200 else 2 if abs(x) > 100 else 1
    return _sig(d, s, f"CCI {x:.0f}")


def _boll_signal(o, h, l, c, v):
    mid, sd, p = _sma_last(c, 20), _std_last(c, 20), c[-1]
    upper, lower = mid + 2 * sd, mid - 2 * sd
    if upper == lower:
        return _sig(NEUTRAL, 0, "flat bands")
    pb = (p - lower) / (upper - lower)
    if p > upper:
        return _sig(LONG, 3, "above upper band")
    if p < lower:
        return _sig(SHORT, 3, "below lower band")
    if pb > 0.75:
        return _sig(LONG, 2, f"%B {pb:.2f}")
    if pb < 0.25:
        return _sig(SHORT, 2, f"%B {pb:.2f}")
    if pb > 0.58:
        return _sig(LONG, 1, f"%B {pb:.2f}")
    if pb < 0.42:
        return _sig(SHORT, 1, f"%B {pb:.2f}")
    return _sig(NEUTRAL, 0, f"%B {pb:.2f}")


def _supertrend_signal(o, h, l, c, v):
    direction, st, atr = _supertrend(h, l, c, 10, 3.0)
    d = LONG if direction == 1 else SHORT
    dist = abs(c[-1] - st) / atr if atr else 0.0
    s = 3 if dist > 2 else 2 if dist > 1 else 1
    return _sig(d, s, f"{'above' if direction==1 else 'below'} ST ({dist:.1f} ATR)")


def _obv_signal(o, h, l, c, v):
    obv = _obv(c, v)
    if len(obv) < 15:
        return _sig(NEUTRAL, 0, "n/a")
    slope = obv[-1] - obv[-15]
    vol = sum(abs(x) for x in v[-14:]) + 1e-9
    norm = slope / vol
    if abs(norm) < 0.05:
        return _sig(NEUTRAL, 0, "flat OBV")
    d = LONG if slope > 0 else SHORT
    s = 3 if abs(norm) > 0.5 else 2 if abs(norm) > 0.2 else 1
    return _sig(d, s, f"OBV {'rising' if slope>0 else 'falling'}")


def _donchian_signal(o, h, l, c, v):
    upper = max(h[-21:-1])
    lower = min(l[-21:-1])
    p = c[-1]
    if upper == lower:
        return _sig(NEUTRAL, 0, "flat channel")
    if p >= upper:
        return _sig(LONG, 3, "20-bar high breakout")
    if p <= lower:
        return _sig(SHORT, 3, "20-bar low breakout")
    pos = (p - lower) / (upper - lower)
    if pos > 0.8:
        return _sig(LONG, 2, f"upper channel ({pos:.0%})")
    if pos < 0.2:
        return _sig(SHORT, 2, f"lower channel ({pos:.0%})")
    if pos > 0.6:
        return _sig(LONG, 1, f"channel {pos:.0%}")
    if pos < 0.4:
        return _sig(SHORT, 1, f"channel {pos:.0%}")
    return _sig(NEUTRAL, 0, f"mid channel ({pos:.0%})")


_INDICATORS = [
    ("EMA 20/50", "Trend", _ema_signal),
    ("MACD", "Trend", _macd_signal),
    ("ADX / DMI", "Trend", _adx_signal),
    ("Supertrend", "Trend", _supertrend_signal),
    ("RSI", "Momentum", _rsi_signal),
    ("Stochastic", "Momentum", _stoch_signal),
    ("CCI", "Momentum", _cci_signal),
    ("Bollinger", "Volatility", _boll_signal),
    ("Donchian", "Volatility", _donchian_signal),
    ("OBV", "Volume", _obv_signal),
]


def compute_signals(candles: List[List[float]]) -> Optional[Dict]:
    """Map a symbol's candles to ten indicator signals plus a net bias.

    Returns None when there isn't enough history. Each signal is
    {name, family, dir, strength, detail}; net is the sum of signed strengths.
    """
    if not candles or len(candles) < MIN_BARS:
        return None
    o = [r[1] for r in candles]
    h = [r[2] for r in candles]
    l = [r[3] for r in candles]
    c = [r[4] for r in candles]
    v = [r[5] if len(r) > 5 else 0.0 for r in candles]

    signals: List[Dict] = []
    net = 0
    longs = shorts = 0
    for name, family, fn in _INDICATORS:
        try:
            d, s, detail = fn(o, h, l, c, v)
        except Exception:  # noqa: BLE001 — one bad indicator must not sink the board
            d, s, detail = NEUTRAL, 0, "error"
        if d == LONG:
            net += s
            longs += 1
        elif d == SHORT:
            net -= s
            shorts += 1
        signals.append(
            {"name": name, "family": family, "dir": d, "strength": s, "detail": detail}
        )

    if net >= 4:
        bias = "LONG"
    elif net <= -4:
        bias = "SHORT"
    else:
        bias = "NEUTRAL"
    return {
        "signals": signals,
        "net": net,
        "bias": bias,
        "longs": longs,
        "shorts": shorts,
        "tf": "5m",
        "bars": len(candles),
    }

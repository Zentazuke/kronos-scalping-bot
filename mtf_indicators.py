"""mtf_indicators.py — faithful full-series technical indicators for the
multi-timeframe scalping backtest. Every function returns a list aligned to the
input bars, with None during warm-up. Standard (Wilder / convention) maths so
the values match TradingView / TA-Lib conventions.

candles are [[ts, open, high, low, close, volume], ...] ascending.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Num = Optional[float]


def ema(values: Sequence[float], period: int) -> List[Num]:
    out: List[Num] = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    e = seed
    for i in range(period, len(values)):
        e = values[i] * k + e * (1 - k)
        out[i] = e
    return out


def _wilder(values: Sequence[Num], period: int) -> List[Num]:
    """Wilder's RMA over a series that may start with leading None."""
    out: List[Num] = [None] * len(values)
    # find first index with `period` consecutive real numbers
    start = None
    for i in range(len(values)):
        if values[i] is None:
            continue
        if start is None:
            start = i
        if i - start + 1 >= period:
            break
    if start is None or start + period > len(values):
        return out
    first = start + period - 1
    seed = sum(float(v) for v in values[start:first + 1]) / period  # type: ignore[arg-type]
    out[first] = seed
    r = seed
    for i in range(first + 1, len(values)):
        v = values[i]
        if v is None:
            out[i] = r
            continue
        r = (r * (period - 1) + float(v)) / period
        out[i] = r
    return out


def rsi(closes: Sequence[float], period: int = 14) -> List[Num]:
    n = len(closes)
    out: List[Num] = [None] * n
    if n < period + 1:
        return out
    gains, losses = [], []
    for i in range(1, n):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period

    def rv(ag: float, al: float) -> float:
        return 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)

    out[period] = rv(ag, al)
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + gains[i - 1]) / period
        al = (al * (period - 1) + losses[i - 1]) / period
        out[i] = rv(ag, al)
    return out


def atr(candles: Sequence[Sequence[float]], period: int = 14) -> List[Num]:
    n = len(candles)
    out: List[Num] = [None] * n
    if n < period + 1:
        return out
    trs = []
    for i in range(1, n):
        h, l, pc = candles[i][2], candles[i][3], candles[i - 1][4]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    a = sum(trs[:period]) / period
    out[period] = a
    for i in range(period + 1, n):
        a = (a * (period - 1) + trs[i - 1]) / period
        out[i] = a
    return out


def adx_di(candles: Sequence[Sequence[float]], period: int = 14
           ) -> Tuple[List[Num], List[Num], List[Num]]:
    """Returns (+DI, -DI, ADX) series (Wilder)."""
    n = len(candles)
    plus_di: List[Num] = [None] * n
    minus_di: List[Num] = [None] * n
    adx: List[Num] = [None] * n
    if n < 2 * period + 1:
        return plus_di, minus_di, adx
    tr, pdm, ndm = [0.0], [0.0], [0.0]  # index aligned to candle i (i>=1 meaningful)
    for i in range(1, n):
        h, l = candles[i][2], candles[i][3]
        ph, pl, pc = candles[i - 1][2], candles[i - 1][3], candles[i - 1][4]
        up, dn = h - ph, pl - l
        pdm.append(up if (up > dn and up > 0) else 0.0)
        ndm.append(dn if (dn > up and dn > 0) else 0.0)
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    # Wilder smooth starting at index `period`
    str_ = sum(tr[1:period + 1])
    spdm = sum(pdm[1:period + 1])
    sndm = sum(ndm[1:period + 1])
    dx_list: List[Tuple[int, float]] = []

    def di(sp: float, sn: float, st: float) -> Tuple[float, float]:
        p = 100.0 * sp / st if st else 0.0
        m = 100.0 * sn / st if st else 0.0
        return p, m

    p, m = di(spdm, sndm, str_)
    plus_di[period], minus_di[period] = p, m
    denom = p + m
    dx_list.append((period, 100.0 * abs(p - m) / denom if denom else 0.0))
    for i in range(period + 1, n):
        str_ = str_ - str_ / period + tr[i]
        spdm = spdm - spdm / period + pdm[i]
        sndm = sndm - sndm / period + ndm[i]
        p, m = di(spdm, sndm, str_)
        plus_di[i], minus_di[i] = p, m
        denom = p + m
        dx_list.append((i, 100.0 * abs(p - m) / denom if denom else 0.0))
    # ADX = Wilder smooth of DX, first value at index period+ (period-1)
    if len(dx_list) >= period:
        first_adx_pos = period - 1  # within dx_list
        a = sum(d for _, d in dx_list[:period]) / period
        adx[dx_list[first_adx_pos][0]] = a
        for j in range(period, len(dx_list)):
            a = (a * (period - 1) + dx_list[j][1]) / period
            adx[dx_list[j][0]] = a
    return plus_di, minus_di, adx


def supertrend(candles: Sequence[Sequence[float]], period: int = 10, mult: float = 3.0
               ) -> Tuple[List[Num], List[Optional[int]]]:
    """Returns (supertrend_line, direction) where direction is +1 bullish / -1 bearish."""
    n = len(candles)
    line: List[Num] = [None] * n
    direction: List[Optional[int]] = [None] * n
    a = atr(candles, period)
    fu: List[Num] = [None] * n
    fl: List[Num] = [None] * n
    for i in range(n):
        if a[i] is None:
            continue
        hl2 = (candles[i][2] + candles[i][3]) / 2
        bu = hl2 + mult * a[i]  # type: ignore[operator]
        bl = hl2 - mult * a[i]  # type: ignore[operator]
        if i == 0 or fu[i - 1] is None:
            fu[i], fl[i] = bu, bl
            direction[i] = -1 if candles[i][4] <= bu else 1
            line[i] = fu[i] if direction[i] == -1 else fl[i]
            continue
        pc = candles[i - 1][4]
        fu[i] = bu if (bu < fu[i - 1] or pc > fu[i - 1]) else fu[i - 1]  # type: ignore[operator]
        fl[i] = bl if (bl > fl[i - 1] or pc < fl[i - 1]) else fl[i - 1]  # type: ignore[operator]
        prev_dir = direction[i - 1]
        c = candles[i][4]
        if prev_dir == 1:
            direction[i] = -1 if c < fl[i] else 1  # type: ignore[operator]
        else:
            direction[i] = 1 if c > fu[i] else -1  # type: ignore[operator]
        line[i] = fl[i] if direction[i] == 1 else fu[i]
    return line, direction


def stochastic(candles: Sequence[Sequence[float]], k: int = 14, d: int = 3, smooth: int = 3
               ) -> Tuple[List[Num], List[Num]]:
    """Slow stochastic: returns (%K_slow, %D)."""
    n = len(candles)
    raw: List[Num] = [None] * n
    for i in range(n):
        if i < k - 1:
            continue
        hh = max(candles[j][2] for j in range(i - k + 1, i + 1))
        ll = min(candles[j][3] for j in range(i - k + 1, i + 1))
        raw[i] = 100.0 * (candles[i][4] - ll) / (hh - ll) if hh > ll else 50.0
    k_slow = _sma_series(raw, smooth)
    d_line = _sma_series(k_slow, d)
    return k_slow, d_line


def _sma_series(values: Sequence[Num], period: int) -> List[Num]:
    out: List[Num] = [None] * len(values)
    buf: List[float] = []
    for i, v in enumerate(values):
        if v is None:
            buf = []
            continue
        buf.append(float(v))
        if len(buf) > period:
            buf.pop(0)
        if len(buf) == period:
            out[i] = sum(buf) / period
    return out


def macd_hist(closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
              ) -> List[Num]:
    ef, es = ema(closes, fast), ema(closes, slow)
    macd_line: List[Num] = [None] * len(closes)
    for i in range(len(closes)):
        if ef[i] is not None and es[i] is not None:
            macd_line[i] = ef[i] - es[i]  # type: ignore[operator]
    sig = _ema_series(macd_line, signal)
    hist: List[Num] = [None] * len(closes)
    for i in range(len(closes)):
        if macd_line[i] is not None and sig[i] is not None:
            hist[i] = macd_line[i] - sig[i]  # type: ignore[operator]
    return hist


def _ema_series(values: Sequence[Num], period: int) -> List[Num]:
    out: List[Num] = [None] * len(values)
    k = 2.0 / (period + 1)
    e = None
    seedbuf: List[float] = []
    for i, v in enumerate(values):
        if v is None:
            continue
        if e is None:
            seedbuf.append(float(v))
            if len(seedbuf) == period:
                e = sum(seedbuf) / period
                out[i] = e
            continue
        e = float(v) * k + e * (1 - k)
        out[i] = e
    return out


def bollinger_width(closes: Sequence[float], period: int = 20, mult: float = 2.0) -> List[Num]:
    n = len(closes)
    out: List[Num] = [None] * n
    for i in range(period - 1, n):
        win = closes[i - period + 1:i + 1]
        m = sum(win) / period
        var = sum((x - m) ** 2 for x in win) / period
        sd = var ** 0.5
        out[i] = (2 * mult * sd) / m if m else None
    return out


def donchian(candles: Sequence[Sequence[float]], period: int = 20
             ) -> Tuple[List[Num], List[Num]]:
    n = len(candles)
    hi: List[Num] = [None] * n
    lo: List[Num] = [None] * n
    for i in range(period, n):  # previous N bars (excludes current)
        hi[i] = max(candles[j][2] for j in range(i - period, i))
        lo[i] = min(candles[j][3] for j in range(i - period, i))
    return hi, lo


def obv(candles: Sequence[Sequence[float]]) -> List[float]:
    out = [0.0] * len(candles)
    for i in range(1, len(candles)):
        if candles[i][4] > candles[i - 1][4]:
            out[i] = out[i - 1] + candles[i][5]
        elif candles[i][4] < candles[i - 1][4]:
            out[i] = out[i - 1] - candles[i][5]
        else:
            out[i] = out[i - 1]
    return out


def rel_volume(candles: Sequence[Sequence[float]], period: int = 20) -> List[Num]:
    vols = [c[5] for c in candles]
    sma = _sma_series([float(v) for v in vols], period)
    out: List[Num] = [None] * len(candles)
    for i in range(len(candles)):
        if sma[i] and sma[i] > 0:  # type: ignore[operator]
            out[i] = vols[i] / sma[i]  # type: ignore[operator]
    return out


def vwap_daily(candles: Sequence[Sequence[float]]) -> List[Num]:
    """Daily-anchored VWAP (resets each UTC day)."""
    out: List[Num] = [None] * len(candles)
    cur_day = None
    cum_pv = cum_v = 0.0
    for i, c in enumerate(candles):
        day = int(c[0]) // 86_400_000
        if day != cur_day:
            cur_day, cum_pv, cum_v = day, 0.0, 0.0
        tp = (c[2] + c[3] + c[4]) / 3
        cum_pv += tp * c[5]
        cum_v += c[5]
        out[i] = cum_pv / cum_v if cum_v else None
    return out


def swing_low(candles: Sequence[Sequence[float]], i: int, lookback: int = 10) -> float:
    return min(candles[j][3] for j in range(max(0, i - lookback), i + 1))


def swing_high(candles: Sequence[Sequence[float]], i: int, lookback: int = 10) -> float:
    return max(candles[j][2] for j in range(max(0, i - lookback), i + 1))

"""Signal generators. Each returns boolean arrays (long, short) evaluated on the CLOSE of bar i.
Entry is simulated on the OPEN of bar i+1, so there is no look-ahead."""
from itertools import product

import numpy as np
import pandas as pd


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _arr(x):
    return np.asarray(x.fillna(False) if x.dtype == object else x, dtype=bool)


def ema_trend(df, fast, slow, **_):
    c = df["close"]
    f, s, base = ema(c, fast), ema(c, slow), ema(c, 200)
    up = (f > s) & (f.shift() <= s.shift())
    dn = (f < s) & (f.shift() >= s.shift())
    return _arr(up & (c > base)), _arr(dn & (c < base))


def rsi_reversion(df, th, **_):
    c = df["close"]
    r, base = rsi(c, 14), ema(c, 200)
    return _arr((r < th) & (r.shift() >= th) & (c > base)), _arr((r > 100 - th) & (r.shift() <= 100 - th) & (c < base))


def adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def donchian(df, n, adx_min=0, vol_mult=0, **_):
    """Breakout of the n-bar channel, with EMA200 trend filter, optional ADX (trend strength)
    and volume (> vol_mult x 20-bar average) confirmation."""
    c, base = df["close"], ema(df["close"], 200)
    hi, lo = df["high"].rolling(n).max().shift(), df["low"].rolling(n).min().shift()
    ok = pd.Series(True, index=df.index)
    if adx_min:
        ok &= adx(df) > adx_min
    if vol_mult:
        ok &= df["volume"] > vol_mult * df["volume"].rolling(20).mean()
    return _arr((c > hi) & (c > base) & ok), _arr((c < lo) & (c < base) & ok)


def bb_reversion(df, k, **_):
    c = df["close"]
    m, sd, r, base = c.rolling(20).mean(), c.rolling(20).std(), rsi(c, 14), ema(c, 200)
    return _arr((c < m - k * sd) & (r < 40) & (c > base)), _arr((c > m + k * sd) & (r > 60) & (c < base))


def _grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, v)) for v in product(*axes.values())]


# `sl_atr` = stop distance in ATRs, `tp_r` = target in multiples of the stop distance
STRATEGIES = {
    "ema_trend": (ema_trend, _grid(fast=[10, 20], slow=[50, 100], sl_atr=[1.5, 2.5], tp_r=[2, 3])),
    "rsi_reversion": (rsi_reversion, _grid(th=[30, 35, 40], sl_atr=[1.5, 2.5], tp_r=[1, 1.5, 2])),
    "donchian": (donchian, _grid(n=[20, 55], sl_atr=[2, 3], tp_r=[2, 3])),
    # trend following with a trailing stop: let winners run (hold up to 20 days)
    "donchian_trail": (donchian, _grid(n=[20, 55], adx_min=[0, 20], vol_mult=[0, 1.2], sl_atr=[2.5],
                                       tp_r=[0], trail=[3, 4.5], hold=[20])),
    "ema_trail": (ema_trend, _grid(fast=[10, 20], slow=[50, 100], sl_atr=[2.5], tp_r=[0], trail=[3, 4.5], hold=[20])),
    "bb_reversion": (bb_reversion, _grid(k=[2, 2.5], sl_atr=[1.5, 2.5], tp_r=[1, 2])),
}

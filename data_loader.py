"""Download and cache OHLCV candles from public exchange endpoints (no API key needed)."""
import time
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
EXCHANGES = ["binance", "bybit", "okx", "kucoin", "gate", "bitget", "kraken"]  # fallback order (several are geo-blocked from the US, where GitHub runs)
_PREFERRED = None  # exchange that worked last (so blocked ones are not retried for every coin)
TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def _cache_path(symbol: str, tf: str) -> Path:
    return DATA_DIR / f"{symbol.replace('/', '_')}_{tf}.csv"


def _download(ex, symbol: str, tf: str, since_ms: int) -> list:
    rows, step = [], TF_MS[tf]
    while True:
        for attempt in range(3):
            try:
                batch = ex.fetch_ohlcv(symbol, tf, since=since_ms, limit=1000)
                break
            except (ccxt.RequestTimeout, ccxt.RateLimitExceeded, ccxt.DDoSProtection):  # transient -> retry
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"network failure fetching {symbol} {tf}")
        if not batch:
            break
        rows += batch
        nxt = batch[-1][0] + step
        if nxt <= since_ms:
            break
        since_ms = nxt
        if len(batch) < 2:
            break
    return rows


def load(symbol: str, tf: str, since: str = "2019-01-01", refresh: bool = True) -> pd.DataFrame:
    """Return DataFrame indexed by UTC open time with open/high/low/close/volume (closed candles only)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = _cache_path(symbol, tf)
    cached = None
    since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    if path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        cached.index = pd.to_datetime(cached.index, utc=True)
        since_ms = int(cached.index[-1].timestamp() * 1000)
    if refresh or cached is None:
        last_err = None
        for name in EXCHANGES:
            try:
                ex = getattr(ccxt, name)({"enableRateLimit": True})
                rows = _download(ex, symbol, tf, since_ms)
                break
            except Exception as e:  # geo-block, unknown symbol, ...
                last_err = e
        else:
            if cached is None:
                raise RuntimeError(f"all exchanges failed for {symbol}: {last_err}")
            rows = []
        if rows:
            new = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            new.index = pd.to_datetime(new.pop("ts"), unit="ms", utc=True)
            df = new if cached is None else pd.concat([cached, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(path)
            cached = df
    df = cached
    # drop the still-forming candle
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(milliseconds=TF_MS[tf]) <= now]
    return df


def fetch_recent(symbol: str, tf: str, limit: int = 700):
    """Latest `limit` closed candles straight from an exchange (used by the live bot). Returns (df, exchange_name).
    Paginates forward from (now - limit candles) because some exchanges (OKX) cap one request at ~300 candles."""
    global _PREFERRED
    last_err = None
    step = TF_MS[tf]
    since_ms = int(time.time() * 1000) - (limit + 5) * step
    order = ([_PREFERRED] if _PREFERRED else []) + [n for n in EXCHANGES if n != _PREFERRED]
    for name in order:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            rows = _download(ex, symbol, tf, since_ms)
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            now = pd.Timestamp.now(tz="UTC")
            df = df[df.index + pd.Timedelta(milliseconds=step) <= now]
            if len(df) < min(limit, 400) * 0.9:
                raise RuntimeError(f"{name} returned only {len(df)} candles")
            _PREFERRED = name
            return df, name
        except Exception as e:
            last_err = f"{name}: {str(e)[:80]}"
            if name == _PREFERRED:
                _PREFERRED = None
    raise RuntimeError(f"all exchanges failed for {symbol}: {last_err}")

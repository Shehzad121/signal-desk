"""Search the best strategy PER TIMEFRAME with honest walk-forward (parameters picked on past data only,
judged on the unseen next window), futures fees, 7 coins.   python mtf_search.py  -> results/mtf_results.csv"""
import sys
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

COINS = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB"]
TFS = ["15m", "30m", "1h", "2h", "4h", "1d"]
BPD = {"15m": 96, "30m": 48, "1h": 24, "2h": 12, "4h": 6, "1d": 1}
HOLD = {"15m": 1, "30m": 2, "1h": 3, "2h": 3, "4h": 3, "1d": 10}              # days, fixed-target strategies
HOLD_TRAIL = {"15m": 3, "30m": 5, "1h": 10, "2h": 15, "4h": 20, "1d": 40}      # days, trailing-stop strategies
FEE, SLIP = 0.0005, 0.0003
FAMILIES = ["donchian", "don_trail", "ema_trend", "rsi_reversion", "bb_reversion"]


def load(coin, tf):
    import data_loader
    if tf in ("15m", "1h", "4h"):
        return data_loader.load(f"{coin}/USDT", tf, refresh=False)
    src, rule = {"30m": ("15m", "30min"), "2h": ("1h", "2h"), "1d": ("4h", "1D")}[tf]
    d = data_loader.load(f"{coin}/USDT", src, refresh=False)
    return d.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def setup_grids(tf):
    from strategies import STRATEGIES, _grid, donchian
    STRATEGIES["donchian"] = (donchian, _grid(n=[20, 40, 80, 160], adx_min=[0, 20], sl_atr=[2, 3], tp_r=[2, 3, 4]))
    STRATEGIES["don_trail"] = (donchian, _grid(n=[20, 40, 80, 160], adx_min=[0, 20], sl_atr=[2.5, 3], tp_r=[0],
                                               trail=[3, 4.5, 6], hold=[HOLD_TRAIL[tf]]))


def task(args_):
    coin, tf, fam, long_only = args_
    import run_backtest as rb
    from strategies import STRATEGIES  # noqa: F401
    rb.BARS_PER_DAY.update(BPD)
    setup_grids(tf)
    df = load(coin, tf)
    daily = tf == "1d"
    a = Namespace(train_days=730 if daily else 365, test_days=180 if daily else 90, max_hold_days=HOLD[tf], fee=FEE, slip=SLIP,
                  risk=0.005, max_lev=3.0, long_only=long_only, min_trades=15 if daily else 30)
    try:
        tr, folds, _ = rb.walk_forward(df, fam, a, tf)
    except Exception as e:  # noqa: BLE001
        return args_, None, str(e)
    if not len(tr):
        return args_, None, "no trades"
    d = pd.DataFrame(tr, columns=["entry_i", "exit_i", "side", "R", "risk_frac", "reason"])
    d["entry_time"] = df.index[d.entry_i.astype(int)]
    d["exit_time"] = df.index[d.exit_i.astype(int)]
    d["coin"] = coin
    fpos = [f["exp_R"] > 0 for f in folds if f["params"] is not None]
    return args_, d[["coin", "entry_time", "exit_time", "side", "R", "risk_frac"]], (sum(fpos), len(fpos))


def main():
    from metrics import summarize, bootstrap
    from run_backtest import portfolio
    tasks = [(c, tf, f, lo) for tf in TFS for f in FAMILIES for lo in (False, True) for c in COINS]
    print(f"{len(tasks)} tasks on 12 cores", flush=True)
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=11) as ex:
        for i, (key, d, info) in enumerate(ex.map(task, tasks, chunksize=1), 1):
            res[key] = (d, info)
            if i % 20 == 0:
                print(f"  {i}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)
    rows = []
    for tf in TFS:
        for fam in FAMILIES:
            for lo in (False, True):
                parts = [res[(c, tf, fam, lo)][0] for c in COINS if res[(c, tf, fam, lo)][0] is not None]
                if not parts:
                    continue
                d = pd.concat(parts)
                s = summarize(d[["R"]].assign(a=0, b=0, c=d.R, e=d.risk_frac).to_numpy()[:, [0, 1, 2, 3, 4]][:, [0, 1, 2, 2, 3]] if False else
                              np.column_stack([np.zeros(len(d)), np.zeros(len(d)), d.side, d.R, d.risk_frac, np.zeros(len(d))]))
                bs = bootstrap(np.column_stack([np.zeros(len(d)), np.zeros(len(d)), d.side, d.R, d.risk_frac, np.zeros(len(d))]))
                p = portfolio(d, 0.02)
                fp = [res[(c, tf, fam, lo)][1] for c in COINS if res[(c, tf, fam, lo)][0] is not None]
                rows.append(dict(tf=tf, strategy=fam, dirs="long" if lo else "both", n=s["n"], win=s["win"], exp_R=s["exp_R"], pf=s["pf"], t=s["t"],
                                 boot_p=bs["p_profit"], ann=p["ann"], dd=p["dd"], calmar=p["calmar"],
                                 folds_pos=sum(a for a, b in fp) / max(1, sum(b for a, b in fp))))
    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/mtf_results.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n==== ALL (out-of-sample, 7 coins pooled) ====")
    print(out.round(3).to_string(index=False))
    print("\n==== BEST PER TIMEFRAME (by t-stat, needs n>=150 and exp>0) ====")
    ok = out[(out.n >= 150) & (out.exp_R > 0)]
    print(ok.sort_values(["tf", "t"], ascending=[True, False]).groupby("tf").head(2).round(3).to_string(index=False))
    print(f"\nfinished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

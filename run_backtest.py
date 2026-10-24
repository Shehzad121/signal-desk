"""Walk-forward backtest. Parameters are picked ONLY on past data, then judged on the unseen next window.

    python run_backtest.py                       # default coins, 1h + 4h, all strategies
    python run_backtest.py --symbols BTC ETH --tfs 4h --strategies donchian ema_trend
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data_loader
from engine import COLS, simulate
from metrics import bootstrap, summarize
from strategies import STRATEGIES, atr

BARS_PER_DAY = {"15m": 96, "1h": 24, "4h": 6, "1d": 1}
DEFAULT_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LINK"]
OUT = Path(__file__).parent / "results"


def walk_forward(df, name, args, tf):
    fn, grid = STRATEGIES[name]
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr(df).to_numpy(float)
    bpd = BARS_PER_DAY[tf]
    train, test = args.train_days * bpd, args.test_days * bpd
    max_bars = args.max_hold_days * bpd
    sigs = [fn(df, **p) for p in grid]  # signals are causal, so compute once per param set

    def run(k, s, e):
        p = grid[k]
        mb = p.get("hold", args.max_hold_days) * bpd
        return simulate(o, h, l, c, a, sigs[k][0], sigs[k][1], s, e, p["sl_atr"], p["tp_r"], mb,
                        args.fee, args.slip, not args.long_only, args.risk, args.max_lev, p.get("trail", 0.0))

    oos, folds, is_exp = [], [], []
    start = 250  # indicator warm-up
    while start + train + test <= len(df):
        tr_s, tr_e, te_e = start, start + train, start + train + test
        best, best_score, best_is = None, -np.inf, 0
        for k in range(len(grid)):
            t = run(k, tr_s, tr_e)
            if len(t) < args.min_trades:
                continue
            s = summarize(t)
            score = s["exp_R"] * np.sqrt(s["n"])  # reward edge AND sample size
            if score > best_score:
                best, best_score, best_is = k, score, s["exp_R"]
        if best is not None and best_score > 0:
            t = run(best, tr_e, te_e)
            oos.append(t)
            folds.append(dict(start=df.index[tr_e], params=grid[best], n=len(t), exp_R=t[:, 3].mean() if len(t) else 0))
            is_exp.append(best_is)
        else:
            folds.append(dict(start=df.index[tr_e], params=None, n=0, exp_R=0))  # no edge found -> sit out
        start += test
    tr = np.vstack(oos) if oos else np.zeros((0, 6))
    return tr, folds, float(np.mean(is_exp)) if is_exp else 0.0


def portfolio(d, risk_cap):
    """Realistic multi-coin account: trades from all coins share one equity curve, and a new trade is
    skipped if total open risk would exceed risk_cap (e.g. 0.04 = 4% of equity at once)."""
    if d.empty:
        return dict(ann=0.0, dd=0.0, calmar=0.0, taken=0, skipped=0)
    d = d.sort_values("entry_time")
    open_, taken = [], []
    for r in d.itertuples():
        open_ = [(x, k) for x, k in open_ if x > r.entry_time]
        if sum(k for _, k in open_) + r.risk_frac <= risk_cap + 1e-12:
            open_.append((r.exit_time, r.risk_frac))
            taken.append(r)
    t = pd.DataFrame(taken).sort_values("exit_time")
    eq = np.cumprod(1 + t["R"].to_numpy() * t["risk_frac"].to_numpy())
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    years = max((t["exit_time"].max() - d["entry_time"].min()).days / 365.25, 0.25)
    ann = eq[-1] ** (1 / years) - 1
    return dict(ann=ann, dd=dd, calmar=ann / abs(dd) if dd < 0 else 0.0, taken=len(t), skipped=len(d) - len(t))


def verdict(s, bs, folds, args):
    active = [f for f in folds if f["params"] is not None]
    pos = np.mean([f["exp_R"] > 0 for f in active]) if active else 0
    checks = {
        f"trades>={args.min_oos_trades}": s["n"] >= args.min_oos_trades,
        "expectancy>0.05R": s["exp_R"] > 0.05,
        "profit factor>1.15": s["pf"] > 1.15,
        f"t-stat>={args.min_t}": s["t"] >= args.min_t,
        "profitable folds>=55%": pos >= 0.55,
        "bootstrap P(profit)>=90%": bs["p_profit"] >= 0.90,
    }
    return all(checks.values()), [k for k, v in checks.items() if not v], pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_COINS)
    ap.add_argument("--tfs", nargs="+", default=["1h", "4h"])
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--max-hold-days", type=int, default=3)
    ap.add_argument("--fee", type=float, default=0.001, help="per side, 0.001 = 0.1%% (Binance spot taker)")
    ap.add_argument("--slip", type=float, default=0.0005, help="per side slippage")
    ap.add_argument("--risk", type=float, default=0.01, help="equity risked per trade")
    ap.add_argument("--max-lev", type=float, default=3.0)
    ap.add_argument("--risk-cap", type=float, default=0.04, help="max total open risk across coins")
    ap.add_argument("--max-dd", type=float, default=0.35, help="portfolio max drawdown allowed for PASS")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--min-trades", type=int, default=30, help="min in-sample trades to accept a param set")
    ap.add_argument("--min-oos-trades", type=int, default=150)
    ap.add_argument("--min-t", type=float, default=2.5, help="strict because we test many strategies")
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    summary, all_tr = [], []
    for tf in args.tfs:
        data = {}
        for sym in args.symbols:
            try:
                data[sym] = data_loader.load(f"{sym}/USDT", tf, args.since, refresh=not args.no_refresh)
            except Exception as e:
                print(f"skip {sym} {tf}: {e}")
        for name in args.strategies:
            pooled, per_coin, folds_all, is_all, strat_tr = [], [], [], [], []
            for sym, df in data.items():
                tr, folds, is_e = walk_forward(df, name, args, tf)
                if len(tr):
                    pooled.append(tr)
                    d = pd.DataFrame(tr, columns=COLS)
                    d["symbol"], d["strategy"], d["tf"] = sym, name, tf
                    d["entry_time"] = df.index[d["entry_i"].astype(int)]
                    d["exit_time"] = df.index[d["exit_i"].astype(int)]
                    all_tr.append(d)
                    strat_tr.append(d)
                folds_all += folds
                is_all.append(is_e)
                years = (len(df) - 250 - args.train_days * BARS_PER_DAY[tf]) / BARS_PER_DAY[tf] / 365
                s = summarize(tr, years)
                per_coin.append((sym, s))
            tr = np.vstack(pooled) if pooled else np.zeros((0, 6))
            s, bs = summarize(tr), bootstrap(tr)
            ok, fails, pos = verdict(s, bs, folds_all, args)
            pf_ = portfolio(pd.concat(strat_tr) if strat_tr else pd.DataFrame(), args.risk_cap)
            if pf_["dd"] < -args.max_dd:
                ok = False
                fails.append(f"portfolio maxDD worse than -{args.max_dd:.0%}")
            summary.append(dict(tf=tf, strategy=name, trades=s["n"], win=s["win"], avg_win_R=s["avg_win"],
                                avg_loss_R=s["avg_loss"], exp_R=s["exp_R"], pf=s["pf"], t=s["t"],
                                IS_exp_R=np.mean(is_all), prof_folds=pos, p_profit=bs["p_profit"],
                                port_ann=pf_["ann"], port_dd=pf_["dd"], calmar=pf_["calmar"],
                                verdict="PASS" if ok else "FAIL", why_failed="; ".join(fails)))
            print(f"portfolio ({args.risk:.1%} risk/trade, cap {args.risk_cap:.0%}): ann={pf_['ann']:+.1%} "
                  f"maxDD={pf_['dd']:.1%} calmar={pf_['calmar']:.2f} taken={pf_['taken']} skipped={pf_['skipped']}")
            print(f"\n=== {name} [{tf}]  {'PASS' if ok else 'FAIL'}  ===")
            print(f"pooled OOS: n={s['n']} win={s['win']:.1%} avgW={s['avg_win']:.2f}R avgL={s['avg_loss']:.2f}R "
                  f"exp={s['exp_R']:+.3f}R PF={s['pf']:.2f} t={s['t']:.2f} | in-sample exp={np.mean(is_all):+.3f}R")
            print("per coin OOS:", "  ".join(f"{sym}:{x['exp_R']:+.2f}R/{x['n']}tr" for sym, x in per_coin))
            if fails:
                print("failed:", ", ".join(fails))
    df_sum = pd.DataFrame(summary)
    OUT.mkdir(exist_ok=True)
    df_sum.to_csv(OUT / "summary.csv", index=False)
    if all_tr:
        pd.concat(all_tr).to_csv(OUT / "oos_trades.csv", index=False)
    fmt = df_sum.copy()
    for col in ("win", "prof_folds", "p_profit", "port_ann", "port_dd"):
        fmt[col] = (fmt[col] * 100).round(1).astype(str) + "%"
    print("\n\n######## SUMMARY (out-of-sample only) ########")
    print(fmt.drop(columns="why_failed").round(3).to_string(index=False))
    print("\nSaved: results/summary.csv, results/oos_trades.csv")


if __name__ == "__main__":
    main()

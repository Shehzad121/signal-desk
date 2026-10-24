import numpy as np


def summarize(tr: np.ndarray, years: float = None) -> dict:
    """tr = trade array from engine.simulate (can be concatenated across folds/coins)."""
    n = len(tr)
    if n == 0:
        return dict(n=0, win=0, avg_win=0, avg_loss=0, exp_R=0, pf=0, t=0, ret=0, dd=0, per_year=0)
    R = tr[:, 3]
    ret = R * tr[:, 4]
    wins, losses = R[R > 0], R[R <= 0]
    eq = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    sd = R.std(ddof=1) if n > 1 else 0
    return dict(
        n=n,
        win=len(wins) / n,
        avg_win=wins.mean() if len(wins) else 0,
        avg_loss=losses.mean() if len(losses) else 0,
        exp_R=R.mean(),
        pf=wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        t=R.mean() / (sd / np.sqrt(n)) if sd > 0 else 0,
        ret=eq[-1] - 1,
        dd=((eq - peak) / peak).min(),
        per_year=n / years if years else 0,
    )


def bootstrap(tr: np.ndarray, sims: int = 2000, seed: int = 1) -> dict:
    """Resample trades with replacement -> how lucky/unlucky could this edge be?"""
    if len(tr) < 20:
        return dict(p_profit=0.0, ret_p5=0.0, ret_med=0.0, dd_p95=0.0)
    rng = np.random.default_rng(seed)
    ret = tr[:, 3] * tr[:, 4]
    idx = rng.integers(0, len(ret), size=(sims, len(ret)))
    eq = np.cumprod(1 + ret[idx], axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = ((eq - peak) / peak).min(axis=1)
    fin = eq[:, -1] - 1
    return dict(p_profit=(fin > 0).mean(), ret_p5=np.percentile(fin, 5), ret_med=np.median(fin), dd_p95=np.percentile(dd, 5))

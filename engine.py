"""Trade simulator: one position at a time, ATR stop + R-multiple target, realistic costs.

Conservative rules:
* signal on close of bar i -> entry at open of bar i+1 (+ slippage against us)
* if stop and target are both inside one candle, the STOP is assumed hit first
* stops/timeouts pay slippage; gaps through the stop fill at the open
* fees are charged on both entry and exit notional
* risk per trade is a fixed fraction of equity, with a leverage cap
"""
import numpy as np

COLS = ["entry_i", "exit_i", "side", "R", "risk_frac", "reason"]  # reason: 0=SL 1=TP 2=time 3=end


def simulate(o, h, l, c, atr, long_sig, short_sig, start, end, sl_atr, tp_r,
             max_bars=48, fee=0.001, slip=0.0005, allow_short=True, risk=0.01, max_lev=3.0, trail_atr=0.0):
    """Simulate over bars [start, end). Returns an (n, 6) float array of trades (see COLS).
    trail_atr > 0 replaces the fixed target by a chandelier trailing stop (trail_atr * ATR from the extreme);
    the stop only moves with information up to the previous bar, so no look-ahead."""
    trades = []
    sig_idx = np.flatnonzero(long_sig[start:end - 1] | (short_sig[start:end - 1] if allow_short else False)) + start
    next_free = start
    for i in sig_idx:
        if i < next_free or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        side = 1 if long_sig[i] else -1
        if side == -1 and not allow_short:
            continue
        entry = o[i + 1] * (1 + slip * side)
        dist = sl_atr * atr[i]
        stop = entry - side * dist
        tp = entry + side * dist * tp_r if trail_atr <= 0 else np.inf * side
        tdist = trail_atr * atr[i]
        last = min(i + max_bars, end - 1)
        exit_px, reason, j = None, 3, last
        for j in range(i + 1, last + 1):
            if side == 1:
                if o[j] <= stop or l[j] <= stop:
                    exit_px, reason = min(o[j], stop) * (1 - slip), 0
                elif h[j] >= tp:
                    exit_px, reason = max(o[j], tp) if j > i + 1 else tp, 1
                elif trail_atr > 0:
                    stop = max(stop, h[j] - tdist)
            else:
                if o[j] >= stop or h[j] >= stop:
                    exit_px, reason = max(o[j], stop) * (1 + slip), 0
                elif l[j] <= tp:
                    exit_px, reason = min(o[j], tp) if j > i + 1 else tp, 1
                elif trail_atr > 0:
                    stop = min(stop, l[j] + tdist)
            if exit_px is not None:
                break
        if exit_px is None:
            exit_px = c[j] * (1 - slip * side)
            reason = 2 if j == i + max_bars else 3
        gross = side * (exit_px - entry) / dist
        net = gross - fee * (entry + exit_px) / dist
        eff_risk = min(risk, max_lev * dist / entry)
        trades.append((i + 1, j, side, net, eff_risk, reason))
        next_free = j  # a new signal may fire on the exit bar's close
    return np.array(trades, dtype=float).reshape(-1, 6)

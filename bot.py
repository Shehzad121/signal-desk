"""Live signal bot. Run it right after each 4h candle closes (GitHub Actions does this, see .github/workflows).

    python bot.py                 # live: fetch candles, update open signals, send new ones to Telegram
    python bot.py --dry-run       # same, but nothing is sent to Telegram
    python bot.py --offline --asof 2025-03-01T00:00:00Z --state tmp.json   # replay on cached data

It NEVER trades. It only posts signals (entry / stop / target) and tracks how they would have ended,
using exactly the same rules as the backtest (stop checked first, gap fills, fees + slippage).
Telegram: set env TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (never commit them).
"""
import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

import data_loader
from strategies import adx, atr, donchian, ema

ROOT = Path(__file__).parent
TF_MIN = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


# ----------------------------------------------------------------------------- config / data
def load_config():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def enabled_coins(cfg):
    out = []
    for gname, g in cfg["groups"].items():
        if g.get("enabled"):
            out += [(c, gname, float(g.get("risk_multiplier", 1.0))) for c in g["coins"]]
    return out


_OFFLINE = {}


def get_candles(coin, cfg, asof, offline, tf=None):
    sym, tf = f"{coin}/USDT", tf or cfg["timeframe"]
    if offline:
        if (coin, tf) not in _OFFLINE:
            src = {"30m": ("15m", "30min"), "2h": ("1h", "2h")}.get(tf)
            if src:
                d = data_loader.load(sym, src[0], refresh=False)
                _OFFLINE[(coin, tf)] = d.resample(src[1]).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                                                 "volume": "sum"}).dropna()
            else:
                _OFFLINE[(coin, tf)] = data_loader.load(sym, tf, refresh=False)
        df, ex = _OFFLINE[(coin, tf)], "cache"
    else:
        df, ex = data_loader.fetch_recent(sym, tf, 700 if tf != "15m" else 400)
    if asof is not None:
        df = df[df.index + pd.Timedelta(minutes=TF_MIN[tf]) <= asof]
        df = df.iloc[-800:]
    return df, ex


# ----------------------------------------------------------------------------- trade tracking
def evaluate(sig, df, cfg):
    """Replay a signal over the candles after it. Same rules as engine.simulate (incl. trailing stop).
    Returns updates dict."""
    side, tp = sig["side"], sig.get("tp")
    trail = bool(sig.get("trail_dist"))
    stop = sig.get("stop0", sig["stop"]) if trail else sig["stop"]
    tdist = sig.get("trail_dist") or 0.0
    entry, dist = sig["entry_eff"], sig["dist"]
    fee, slip = cfg["costs"]["fee"], cfg["costs"]["slippage"]
    max_bars = int(cfg["strategy"]["max_hold_days"] * 1440 / TF_MIN[sig.get("tf", cfg["timeframe"])])
    after = df[df.index > pd.Timestamp(sig["signal_candle"])]
    exit_px = reason = exit_ts = None
    k = 0
    for k, (ts, r) in enumerate(after.iloc[:max_bars].iterrows(), start=1):
        o, h, l, c = r["open"], r["high"], r["low"], r["close"]
        if side == 1:
            if o <= stop or l <= stop:
                exit_px, reason = min(o, stop) * (1 - slip), "stop"
            elif tp is not None and h >= tp:
                exit_px, reason = (max(o, tp) if k > 1 else tp), "target"
            elif trail:
                stop = max(stop, h - tdist)
        else:
            if o >= stop or h >= stop:
                exit_px, reason = max(o, stop) * (1 + slip), "stop"
            elif tp is not None and l <= tp:
                exit_px, reason = (min(o, tp) if k > 1 else tp), "target"
            elif trail:
                stop = min(stop, l + tdist)
        if exit_px is None and k == max_bars:
            exit_px, reason = c * (1 - slip * side), "time"
        if exit_px is not None:
            exit_ts = ts
            break
    upd = {"bars_held": k}
    if trail:
        upd["stop"] = float(stop)                      # current trailing stop level
    if exit_px is None:
        upd["last_price"] = float(after["close"].iloc[-1]) if len(after) else sig["entry"]
        return upd
    if reason == "stop" and trail and ((side == 1 and stop > entry) or (side == -1 and stop < entry)):
        reason = "trail"                               # trailing stop hit while in profit
    gross = side * (exit_px - entry) / dist
    upd.update(status="closed", exit_price=float(exit_px), exit_reason=reason, exit_time=exit_ts.isoformat(),
               R=round(float(gross - fee * (entry + exit_px) / dist), 3))
    return upd


def _build(coin, group, mult, side, close, a, cand, tf, mode, cfg, open_risk):
    st, rk = cfg["strategy"], cfg["risk"]
    entry_eff = close * (1 + cfg["costs"]["slippage"] * side)
    dist = st["sl_atr"] * a
    eff = min(rk["per_trade_pct"] / 100 * mult, rk["max_leverage"] * dist / entry_eff)
    if open_risk + eff * 100 > rk["max_open_risk_pct"] + 1e-9:
        return None, f"{coin} {'LONG' if side == 1 else 'SHORT'} skipped (open-risk cap {rk['max_open_risk_pct']}%)"
    end = cand + pd.Timedelta(minutes=TF_MIN[tf])
    trail_atr = st.get("trail_atr", 0) or 0
    stop0 = entry_eff - side * dist
    sig = dict(
        id=f"{coin}-{cand:%Y%m%d%H%M}-{tf}", symbol=f"{coin}/USDT", coin=coin, group=group, tf=tf, mode=mode,
        stream=cfg.get("stream", tf),
        side=side, status="open", signal_candle=cand.isoformat(), signal_time=end.isoformat(),
        entry=close, entry_eff=entry_eff, stop=stop0, dist=dist, risk_pct=round(eff * 100, 3),
        size_pct=round(eff / (dist / entry_eff) * 100, 1), bars_held=0, last_price=close,
        deadline=(end + pd.Timedelta(days=st["max_hold_days"])).isoformat(),
    )
    if trail_atr:      # no fixed target: a trailing stop follows price and locks in profit
        sig.update(tp=None, tp_r=None, stop0=stop0, trail_atr=trail_atr, trail_dist=trail_atr * a)
    else:
        sig.update(tp=entry_eff + side * dist * st["tp_r"], tp_r=st["tp_r"])
    return sig, None


def new_signal(coin, group, mult, df, cfg, open_risk):
    """Mode 'close': breakout confirmed by the CLOSE of a 4h candle."""
    st = cfg["strategy"]
    long_s, short_s = donchian(df, n=st["n"], adx_min=st.get("adx_min", 0))
    if long_s[-1]:
        side = 1
    elif short_s[-1] and st.get("allow_short", True):
        side = -1
    else:
        return None, None
    return _build(coin, group, mult, side, float(df["close"].iloc[-1]), float(atr(df).iloc[-1]),
                  df.index[-1], cfg["timeframe"], "close", cfg, open_risk)


def early_signal(coin, group, mult, df4, df15, cfg, open_risk):
    """Mode 'early': alert as soon as a 15m candle closes beyond the 4h channel (channel/EMA200/ATR come from
    the last CLOSED 4h candles). Same stop/target rules, but more false breakouts than waiting for the 4h close."""
    st, n = cfg["strategy"], cfg["strategy"]["n"]
    if len(df15) < 3:
        return None, None

    def levels(ts):
        prior = df4[df4.index < ts.floor("4h")]
        if len(prior) < 250:
            return None
        return (float(prior["high"].iloc[-n:].max()), float(prior["low"].iloc[-n:].min()),
                float(ema(prior["close"], 200).iloc[-1]), float(atr(prior).iloc[-1]))

    c, pc = float(df15["close"].iloc[-1]), float(df15["close"].iloc[-2])
    lv, plv = levels(df15.index[-1]), levels(df15.index[-2])
    if not lv or not plv:
        return None, None
    hi, lo, base, a = lv
    if c > hi and pc <= plv[0] and c > base:
        side = 1
    elif c < lo and pc >= plv[1] and c < base and st.get("allow_short", True):
        side = -1
    else:
        return None, None
    return _build(coin, group, mult, side, c, a, df15.index[-1], "15m", "early", cfg, open_risk)


# ----------------------------------------------------------------------------- stats
def compute_stats(signals):
    closed = [s for s in signals if s["status"] == "closed"]
    R = np.array([s["R"] for s in closed], float)
    rk = np.array([s["risk_pct"] / 100 for s in closed], float)
    if not len(R):
        return dict(closed=0, open=sum(s["status"] == "open" for s in signals))
    eq = np.cumprod(1 + R * rk)
    peak = np.maximum.accumulate(eq)
    wins, losses = R[R > 0], R[R <= 0]
    return dict(
        closed=len(R), open=sum(s["status"] == "open" for s in signals), win_rate=round(float((R > 0).mean()), 3),
        expectancy_R=round(float(R.mean()), 3), total_R=round(float(R.sum()), 2),
        profit_factor=round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        equity_return_pct=round(float((eq[-1] - 1) * 100), 2), max_drawdown_pct=round(float(((eq - peak) / peak).min() * 100), 2),
    )


# ----------------------------------------------------------------------------- telegram
def fmt(p):
    return f"{p:,.2f}" if p >= 100 else f"{p:,.4f}" if p >= 1 else f"{p:.6g}"


def send(text, dry):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    print("\n--- telegram ---\n" + html.unescape(text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")))
    if dry or not token or not chat:
        if not dry:
            print("(TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set -> not sent)")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=20)
        if not r.ok:
            print("telegram error:", r.status_code, r.text[:200])
    except requests.RequestException as e:
        print("telegram error:", e)


def msg_new(s, cfg):
    long_ = s["side"] == 1
    d, icon = ("LONG (BUY)", "🟢") if long_ else ("SHORT (SELL)", "🔴")
    stop0 = s.get("stop0", s["stop"])
    sp = (stop0 / s["entry"] - 1) * 100
    dl = pd.Timestamp(s["deadline"]).strftime("%d %b %H:%M UTC")
    head = (f"{icon} <b>NAYA SIGNAL [{s.get('stream', s.get('tf'))}]: {d} {html.escape(s['symbol'])}</b>\n\n"
            f"1️⃣ <b>Entry:</b> abhi market price par {'buy' if long_ else 'sell/short'} karo (≈ <code>{fmt(s['entry'])}</code>)\n"
            f"2️⃣ <b>Stop-loss</b> foran lagao: <code>{fmt(stop0)}</code> ({sp:+.1f}%)\n")
    if s.get("trail_dist"):
        rate = max(0.1, min(10.0, s["trail_dist"] / s["entry"] * 100))
        mid = (f"3️⃣ <b>Target nahi hai - Trailing stop:</b> exchange par <i>Trailing Stop</i> order lagao, callback rate ≈ <b>{rate:.1f}%</b>. "
               f"Ye price ke saath saath chalta hai aur profit lock karta hai; price palatne par khud band.\n"
               f"4️⃣ <b>Deadline:</b> {dl} tak band na ho to position khud band kar do\n\n")
    else:
        tp_ = (s["tp"] / s["entry"] - 1) * 100
        mid = (f"3️⃣ <b>Take-profit</b> lagao: <code>{fmt(s['tp'])}</code> ({tp_:+.1f}%)\n"
               f"4️⃣ <b>Deadline:</b> {dl} tak stop ya target na lage to position khud band kar do\n\n")
    tail = (f"💰 Risk: account ka {s['risk_pct']}% → position ≈ {s['size_pct']}% of account ({s['size_pct'] / 100:.2f}x)\n"
            f"Bot khud trade nahi karta.")
    early = ("\n\n⚡ <b>JALDI ALERT:</b> 4h candle abhi band nahi hui, isliye false signal ka chance thora zyada hai "
             "(backtest mein is ka edge ~30% kam tha)." if s.get("mode") == "early" else "")
    return head + mid + tail + early


def msg_closed(s):
    icon = {"target": "✅", "stop": "❌", "trail": "🔒", "time": "⏱"}[s["exit_reason"]]
    d = "LONG" if s["side"] == 1 else "SHORT"
    why = {"target": "Target lag gaya (profit)", "stop": "Stop-loss lag gaya (nuksan)",
           "trail": "Trailing stop lag gaya (profit lock hua)",
           "time": "Deadline poori - agar position khuli hai to ab band kar do"}[s["exit_reason"]]
    return (f"{icon} <b>[{s.get('stream', s.get('tf'))}] {d} {html.escape(s['symbol'])} band</b>\n{why}\n"
            f"Exit ≈ <code>{fmt(s['exit_price'])}</code> · natija <b>{s['R']:+.2f}R</b> "
            f"(1R = jitna risk liya tha)")


def msg_trail(s):
    d = "LONG" if s["side"] == 1 else "SHORT"
    return (f"🔒 <b>{d} {html.escape(s['symbol'])}: trailing stop badla</b>\n"
            f"Naya stop ≈ <code>{fmt(s['stop'])}</code>. Agar tum ne exchange ka Trailing Stop order lagaya hai to kuch nahi karna; "
            f"warna manual stop yahan le aao.")


def msg_summary(state):
    st = state["stats"]
    openers = [s for s in state["signals"] if s["status"] == "open"]
    lines = ["📊 <b>Daily summary</b>", f"Open signals: {len(openers)}"]
    lines += [f"• {'LONG' if s['side'] == 1 else 'SHORT'} {s['symbol']} entry {fmt(s['entry'])}" for s in openers]
    if st.get("closed"):
        lines.append(f"Closed: {st['closed']} · win {st['win_rate']:.0%} · expectancy {st['expectancy_R']:+.2f}R · "
                     f"total {st['total_R']:+.1f}R · maxDD {st['max_drawdown_pct']}%")
    else:
        lines.append("No closed signals yet (forward test just started).")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def get_streams(cfg):
    """Each stream = one timeframe with its own strategy settings, risk share and track record."""
    base = [dict(name=cfg["timeframe"], timeframe=cfg["timeframe"], group_overrides=True, early=True)]
    return [{"enabled": True, "risk_multiplier": 1.0, "strategy": {}, **st} for st in (cfg.get("streams") or base)
            if st.get("enabled", True)]


def stream_of(sig):
    return sig.get("stream") or ("4h" if sig.get("tf") in ("4h", "15m") else sig.get("tf"))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # emoji in messages on Windows consoles
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="do not send to Telegram")
    ap.add_argument("--offline", action="store_true", help="use cached CSV candles (testing)")
    ap.add_argument("--asof", help="pretend the time is this UTC timestamp (testing)")
    ap.add_argument("--state", default=str(ROOT / "docs" / "signals.json"))
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--run-type", choices=["close", "early"], default="close",
                    help="close = hourly run (checks every stream whose candle just closed); early = every ~15 min (only for signal_mode early/both)")
    args = ap.parse_args()

    cfg = load_config()
    now = pd.Timestamp(args.asof, tz="UTC") if args.asof else pd.Timestamp.now(tz="UTC")
    asof = pd.Timestamp(args.asof, tz="UTC") if args.asof else None
    path = Path(args.state)
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"signals": []}
    first_run = not state.get("welcomed", False)   # Telegram "bot started" message goes out once
    signals = state["signals"]
    radar_by = {}
    coins = enabled_coins(cfg)
    streams = get_streams(cfg)
    run = dict(time=now.isoformat(), coins_checked=[], errors=[], new=[], closed=[], notes=[], exchange=None)
    msgs = []

    if first_run:
        msgs.append("🤖 <b>Signal bot started</b>\nWatching: " + ", ".join(c for c, _, _ in coins) +
                    "\nStreams: " + " · ".join(f"{s['name']} (Donchian {({**cfg['strategy'], **s['strategy']})['n']})" for s in streams) +
                    "\nSignals only, no auto-trading.")

    mode = cfg.get("signal_mode", "close")
    if mode not in ("close", "early", "both"):
        raise SystemExit(f"config.yaml: signal_mode must be close, early or both (got {mode!r})")
    if args.run_type == "early" and mode == "close":
        print("signal_mode is 'close' - the 15-minute run has nothing to do.")
        return

    for st in streams:
        tf, sname = st["timeframe"], st["name"]
        is_main = bool(st.get("early"))                        # only the main (4h) stream has the 15-minute early alerts
        check_close = args.run_type == "close" and (mode in ("close", "both") if is_main else True)
        check_early = is_main and mode in ("early", "both")
        if args.run_type == "early" and not check_early:
            continue
        fresh_lim = pd.Timedelta(minutes=max(45, min(120, 0.6 * TF_MIN[tf])))   # a late run must not emit a stale entry
        radar = radar_by.setdefault(sname, {})
        for coin, group, gmult in coins:
            try:
                ccfg = {**cfg, "timeframe": tf, "stream": sname, "strategy": {**cfg["strategy"], **st["strategy"]}}
                if st.get("group_overrides"):   # a group may override settings (e.g. BNB: trailing stop) on the main stream
                    ccfg["strategy"] = {**ccfg["strategy"], **cfg["groups"][group].get("strategy", {})}
                mult = gmult * float(st["risk_multiplier"])
                df, ex = get_candles(coin, ccfg, asof, args.offline)
                run["exchange"] = ex
                if len(df) < 300:
                    raise RuntimeError(f"not enough {tf} candles")
                if coin not in run["coins_checked"]:
                    run["coins_checked"].append(coin)
                mine = [x for x in signals if x["status"] == "open" and x["coin"] == coin and stream_of(x) == sname]
                df15 = get_candles(coin, ccfg, asof, args.offline, "15m")[0] if (check_early or any(x.get("tf") == "15m" for x in mine)) else None
                cs = ccfg["strategy"]
                n = cs["n"]                            # "radar": how far is each coin from triggering a signal?
                ema200 = float(df["close"].ewm(span=200, adjust=False).mean().iloc[-1])
                radar[coin] = dict(price=float(df["close"].iloc[-1]), ema200=ema200, trend="up" if df["close"].iloc[-1] > ema200 else "down",
                                   long_trigger=float(df["high"].iloc[-n:].max()), short_trigger=float(df["low"].iloc[-n:].min()),
                                   atr=float(atr(df).iloc[-1]), adx=float(adx(df).iloc[-1]), adx_min=cs.get("adx_min", 0),
                                   long_only=not cs.get("allow_short", True), sl_atr=cs["sl_atr"], tp_r=cs.get("tp_r"), trail_atr=cs.get("trail_atr", 0))
                for s_ in mine:                        # 1) update this stream's open signals of this coin
                    s_.update(evaluate(s_, df15 if s_.get("tf") == "15m" else df, ccfg))
                    if s_["status"] == "closed":
                        run["closed"].append(s_["id"])
                        msgs.append(msg_closed(s_))
                    elif s_.get("trail_dist"):
                        step = s_["trail_dist"] / s_["trail_atr"]          # 1 ATR at signal time
                        if abs(s_["stop"] - s_.get("notified_stop", s_["stop0"])) >= step:
                            s_["notified_stop"] = s_["stop"]
                            msgs.append(msg_trail(s_))
                if any(x["status"] == "open" and x["coin"] == coin and stream_of(x) == sname for x in signals):
                    continue
                open_risk = sum(x["risk_pct"] for x in signals if x["status"] == "open")
                fresh = (now - (df.index[-1] + pd.Timedelta(minutes=TF_MIN[tf]))) <= fresh_lim
                sig, note = (new_signal(coin, group, mult, df, ccfg, open_risk) if (check_close and fresh) else (None, None))   # 2) new signal?
                if sig is None and note is None and check_early:
                    sig, note = early_signal(coin, group, mult, df, df15, ccfg, open_risk)
                if note:
                    run["notes"].append(note)
                if sig and not any(x["id"] == sig["id"] for x in signals):
                    signals.append(sig)
                    run["new"].append(sig["id"])
                    msgs.append(msg_new(sig, ccfg))
            except Exception as e:
                run["errors"].append(f"{sname}/{coin}: {e}")

    main_name = next((s["name"] for s in streams if s.get("early")), streams[0]["name"])
    state.update(
        updated=now.isoformat(), last_run=run, signals=signals, stats=compute_stats(signals),
        radar=radar_by.get(main_name) or state.get("radar", {}),
        radar_by_stream={**state.get("radar_by_stream", {}), **radar_by},
        stats_by_stream={s["name"]: compute_stats([x for x in signals if stream_of(x) == s["name"]]) for s in streams},
        stats_by_mode={m: compute_stats([x for x in signals if x.get('mode', 'close') == m]) for m in ('close', 'early')},
        welcomed=state.get("welcomed", False) or (first_run and not args.dry_run and bool(os.environ.get("TELEGRAM_BOT_TOKEN"))),
        config=dict(timeframe=cfg["timeframe"], signal_mode=mode, strategy=cfg["strategy"], risk=cfg["risk"],
                    streams=[dict(name=s["name"], timeframe=s["timeframe"], risk_multiplier=s["risk_multiplier"],
                                  strategy={**cfg["strategy"], **s["strategy"]}, ref=s.get("ref"), main=bool(s.get("early"))) for s in streams],
                    groups={g: dict(enabled=v["enabled"], coins=v["coins"], **({"strategy": v["strategy"]} if v.get("strategy") else {}))
                            for g, v in cfg["groups"].items()},
                    backtest_reference=cfg["backtest_reference"]),
    )
    if now.hour == 0 and args.run_type == "close" and not args.no_summary and not first_run:
        msgs.append(msg_summary(state))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    for m in msgs:
        send(m, args.dry_run)
    print(f"\n[{now:%Y-%m-%d %H:%M}] checked={len(run['coins_checked'])} new={len(run['new'])} "
          f"closed={len(run['closed'])} errors={run['errors']} notes={run['notes']}")
    if run["errors"] and not run["coins_checked"]:
        raise SystemExit(1)  # make the GitHub Action fail visibly if nothing worked


if __name__ == "__main__":
    main()

# Crypto signal bot (signals only, no auto-trading)

4h Donchian breakout + EMA200 trend filter on liquid coins. Posts entry / stop / target to Telegram and tracks
every signal on a dashboard. Settings (coins on/off, risk) live in `config.yaml`.

## Files
- `run_backtest.py` – walk-forward backtester (strategies in `strategies.py`, simulator in `engine.py`)
- `bot.py` – live bot (what GitHub Actions runs every 4h)
- `docs/index.html` + `docs/signals.json` – dashboard (GitHub Pages serves `docs/`)

## Signal mode (config.yaml -> `signal_mode`)
- `close` (default): signal after a 4h candle CLOSES beyond the channel. Best edge in the backtest (+0.14R/trade), ~5 signals/week.
- `early`: alert within ~15 min of price crossing the channel (+0.10R/trade, more false alarms, signals ~2h earlier).
- `both`: both kinds (early signals are marked with a lightning badge).
For `early`/`both` the 15-minute workflow `.github/workflows/early.yml` must stay enabled (Actions tab -> signal-bot-early).
If you only use `close`, DISABLE that workflow (Actions -> signal-bot-early -> ... -> Disable workflow) to save Actions minutes.
Every 15 minutes = ~2900 runner-minutes/month: free only on a PUBLIC repo (private repos get 2000 free min/month).
To change a setting later: GitHub -> your repo -> config.yaml -> pencil icon -> edit -> Commit. The next run uses it.

## Try it locally (no Telegram needed)
    pip install -r requirements.txt
    python bot.py --dry-run
    python -m http.server 8000 -d docs     # open http://localhost:8000

## Go live (free)
1. Telegram: talk to @BotFather -> /newbot -> copy the token. Send any message to your new bot, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id`.
2. Create a GitHub repo, push this folder.
3. Repo -> Settings -> Secrets and variables -> Actions -> add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
4. Repo -> Settings -> Pages -> Source: "Deploy from a branch", branch `main`, folder `/docs`.
5. Actions tab -> "signal-bot" -> Run workflow once to test. After that it runs by itself every 4h.

Note: GitHub runners are in the US and Binance blocks US IPs, so the bot automatically falls back to OKX / Bybit
candles (`EXCHANGES` in `data_loader.py`). Prices are nearly identical but not byte-for-byte equal to Binance.

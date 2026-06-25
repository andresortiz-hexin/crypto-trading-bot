# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Bot

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full stack (health monitor supervises bot.py)
python health_monitor.py

# Run bot directly (no watchdog)
python bot.py
```

The `Procfile` entry `worker: python health_monitor.py` is what Railway executes. `health_monitor.py` spawns `bot.py` as a subprocess and auto-restarts it on crash or if it goes silent for >5 minutes (up to 15 restarts).

There is no test suite or linter configuration in this repo.

## Architecture

This is a Python 3.13 automated trading bot using the Alpaca Markets API, deployed on Railway (auto-deploy from GitHub `main`).

### Two Parallel Trading Systems

The bot runs two trading strategies concurrently in each `trade_cycle()`:

**V2 – Tactical (signal-based):** `bot.py` → `SignalEngine` + `RegimeEngine` + `RiskEngine` + `intelligence.py`
- Evaluates individual symbols on each cycle and places trades based on multi-factor signal scores
- Operates on 1-minute and 1-hour OHLCV bars from Alpaca
- Crypto traded 24/7; stocks traded only during UTC 14:00–20:00 (approximately US market hours)

**V3 – Strategic (allocation-based):** `bot.py` → `MomentumEngine` + `AllocationEngine` + `ExecutionEngine` + `universe.py`
- Runs every 30 minutes (`REBALANCE_INTERVAL = 900s`) via `run_v3_rebalance()`
- Computes portfolio-level target allocations across a 19-instrument universe, then executes rebalancing trades (sells first, then buys)
- Uses daily bars for momentum scoring across 4 asset classes: equity, fixed_income, commodity, crypto

### Module Responsibilities

| Module | Role |
|---|---|
| `bot.py` | Orchestrator: main loop, data fetching, order placement, Telegram alerts |
| `signal_engine.py` | Scores assets 0–100 via trend, momentum, RSI/StochRSI, volume, ATR, MACD, multi-timeframe bonus |
| `regime_engine.py` | Classifies market as `uptrend`/`sideways`/`stress` using SPY hourly bars; gates which assets can trade |
| `risk_engine.py` | Kill switch, daily/weekly loss limits, position sizing, per-class exposure caps, stop cooldowns |
| `intelligence.py` | Sentiment overlay from Fear&Greed (Alternative.me) + CoinGecko price/community data |
| `self_learner.py` | Analytics only (no parameter mutation): trade logging, Sharpe, win rate, benchmark comparison |
| `universe.py` | 19-instrument universe definition with strategic weights and per-class position limits |
| `momentum_engine.py` | Composite momentum (1W/1M/3M/6M/12M weighted) + relative ranking within asset class |
| `allocation_engine.py` | SAA base → regime overlay → momentum filter → volatility targeting → position sizing |
| `execution_engine.py` | Order routing for rebalance trades; closes positions for sells before opening buys |
| `health_monitor.py` | Watchdog process: supervises `bot.py` subprocess, auto-restarts, sends Telegram alerts |

### State and Data Flow

- Alpaca bar data returns DataFrames with MultiIndex columns/index — always normalize with `normalize_df()` in `bot.py` before passing to engines
- `SelfLearner` persists state to `/tmp/learner_state.json`; heartbeat/health written to `/tmp/bot_heartbeat` and `/tmp/bot_health.json`
- Intelligence results are cached in `intel_cache` (module-level dict in `bot.py`) for 5 minutes to avoid CoinGecko rate limits; `intelligence.py` also has its own `_sentiment_cache` with a 10-minute TTL

### Critical Naming Conventions

These have caused bugs before; get them exactly right:

- **Regime names:** `uptrend`, `sideways`, `stress` — never `bull`, `bear`, or `risk_on/off`
- **Signal engine method:** `compute_score(bars, bars_hourly=None)` — not `generate_signal()`
- **Regime engine method:** `classify(bars_daily, symbol='QQQ')` — not `detect()` or `update()`
- **RiskEngine constructor:** accepts `**kwargs` only (e.g. `RiskEngine(max_trades_per_day=15, daily_loss_limit_pct=0.025, ...)`)
- **Crypto symbol format:** `BTC/USD` everywhere in Python; strip the slash (`BTCUSD`) only when calling Alpaca API (`symbol.replace('/', '')`)
- **`record_trade()` in RiskEngine:** signature is `(symbol=None, side=None, pnl_pct=None)` — all optional kwargs

### Risk Parameters (configured in `bot.py`)

- Kill switch: triggers at −1.5% daily or −4% weekly loss
- Max position: 25% core ETFs, 20% crypto, 15% leveraged ETFs
- Max total exposure: 90% of portfolio
- Cooldown after stop-loss: symbol blocked until `clear_cooldown()` is called
- V2 signal buy threshold: score ≥ 50 (per `SignalEngine.BUY_THRESHOLD`)
- V3 rebalance drift threshold: 1% weight difference (in `AllocationEngine.compute_rebalance_trades`)

### Environment Variables

| Variable | Description |
|---|---|
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_SECRET_KEY` | Alpaca secret key |
| `ALPACA_PAPER` | `true` for paper trading (default: true) |
| `TELEGRAM_TOKEN` | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | Telegram chat ID |

### External APIs

- **Alpaca:** Trading orders + historical OHLCV bars (crypto and stocks)
- **CoinGecko:** Price changes, community sentiment, global market data (free tier — respect rate limits via `_cached_get()`)
- **Alternative.me:** Fear & Greed index
- **Yahoo Finance:** Benchmark index data in `self_learner.py`
- **Telegram:** Real-time trade notifications, regime changes, error alerts

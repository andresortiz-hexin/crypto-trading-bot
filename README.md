# crypto-trading-bot V3 Institutional

Automated ETF/crypto trading bot using Alpaca Markets API with institutional-grade risk management.

## V3 Architecture (April 2026)

Complete rewrite from v4 based on BlackRock/JPM-style audit. Key changes:

### Modules

| File | Purpose |
|---|---|
| `bot.py` | Main orchestrator - integrates all V3 modules |
| `signal_engine.py` | Multi-factor signal scoring (RSI, MACD, EMA, BB, VWAP, Volume, ATR) |
| `regime_engine.py` | Market regime detection (uptrend/sideways/stress) using SPY data |
| `risk_engine.py` | Position sizing, kill switch, daily/weekly loss limits, exposure caps |
| `intelligence.py` | Weighted sentiment from Fear&Greed, CryptoPanic, CoinGecko with caching |
| `self_learner.py` | Analytics and reporting (no longer mutates params in production) |
| `universe.py` | Multi-asset universe (19 instruments across 6 asset classes) |
| `momentum_engine.py` | Dual momentum (absolute + relative) with composite scoring |
| `allocation_engine.py` | Volatility targeting, regime overlays, risk budgeting |
| `execution_engine.py` | Smart order execution with rebalance support |

### Key Improvements over v4

1. **Conservative Targets**: 0.5-1% daily (was 3% - unrealistic)
3. **No Force Entry**: Only trades on confirmed multi-factor signals
4. **Risk Engine**: Kill switch activates on 1.5% daily loss or 4% weekly loss
5. **Regime-Aware**: Adjusts position sizes and thresholds based on uptrend/sideways/stress
6. **Position Sizing**: Max 25% per position, 70% total exposure, volatility-adjusted
7. **Dynamic Stops**: ATR-based stop-loss and take-profit that adapt to regime
8. **Removed TQQQ**: No leveraged ETFs (excessive risk)
9. **Sentiment V3**: Weighted keywords, recency decay, contrarian signals at extremes

### Risk Parameters

- Max trades per day: 12
- Daily loss limit: -1.5%
- Weekly loss limit: -4%
- Max position size: 25% of portfolio
- Max total exposure: 70%
- Max crypto exposure: 25%
- Cooldown after stop-loss: 30 bars

### Regime States

| Regime | Allowed Assets | Min Signal Score | Max Trades/Day | Description |
|---|---|---|---|---|
| **Uptrend** | QQQ, SPY, TQQQ, BTC/USD | 55 | 6 | Favorable - tactical leverage allowed |
| **Sideways** | QQQ, SPY, BTC/USD | 65 | 4 | Conservative mode, no leverage |
| **Stress** | SPY only | 80 | 2 | Minimal exposure, cash preferred |

### Signal Engine Scoring (0-100)

- Trend (25 pts): EMA 9/21/50 alignment + slope
- Momentum (25 pts): 20-bar and 5-bar returns, relative strength
- RSI reversal (20 pts): RSI recovering from oversold in trend direction
- Volume (15 pts): Current volume vs 20-day average
- Volatility (15 pts): ATR within acceptable range
- Multi-timeframe bonus (+10 pts): Hourly confirmation

### API Integration

- **Alpaca Markets**: Trading + historical data (crypto 24/7, stocks market hours)
- **CoinGecko**: Market data, 24h/7d changes
- **CryptoPanic**: News sentiment
- **Alternative.me**: Fear & Greed index
- **Telegram**: Real-time trade notifications and alerts

### Deployment

- **Platform**: Railway (auto-deploy from GitHub main branch)
- **Runtime**: Python 3.13
- **Region**: us-west2

### Environment Variables

| Variable | Description |
|---|---|
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_SECRET_KEY` | Alpaca secret key |
| `ALPACA_PAPER` | `true` for paper trading |
| `TELEGRAM_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Telegram chat ID for alerts |

### Changelog

**V3.1 (April 3, 2026)** - Critical bug fixes:
- Fixed `RiskEngine.__init__()` to accept kwargs for configurable parameters
- Fixed `bot.py` to use correct API signatures (`compute_score` instead of `generate_signal`)
- Fixed regime name mismatch (uptrend/sideways/stress, not bull/bear/sideways)
- Added regime-based asset filtering and minimum signal score checks
- Added `record_stop_loss` calls for proper cooldown tracking

**V3.0 (April 3, 2026)** - Full institutional rewrite:
- New modular architecture (SignalEngine, RegimeEngine, RiskEngine, Intelligence, SelfLearner)
- BlackRock/JPM-style risk management
- Conservative targets and position sizing
- Multi-factor signal scoring replacing simple RSI/MACD


## V3 Upgrade (April 2026)

### New Modules
- **universe.py**: 19-instrument multi-asset universe (crypto, US equities, intl, fixed income, commodities)
- **momentum_engine.py**: Dual momentum (1M/3M/6M/12M composite) with absolute + relative scoring
- **allocation_engine.py**: Strategic Asset Allocation with regime tactical overlay, volatility targeting
- **execution_engine.py**: Smart rebalance execution (sells first, then buys)

### V3 Key Changes
1. **Multi-Asset Diversification**: Expanded from 5 symbols to 19-instrument universe across 6 asset classes
2. **Allocation-Based Trading**: Portfolio-level allocation targets replace individual trade decisions
3. **Volatility Targeting**: 12% annualized target with dynamic scaling (0.2x-1.0x)
4. **Regime Tactical Overlay**: Regime multipliers shift allocation between risk-on and risk-off assets
5. **Rebalance Engine**: 30-min interval checks with 5% drift threshold
6. **Cash Management**: Regime-based cash floors (5% uptrend, 20% sideways, 40% stress)
7. **Position Limits**: Max 15% single position, class-level caps

### V2.1 Hotfix (April 2026)
- Fixed RiskEngine constructor kwargs
- Fixed API calls: compute_score(), classify()
- Fixed regime names: uptrend/sideways/stress
- Fixed record_trade() argument mismatch

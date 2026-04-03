# crypto-trading-bot V2 Institutional

Automated ETF/crypto trading bot using Alpaca Markets API with institutional-grade risk management.

## V2 Architecture (April 2026)

Complete rewrite from v4 based on BlackRock/JPM-style audit. Key changes:

### Modules

| File | Purpose |
|---|---|
| `bot.py` | Main orchestrator - integrates all V2 modules |
| `signal_engine.py` | Multi-factor signal scoring (RSI, MACD, EMA, BB, VWAP, Volume, ATR) |
| `regime_engine.py` | Market regime detection (bull/bear/sideways) using BTC, ETH, SPY correlation |
| `risk_engine.py` | Position sizing, kill switch, daily/weekly loss limits, exposure caps |
| `intelligence.py` | Weighted sentiment from Fear&Greed, CryptoPanic, CoinGecko with caching |
| `self_learner.py` | Analytics and reporting (no longer mutates params in production) |

### Key Improvements over v4

1. **Conservative Targets**: 0.5-1% daily (was 3% - unrealistic)
2. **No Force Entry**: Only trades on confirmed multi-factor signals
3. **Risk Engine**: Kill switch activates on 1.5% daily loss or 4% weekly loss
4. **Regime-Aware**: Adjusts position sizes and thresholds based on bull/bear/sideways
5. **Position Sizing**: Max 25% per position, 70% total exposure, volatility-adjusted
6. **Dynamic Stops**: ATR-based stop-loss and take-profit that adapt to regime
7. **Removed TQQQ**: No leveraged ETFs (excessive risk)
8. **Sentiment V2**: Weighted keywords, recency decay, contrarian signals at extremes

### Risk Parameters

- Max trades per day: 12
- Max position size: 25% of portfolio
- Max total exposure: 70%
- Daily loss limit: 1.5% (kill switch)
- Weekly loss limit: 4% (kill switch)
- Stop loss: 1.2-2.5% (regime-dependent)
- Take profit: 0.8-2.0% (regime-dependent)

### Signal Engine Factors

- RSI (14) with regime-adjusted thresholds
- MACD crossover with histogram momentum
- EMA 9/21 trend alignment
- Bollinger Bands squeeze/breakout
- VWAP deviation
- Volume profile vs 20-period average
- ATR volatility filter
- Multi-timeframe confirmation (1m + 1h)

### Environment Variables

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Deployment

Deployed on Railway. The `Procfile` runs `bot.py` as a worker process.

### Symbols Traded

- **Crypto** (24/7): BTC/USD, ETH/USD, SOL/USD
- **Stocks** (market hours): SPY, QQQ

### Telegram Notifications

- Buy/Sell orders with confidence and regime info
- Stop-loss and take-profit alerts
- Kill switch activation alerts
- Intelligence updates (every 5 min)
- Learning reports (every 15 min) with benchmark comparison

---

*V2 Institutional upgrade - April 2026*

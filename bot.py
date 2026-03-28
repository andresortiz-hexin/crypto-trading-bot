import os
import time
import logging
import requests
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

# ── Configuration ──────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL   = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
TELEGRAM_TOKEN    = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID  = os.environ.get('TELEGRAM_CHAT_ID', '')

DAILY_PROFIT_TARGET = 0.01   # 1% daily target
STOP_LOSS_PCT       = 0.005  # 0.5% stop-loss per trade
MAX_POSITION_PCT    = 0.20   # max 20% of portfolio per symbol
TRADE_INTERVAL      = 60     # seconds between cycles

# Trading universe: crypto (24/7) + liquid ETFs
SYMBOLS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD',
    'SPY', 'QQQ', 'TQQQ', 'SOXL'
]

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
log = logging.getLogger(__name__)

# ── Telegram notifications ──────────────────────────────────────────────────────
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': message}, timeout=10)
    except Exception as e:
        log.warning(f'Telegram error: {e}')

# ── Alpaca client ───────────────────────────────────────────────────────────────
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# ── Helper functions ────────────────────────────────────────────────────────────
def get_account():
    return api.get_account()

def get_portfolio_value():
    account = get_account()
    return float(account.portfolio_value)

def get_daily_pnl_pct():
    account = get_account()
    equity     = float(account.equity)
    last_eq    = float(account.last_equity)
    if last_eq == 0:
        return 0.0
    return (equity - last_eq) / last_eq

def get_bars(symbol: str, timeframe='1Min', limit=100):
    """Fetch OHLCV bars; handles crypto vs equity symbol format."""
    try:
        sym = symbol.replace('/', '')
        bars = api.get_crypto_bars(sym, tradeapi.TimeFrame.Minute, limit=limit).df \
            if '/' in symbol else \
            api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit).df
        bars = bars[['open','high','low','close','volume']]
        return bars.reset_index(drop=True)
    except Exception as e:
        log.error(f'get_bars({symbol}): {e}')
        return None

def compute_signals(bars: pd.DataFrame):
    """Returns (signal, confidence) where signal in {-1, 0, 1}."""
    if bars is None or len(bars) < 30:
        return 0, 0.0
    close = bars['close']
    rsi   = RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_obj = MACD(close)
    macd_line  = macd_obj.macd().iloc[-1]
    macd_sig   = macd_obj.macd_signal().iloc[-1]
    ema20 = EMAIndicator(close, window=20).ema_indicator().iloc[-1]
    ema50 = EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    bb    = BollingerBands(close, window=20)
    bb_lo = bb.bollinger_lband().iloc[-1]
    bb_hi = bb.bollinger_hband().iloc[-1]
    price = close.iloc[-1]

    score = 0
    if rsi < 35:            score += 1
    elif rsi > 65:          score -= 1
    if macd_line > macd_sig: score += 1
    else:                    score -= 1
    if ema20 > ema50:       score += 1
    else:                   score -= 1
    if price < bb_lo:       score += 1
    elif price > bb_hi:     score -= 1

    confidence = abs(score) / 4.0
    signal = 1 if score >= 2 else (-1 if score <= -2 else 0)
    return signal, confidence

def get_position_qty(symbol: str):
    sym = symbol.replace('/', '')
    try:
        pos = api.get_position(sym)
        return float(pos.qty)
    except:
        return 0.0

def place_order(symbol: str, side: str, qty: float):
    sym = symbol.replace('/', '')
    try:
        order = api.submit_order(
            symbol=sym,
            qty=round(qty, 6),
            side=side,
            type='market',
            time_in_force='gtc'
        )
        msg = f'ORDER {side.upper()} {round(qty,4)} {sym} | id={order.id}'
        log.info(msg)
        send_telegram(f'[BOT] {msg}')
        return order
    except Exception as e:
        log.error(f'place_order({symbol},{side},{qty}): {e}')
        return None

def check_stop_losses():
    """Close any position that has exceeded stop-loss."""
    try:
        positions = api.list_positions()
        for pos in positions:
            unrealized_pct = float(pos.unrealized_plpc)
            if unrealized_pct <= -STOP_LOSS_PCT:
                log.warning(f'Stop-loss triggered for {pos.symbol}: {unrealized_pct:.2%}')
                send_telegram(f'[BOT] STOP-LOSS {pos.symbol} {unrealized_pct:.2%}')
                api.close_position(pos.symbol)
    except Exception as e:
        log.error(f'check_stop_losses: {e}')

# ── Main trading loop ───────────────────────────────────────────────────────────
def trade_cycle():
    portfolio = get_portfolio_value()
    daily_pnl = get_daily_pnl_pct()
    log.info(f'Portfolio: ${portfolio:,.2f} | Daily PnL: {daily_pnl:.2%} | Target: {DAILY_PROFIT_TARGET:.2%}')

    # If daily target reached, close all and wait until next day
    if daily_pnl >= DAILY_PROFIT_TARGET:
        log.info('Daily target reached! Closing all positions.')
        send_telegram(f'[BOT] Daily target {DAILY_PROFIT_TARGET:.0%} REACHED! PnL={daily_pnl:.2%} | Closing all.')
        api.close_all_positions()
        return

    check_stop_losses()

    for symbol in SYMBOLS:
        try:
            bars = get_bars(symbol)
            signal, conf = compute_signals(bars)
            current_qty  = get_position_qty(symbol)
            price = bars['close'].iloc[-1] if bars is not None and len(bars) > 0 else None
            if price is None:
                continue

            max_value = portfolio * MAX_POSITION_PCT
            max_qty   = max_value / price

            if signal == 1 and current_qty <= 0 and conf >= 0.5:
                qty = max_qty * conf
                if qty * price >= 1.0:
                    place_order(symbol, 'buy', qty)

            elif signal == -1 and current_qty > 0:
                place_order(symbol, 'sell', current_qty)

        except Exception as e:
            log.error(f'trade_cycle({symbol}): {e}')

def main():
    log.info('=== Crypto/ETF Trading Bot Started ===')
    send_telegram('[BOT] Trading bot started. Target: 1% daily profit.')
    while True:
        try:
            trade_cycle()
        except Exception as e:
            log.error(f'Main loop error: {e}')
            send_telegram(f'[BOT] ERROR: {e}')
        time.sleep(TRADE_INTERVAL)

if __name__ == '__main__':
    main()

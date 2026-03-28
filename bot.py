import os
import time
import logging
import requests
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

# -- Configuration --
ALPACA_API_KEY    = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')
PAPER             = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

DAILY_PROFIT_TARGET = 0.01
STOP_LOSS_PCT       = 0.005
MAX_POSITION_PCT    = 0.20
TRADE_INTERVAL      = 60

CRYPTO_SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD']
STOCK_SYMBOLS  = ['SPY', 'QQQ', 'TQQQ']

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# -- Telegram --
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': message}, timeout=10)
    except Exception as e:
        log.warning(f'Telegram error: {e}')

# -- Clients --
trade_client  = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
crypto_client = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
stock_client  = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def normalize_df(df):
    """Flatten multi-index and normalize column names."""
    if df is None or df.empty:
        return None
    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    # Flatten MultiIndex rows (symbol-level)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    # Lowercase column names
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index(drop=True)
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in required):
        return None
    return df[required]

# -- Helpers --
def get_portfolio_value():
    acct = trade_client.get_account()
    return float(acct.portfolio_value)

def get_daily_pnl_pct():
    acct = trade_client.get_account()
    eq   = float(acct.equity)
    last = float(acct.last_equity)
    return (eq - last) / last if last else 0.0

def get_crypto_bars(symbol: str, limit: int = 100):
    try:
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=limit)
        raw = crypto_client.get_crypto_bars(req).df
        return normalize_df(raw)
    except Exception as e:
        log.error(f'get_crypto_bars({symbol}): {e}')
        return None

def get_stock_bars(symbol: str, limit: int = 100):
    try:
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=limit)
        raw = stock_client.get_stock_bars(req).df
        return normalize_df(raw)
    except Exception as e:
        log.error(f'get_stock_bars({symbol}): {e}')
        return None

def compute_signals(bars):
    if bars is None or len(bars) < 30:
        return 0, 0.0
    close = bars['close']
    rsi   = RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_obj  = MACD(close)
    macd_line = macd_obj.macd().iloc[-1]
    macd_sig  = macd_obj.macd_signal().iloc[-1]
    ema20 = EMAIndicator(close, window=20).ema_indicator().iloc[-1]
    ema50 = EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    bb    = BollingerBands(close, window=20)
    bb_lo = bb.bollinger_lband().iloc[-1]
    bb_hi = bb.bollinger_hband().iloc[-1]
    price = close.iloc[-1]
    score = 0
    if rsi < 35:             score += 1
    elif rsi > 65:           score -= 1
    if macd_line > macd_sig: score += 1
    else:                    score -= 1
    if ema20 > ema50:        score += 1
    else:                    score -= 1
    if price < bb_lo:        score += 1
    elif price > bb_hi:      score -= 1
    confidence = abs(score) / 4.0
    signal = 1 if score >= 2 else (-1 if score <= -2 else 0)
    return signal, confidence

def get_position_qty(symbol: str):
    sym = symbol.replace('/', '')
    try:
        pos = trade_client.get_open_position(sym)
        return float(pos.qty)
    except:
        return 0.0

def place_order(symbol: str, side: OrderSide, qty: float):
    sym = symbol.replace('/', '')
    try:
        req = MarketOrderRequest(
            symbol=sym,
            qty=round(qty, 6),
            side=side,
            time_in_force=TimeInForce.GTC
        )
        order = trade_client.submit_order(req)
        msg   = f'ORDER {side.value.upper()} {round(qty,4)} {sym} id={order.id}'
        log.info(msg)
        send_telegram(f'[BOT] {msg}')
        return order
    except Exception as e:
        log.error(f'place_order({symbol},{side},{qty}): {e}')
        return None

def check_stop_losses():
    try:
        positions = trade_client.get_all_positions()
        for pos in positions:
            pct = float(pos.unrealized_plpc)
            if pct <= -STOP_LOSS_PCT:
                log.warning(f'Stop-loss {pos.symbol}: {pct:.2%}')
                send_telegram(f'[BOT] STOP-LOSS {pos.symbol} {pct:.2%}')
                trade_client.close_position(pos.symbol)
    except Exception as e:
        log.error(f'check_stop_losses: {e}')

def trade_symbol(symbol: str, is_crypto: bool, portfolio: float):
    bars = get_crypto_bars(symbol) if is_crypto else get_stock_bars(symbol)
    if bars is None or len(bars) == 0:
        return
    signal, conf = compute_signals(bars)
    qty_held = get_position_qty(symbol)
    price = bars['close'].iloc[-1]
    max_qty = (portfolio * MAX_POSITION_PCT) / price
    if signal == 1 and qty_held <= 0 and conf >= 0.5:
        qty = max_qty * conf
        if qty * price >= 1.0:
            place_order(symbol, OrderSide.BUY, qty)
    elif signal == -1 and qty_held > 0:
        place_order(symbol, OrderSide.SELL, qty_held)

def trade_cycle():
    portfolio = get_portfolio_value()
    daily_pnl = get_daily_pnl_pct()
    log.info(f'Portfolio: ${portfolio:,.2f} | Daily PnL: {daily_pnl:.2%}')
    if daily_pnl >= DAILY_PROFIT_TARGET:
        log.info('Daily target reached! Closing all.')
        send_telegram(f'[BOT] Daily target REACHED! PnL={daily_pnl:.2%}')
        trade_client.close_all_positions(cancel_orders=True)
        return
    check_stop_losses()
    for sym in CRYPTO_SYMBOLS:
        try:
            trade_symbol(sym, True, portfolio)
        except Exception as e:
            log.error(f'{sym}: {e}')
    for sym in STOCK_SYMBOLS:
        try:
            trade_symbol(sym, False, portfolio)
        except Exception as e:
            log.error(f'{sym}: {e}')

def main():
    log.info('=== Trading Bot Started ===')
    send_telegram('[BOT] Started. Target: 1% daily profit.')
    while True:
        try:
            trade_cycle()
        except Exception as e:
            log.error(f'Main loop: {e}')
            send_telegram(f'[BOT] ERROR: {e}')
        time.sleep(TRADE_INTERVAL)

if __name__ == '__main__':
    main()

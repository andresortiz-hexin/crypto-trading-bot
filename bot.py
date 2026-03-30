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
from ta.volatility import BollingerBands, AverageTrueRange
from intelligence import build_market_intelligence
from self_learner import SelfLearner

# -- Config --
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')
PAPER = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# BASE CONFIG - v4 (self-learning overrides these)
DAILY_PROFIT_TARGET = 0.03
TRADE_INTERVAL = 60
INTEL_INTERVAL = 300
LEARN_INTERVAL = 900  # Run learning cycle every 15 minutes

CRYPTO_SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD']
STOCK_SYMBOLS = ['SPY', 'QQQ', 'TQQQ']

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

# -- Telegram --
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'}, timeout=10)
    except Exception as e:
        log.warning(f'Telegram: {e}')

# -- Clients --
trade_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
crypto_client = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# -- Self-Learning Engine --
learner = SelfLearner()

# -- State --
intel_cache = {}
last_intel_time = 0
last_learn_time = 0
trailing_highs = {}
idle_cycles = 0
cycle_count = 0

def normalize_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index(drop=True)
    req = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in req):
        return None
    return df[req]

def get_portfolio_value():
    return float(trade_client.get_account().portfolio_value)

def get_daily_pnl_pct():
    a = trade_client.get_account()
    eq, last = float(a.equity), float(a.last_equity)
    return (eq - last) / last if last else 0.0

def get_bars(symbol, is_crypto=True, tf=TimeFrame.Minute, limit=100):
    try:
        if is_crypto:
            req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
            raw = crypto_client.get_crypto_bars(req).df
        else:
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
            raw = stock_client.get_stock_bars(req).df
        return normalize_df(raw)
    except Exception as e:
        log.error(f'get_bars({symbol}): {e}')
        return None

def compute_signals(bars):
    p = learner.current_params
    rsi_buy = p.get('rsi_buy', 45)
    rsi_sell = p.get('rsi_sell', 60)
    if bars is None or len(bars) < 30:
        return 0, 0.0, {}
    close = bars['close']
    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_obj = MACD(close)
    macd_line = macd_obj.macd().iloc[-1]
    macd_sig = macd_obj.macd_signal().iloc[-1]
    ema9 = EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema21 = EMAIndicator(close, window=21).ema_indicator().iloc[-1]
    bb = BollingerBands(close, window=20)
    bb_lo = bb.bollinger_lband().iloc[-1]
    bb_hi = bb.bollinger_hband().iloc[-1]
    atr = AverageTrueRange(bars['high'], bars['low'], close, window=14).average_true_range().iloc[-1]
    price = close.iloc[-1]
    volatility = atr / price if price > 0 else 0
    score = 0
    details = {'rsi': rsi, 'macd_bull': macd_line > macd_sig, 'ema_bull': ema9 > ema21, 'price': price, 'volatility': volatility}
    if rsi < rsi_buy: score += 1
    elif rsi > rsi_sell: score -= 1
    if rsi < 30: score += 1
    if macd_line > macd_sig: score += 1
    else: score -= 1
    if ema9 > ema21: score += 1
    else: score -= 1
    if price < bb_lo: score += 1
    elif price > bb_hi: score -= 1
    avg5 = close.tail(5).mean()
    if price > avg5: score += 1
    confidence = min(abs(score) / 5.0, 1.0)
    min_score = p.get('min_signal_score', 1)
    signal = 1 if score >= min_score else (-1 if score <= -2 else 0)
    return signal, confidence, details

def multi_tf_signal(symbol, is_crypto):
    sig1, conf1, d1 = compute_signals(get_bars(symbol, is_crypto, TimeFrame.Minute, 100))
    sig_h, conf_h, dh = compute_signals(get_bars(symbol, is_crypto, TimeFrame.Hour, 100))
    if sig1 == 1 or sig_h == 1:
        return 1, max(conf1, conf_h), d1 if d1.get('price', 0) > 0 else dh
    elif sig1 == -1 and sig_h == -1:
        return -1, max(conf1, conf_h), d1
    elif sig1 == -1 or sig_h == -1:
        return 0, (conf1 + conf_h) / 2.0, d1
    return 0, 0.0, d1

def get_position_qty(symbol):
    sym = symbol.replace('/', '')
    try:
        pos = trade_client.get_open_position(sym)
        return float(pos.qty)
    except:
        return 0.0

def place_order(symbol, side, qty):
    sym = symbol.replace('/', '')
    try:
        req = MarketOrderRequest(symbol=sym, qty=round(qty, 6), side=side, time_in_force=TimeInForce.GTC)
        order = trade_client.submit_order(req)
        log.info(f'ORDER {side.value.upper()} {round(qty,4)} {sym} id={order.id}')
        return order
    except Exception as e:
        log.error(f'place_order({symbol},{side},{qty}): {e}')
        return None

def check_trailing_stops():
    global trailing_highs
    p = learner.current_params
    stop_loss = p.get('stop_loss_pct', 0.015)
    trailing = p.get('trailing_stop_pct', 0.008)
    try:
        positions = trade_client.get_all_positions()
        for pos in positions:
            sym = pos.symbol
            current = float(pos.current_price)
            pnl_pct = float(pos.unrealized_plpc)
            if pnl_pct <= -stop_loss:
                log.warning(f'STOP-LOSS {sym}: {pnl_pct:.2%}')
                send_telegram(f'<b>STOP-LOSS</b> {sym}: {pnl_pct:.2%}')
                trade_client.close_position(sym)
                learner.record_trade(sym, 'sell', float(pos.qty), current, pnl_pct)
                trailing_highs.pop(sym, None)
                continue
            if sym not in trailing_highs:
                trailing_highs[sym] = current
            if current > trailing_highs[sym]:
                trailing_highs[sym] = current
            drop = (trailing_highs[sym] - current) / trailing_highs[sym] if trailing_highs[sym] > 0 else 0
            if pnl_pct > 0.005 and drop > trailing:
                log.info(f'TRAILING STOP {sym}: drop={drop:.2%}')
                send_telegram(f'<b>TRAILING STOP</b> {sym}: locked profit, drop={drop:.2%}')
                trade_client.close_position(sym)
                learner.record_trade(sym, 'sell', float(pos.qty), current, pnl_pct)
                trailing_highs.pop(sym, None)
    except Exception as e:
        log.error(f'check_trailing_stops: {e}')

def refresh_intelligence():
    global intel_cache, last_intel_time
    now = time.time()
    if now - last_intel_time < INTEL_INTERVAL:
        return
    log.info('--- Refreshing market intelligence ---')
    for sym in CRYPTO_SYMBOLS:
        try:
            intel_cache[sym] = build_market_intelligence(sym)
        except Exception as e:
            log.error(f'Intel error {sym}: {e}')
            intel_cache[sym] = {'score': 0, 'confidence': 0, 'reasons': [], 'fng': 50, 'fng_label': 'N/A', 'top_headlines': []}
    last_intel_time = now
    fng = intel_cache.get('BTC/USD', {}).get('fng', 50)
    send_telegram(f'<b>INTEL UPDATE</b>\nFear&Greed: {fng}\nRegime: {learner.regime.upper()}\n' + '\n'.join([f"{s}: score={intel_cache[s]['score']}" for s in CRYPTO_SYMBOLS if s in intel_cache]))

def run_learning():
    global last_learn_time
    now = time.time()
    if now - last_learn_time < LEARN_INTERVAL:
        return
    last_learn_time = now
    try:
        params, report = learner.run_learning_cycle()
        send_telegram(report)
    except Exception as e:
        log.error(f'Learning cycle error: {e}')

def trade_symbol(symbol, is_crypto, portfolio):
    p = learner.current_params
    bars = get_bars(symbol, is_crypto)
    if bars is None or len(bars) == 0:
        return False
    signal, ta_conf, details = multi_tf_signal(symbol, is_crypto)
    intel = intel_cache.get(symbol, {'confidence': 0, 'score': 0, 'reasons': []})
    intel_conf = intel.get('confidence', 0)
    combined = (ta_conf * 0.7) + (max(0, intel_conf) * 0.3)
    qty_held = get_position_qty(symbol)
    price = details.get('price', 0)
    if price == 0:
        return False
    max_pos = p.get('max_position_pct', 0.40)
    min_conf = p.get('min_confidence', 0.15)
    max_qty = (portfolio * max_pos) / price
    traded = False
    if signal == 1 and qty_held <= 0 and combined >= min_conf:
        qty = max_qty * max(combined, 0.5)
        if qty * price >= 1.0:
            order = place_order(symbol, OrderSide.BUY, qty)
            if order:
                learner.record_trade(symbol, 'buy', qty, price)
                reasons = intel.get('reasons', [])
                rsi = details.get('rsi', 0)
                msg = (f'<b>BUY {symbol}</b>\n'
                       f'Qty: {qty:.4f} @ ${price:,.2f}\n'
                       f'Combined: {combined:.0%} | Regime: {learner.regime}\n'
                       f'RSI: {rsi:.1f} | MACD: {"Bull" if details.get("macd_bull") else "Bear"}\n'
                       f'Reasons: {chr(10).join(reasons[:3])}')
                send_telegram(msg)
                traded = True
    elif signal == -1 and qty_held > 0:
        order = place_order(symbol, OrderSide.SELL, qty_held)
        if order:
            try:
                pos = trade_client.get_open_position(symbol.replace('/', ''))
                pnl = float(pos.unrealized_plpc)
            except:
                pnl = 0
            learner.record_trade(symbol, 'sell', qty_held, price, pnl)
            send_telegram(f'<b>SELL {symbol}</b>\nQty: {qty_held:.4f} @ ${price:,.2f}')
            traded = True
    return traded

def force_entry(portfolio):
    p = learner.current_params
    log.info('FORCE ENTRY: scanning for best opportunity...')
    best = None
    best_score = -999
    for sym in CRYPTO_SYMBOLS:
        bars = get_bars(sym, True)
        if bars is None or len(bars) < 30:
            continue
        sig, conf, details = multi_tf_signal(sym, True)
        intel = intel_cache.get(sym, {'confidence': 0, 'score': 0})
        combined = (conf * 0.7) + (max(0, intel.get('confidence', 0)) * 0.3)
        total_score = sig * combined
        log.info(f'SCAN {sym}: signal={sig} conf={conf:.2f} combined={combined:.2f}')
        if total_score > best_score:
            best_score = total_score
            best = {'symbol': sym, 'confidence': combined, 'details': details, 'intel': intel}
    if best is None or best['details'].get('price', 0) == 0:
        return False
    sym = best['symbol']
    price = best['details']['price']
    if get_position_qty(sym) > 0:
        return False
    max_pos = p.get('max_position_pct', 0.40)
    qty = (portfolio * max_pos * 0.5) / price
    if qty * price < 1.0:
        return False
    order = place_order(sym, OrderSide.BUY, qty)
    if order:
        learner.record_trade(sym, 'buy', qty, price)
        send_telegram(f'<b>FORCE BUY {sym}</b>\nQty: {qty:.4f} @ ${price:,.2f}\nRegime: {learner.regime}\nConf: {best["confidence"]:.0%}')
        return True
    return False

def trade_cycle():
    global idle_cycles, cycle_count
    cycle_count += 1
    portfolio = get_portfolio_value()
    daily_pnl = get_daily_pnl_pct()
    p = learner.current_params
    force_cycles = int(p.get('force_entry_cycles', 5))
    log.info(f'Portfolio: ${portfolio:,.2f} | PnL: {daily_pnl:.2%} | Idle: {idle_cycles} | Regime: {learner.regime} | Cycle: {cycle_count}')
    # Record daily return every 60 cycles (~1 hour)
    if cycle_count % 60 == 0:
        learner.record_daily_return(daily_pnl, portfolio)
    if daily_pnl >= DAILY_PROFIT_TARGET:
        log.info('TARGET REACHED!')
        send_telegram(f'<b>TARGET REACHED!</b> PnL={daily_pnl:.2%} - closing all positions')
        trade_client.close_all_positions(cancel_orders=True)
        idle_cycles = 0
        return
    refresh_intelligence()
    run_learning()
    check_trailing_stops()
    traded = False
    for sym in CRYPTO_SYMBOLS:
        try:
            if trade_symbol(sym, True, portfolio):
                traded = True
        except Exception as e:
            log.error(f'{sym}: {e}')
    for sym in STOCK_SYMBOLS:
        try:
            if trade_symbol(sym, False, portfolio):
                traded = True
        except Exception as e:
            log.error(f'{sym}: {e}')
    if traded:
        idle_cycles = 0
    else:
        idle_cycles += 1
        log.info(f'No trades this cycle. Idle count: {idle_cycles}/{force_cycles}')
    if idle_cycles >= force_cycles:
        try:
            positions = trade_client.get_all_positions()
            if len(positions) == 0:
                if force_entry(portfolio):
                    idle_cycles = 0
        except Exception as e:
            log.error(f'Force entry check: {e}')

def main():
    log.info('=== Self-Learning Trading Bot v4 Started ===')
    send_telegram('<b>Bot v4 SELF-LEARNING Started</b>\nTarget: 3%+ daily\nAuto-optimizing params every 15min\nBenchmarks: BTC, ETH, S&P500, NASDAQ\nRegime detection: bull/bear/sideways\nAdaptive RSI, position sizing, stops')
    while True:
        try:
            trade_cycle()
        except Exception as e:
            log.error(f'Main loop: {e}')
            send_telegram(f'<b>ERROR</b>: {e}')
        time.sleep(TRADE_INTERVAL)

if __name__ == '__main__':
    main()

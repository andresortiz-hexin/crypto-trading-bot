"""V2 Institutional Trading Bot.
Integrates: SignalEngine, RegimeEngine, RiskEngine, Intelligence, SelfLearner.
Key improvements over v4:
- Multi-factor signal scoring (not just RSI/MACD)
- Regime-aware position sizing and thresholds
- Risk engine with kill switch, daily/weekly limits
- No force-entry; only trade on confirmed signals
- ATR-based stop-loss and take-profit
- Conservative daily target (0.5-1% vs old 3%)
"""

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

# V2 Modules
from signal_engine import SignalEngine
from regime_engine import RegimeEngine
from risk_engine import RiskEngine
from intelligence import build_market_intelligence
from self_learner import SelfLearner

# -- Config --
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')
PAPER = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# V2 CONFIG - Conservative institutional targets
DAILY_PROFIT_TARGET = 0.008  # 0.8% daily (realistic)
DAILY_LOSS_LIMIT = 0.015  # 1.5% max daily loss
TRADE_INTERVAL = 90  # 90 seconds between cycles
INTEL_INTERVAL = 300  # 5 min intel refresh
LEARN_INTERVAL = 900  # 15 min learning cycle
REGIME_INTERVAL = 600  # 10 min regime check

CRYPTO_SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD']
STOCK_SYMBOLS = ['SPY', 'QQQ']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger('bot')

# -- Telegram --
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': msg,
            'parse_mode': 'HTML'
        }, timeout=10)
    except Exception as e:
        log.warning(f'Telegram: {e}')

# -- Clients --
trade_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
crypto_client = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# -- V2 Engines --
signal_engine = SignalEngine()
regime_engine = RegimeEngine()
risk_engine = RiskEngine(
    max_trades_per_day=12,
    daily_loss_limit_pct=DAILY_LOSS_LIMIT,
    weekly_loss_limit_pct=0.04,
    max_position_pct=0.25,
    max_portfolio_exposure=0.70,
    max_correlated_exposure=0.40,
)
learner = SelfLearner()

# -- State --
intel_cache = {}
last_intel_time = 0
last_learn_time = 0
last_regime_time = 0
cycle_count = 0

def normalize_df(df):
    """Normalize Alpaca bar data to standard format."""
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

def get_position_qty(symbol):
    sym = symbol.replace('/', '')
    try:
        pos = trade_client.get_open_position(sym)
        return float(pos.qty)
    except:
        return 0.0

def get_position_pnl(symbol):
    """Get unrealized PnL percentage for a position."""
    sym = symbol.replace('/', '')
    try:
        pos = trade_client.get_open_position(sym)
        return float(pos.unrealized_plpc)
    except:
        return 0.0

def place_order(symbol, side, qty):
    sym = symbol.replace('/', '')
    try:
        req = MarketOrderRequest(
            symbol=sym,
            qty=round(qty, 6),
            side=side,
            time_in_force=TimeInForce.GTC
        )
        order = trade_client.submit_order(req)
        log.info(f'ORDER {side.value.upper()} {round(qty, 4)} {sym} id={order.id}')
        return order
    except Exception as e:
        log.error(f'place_order({symbol},{side},{qty}): {e}')
        return None

# =============================================
# V2 CORE: Risk-Managed Trading
# =============================================

def check_risk_managed_stops():
    """V2: Check positions with regime-based dynamic stops."""
    try:
        positions = trade_client.get_all_positions()
        for pos in positions:
            sym = pos.symbol
            pnl_pct = float(pos.unrealized_plpc)
            current = float(pos.current_price)
            cost = float(pos.avg_entry_price)
            qty = float(pos.qty)

            # Dynamic stop based on regime
            regime = regime_engine.current_regime
            if regime == 'stress':
                stop_pct = -0.012   # Tight stop in stress
                take_pct = 0.008
            elif regime == 'uptrend':
                stop_pct = -0.025   # Wider stop in uptrend
                take_pct = 0.02
            else:  # sideways
                stop_pct = -0.018
                take_pct = 0.012

            # STOP LOSS
            if pnl_pct <= stop_pct:
                log.warning(f'STOP-LOSS {sym}: {pnl_pct:.2%} (limit {stop_pct:.2%})')
                send_telegram(f'<b>STOP-LOSS</b> {sym}: {pnl_pct:.2%}')
                trade_client.close_position(sym)
                learner.record_trade(sym, 'sell', qty, current, pnl_pct)
                risk_engine.update_daily_pnl(get_daily_pnl_pct())
                risk_engine.record_stop_loss(sym)
                continue

            # TAKE PROFIT
            if pnl_pct >= take_pct:
                log.info(f'TAKE-PROFIT {sym}: {pnl_pct:.2%}')
                send_telegram(f'<b>TAKE-PROFIT</b> {sym}: +{pnl_pct:.2%}')
                trade_client.close_position(sym)
                learner.record_trade(sym, 'sell', qty, current, pnl_pct)
                risk_engine.update_daily_pnl(get_daily_pnl_pct())
                continue

            # TRAILING STOP: if in profit > 0.5%, trail at 40% of gains
            if pnl_pct > 0.005:
                trail_stop = cost * (1 + pnl_pct * 0.6)  # Lock 60% of gains
                if current < trail_stop:
                    log.info(f'TRAILING STOP {sym}: locked profit {pnl_pct:.2%}')
                    send_telegram(f'<b>TRAILING STOP</b> {sym}: +{pnl_pct:.2%}')
                    trade_client.close_position(sym)
                    learner.record_trade(sym, 'sell', qty, current, pnl_pct)
                    risk_engine.update_daily_pnl(get_daily_pnl_pct())
    except Exception as e:
        log.error(f'check_risk_managed_stops: {e}')

def refresh_intelligence():
    """Refresh market intelligence for all symbols."""
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
            intel_cache[sym] = {'score': 0, 'confidence': 0, 'reasons': [], 'fng': 50}
    last_intel_time = now

    fng = intel_cache.get('BTC/USD', {}).get('fng', 50)
    regime = regime_engine.current_regime
    send_telegram(
        f'<b>INTEL UPDATE</b>\n'
        f'Regime: {regime.upper()} | FnG: {fng}\n' +
        '\n'.join([f"{s}: score={intel_cache[s].get('score', 0)}" for s in CRYPTO_SYMBOLS if s in intel_cache])
    )

def refresh_regime():
    """Update regime detection using SPY hourly bars."""
    global last_regime_time
    now = time.time()
    if now - last_regime_time < REGIME_INTERVAL:
        return
    last_regime_time = now

    try:
        bars_spy = get_bars('SPY', False, TimeFrame.Hour, 100)
        if bars_spy is not None and len(bars_spy) >= 50:
            regime = regime_engine.classify(bars_spy, symbol='SPY')
            log.info(f'Regime: {regime}')
        else:
            log.warning('Insufficient data for regime classification')
    except Exception as e:
        log.error(f'Regime detection error: {e}')

def run_learning():
    """Run learning cycle."""
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
    """V2: Trade a symbol using multi-factor signals + risk management."""
    regime = regime_engine.current_regime

    # Get bars for signal analysis
    bars_1m = get_bars(symbol, is_crypto, TimeFrame.Minute, 100)
    bars_1h = get_bars(symbol, is_crypto, TimeFrame.Hour, 100)

    if bars_1m is None or len(bars_1m) < 30:
        return False

    # Get multi-factor signal from V2 signal engine (FIXED: use compute_score)
    score, direction, details = signal_engine.compute_score(
        bars=bars_1m,
        bars_hourly=bars_1h,
    )

    # Get intelligence overlay
    intel = intel_cache.get(symbol, {'confidence': 0, 'score': 0, 'reasons': []})
    intel_conf = intel.get('confidence', 0)

    # Normalize score to 0-1 confidence
    confidence = score / 100.0

    # Combined confidence: 60% technical, 40% intelligence
    combined = (confidence * 0.6) + (max(0, intel_conf) * 0.4)

    # Regime-adjusted thresholds (FIXED: use actual regime names)
    if regime == 'stress':
        min_conf = 0.45   # Need strong signal in stress
        position_scale = 0.5  # Half-size positions
    elif regime == 'uptrend':
        min_conf = 0.25
        position_scale = 1.0
    else:  # sideways
        min_conf = 0.35
        position_scale = 0.7

    # Also check regime min signal score
    min_score = regime_engine.get_min_signal_score()
    if score < min_score:
        log.info(f'{symbol}: score {score} < regime min {min_score}')
        return False

    # Check if asset is allowed in current regime
    if not regime_engine.is_asset_allowed(symbol):
        log.info(f'{symbol}: not allowed in {regime} regime')
        return False

    qty_held = get_position_qty(symbol)
    price = details.get('price', 0)
    if price == 0:
        return False

    # Risk engine checks
    positions = []
    try:
        positions = trade_client.get_all_positions()
    except:
        pass

    traded = False

    # === BUY LOGIC ===
    if direction == 1 and qty_held <= 0 and combined >= min_conf:
        # Check risk engine approval
        can_trade, reason = risk_engine.can_trade(symbol, portfolio, positions)
        if not can_trade:
            log.info(f'RISK BLOCKED {symbol}: {reason}')
            return False

        # Calculate position size via risk engine (FIXED: correct signature)
        volatility = details.get('atr_pct', 0.02)
        size_pct = risk_engine.calculate_position_size(
            symbol=symbol,
            confidence=combined,
            regime=regime,
            volatility=volatility,
            portfolio=portfolio,
        )
        size_pct *= position_scale

        qty = (portfolio * size_pct) / price

        if qty * price >= 1.0:
            order = place_order(symbol, OrderSide.BUY, qty)
            if order:
                learner.record_trade(symbol, 'buy', qty, price)
                risk_engine.record_trade(symbol, 'buy')
                reasons = intel.get('reasons', [])[:2]
                msg = (
                    f'<b>BUY {symbol}</b>\n'
                    f'Qty: {qty:.4f} @ ${price:,.2f}\n'
                    f'Size: {size_pct:.1%} | Conf: {combined:.0%}\n'
                    f'Regime: {regime} | Score: {score}\n'
                    f'RSI: {details.get("rsi", 0):.0f}\n'
                    f'Intel: {" | ".join(reasons)}'
                )
                send_telegram(msg)
                traded = True

    # === SELL LOGIC ===
    elif direction == -1 and qty_held > 0:
        pnl = get_position_pnl(symbol)
        order = place_order(symbol, OrderSide.SELL, qty_held)
        if order:
            learner.record_trade(symbol, 'sell', qty_held, price, pnl)
            risk_engine.record_trade(symbol, 'sell', pnl)
            risk_engine.update_daily_pnl(get_daily_pnl_pct())
            send_telegram(f'<b>SELL {symbol}</b>\nQty: {qty_held:.4f} @ ${price:,.2f}\nPnL: {pnl:.2%}')
            traded = True

    return traded

def trade_cycle():
    """V2: Main trading cycle with full risk management."""
    global cycle_count
    cycle_count += 1

    portfolio = get_portfolio_value()
    daily_pnl = get_daily_pnl_pct()
    regime = regime_engine.current_regime

    # Update risk engine with current PnL
    risk_engine.update_daily_pnl(daily_pnl)

    log.info(
        f'Cycle {cycle_count} | ${portfolio:,.2f} | PnL: {daily_pnl:.2%} | '
        f'Regime: {regime} | Risk: {"KILL" if risk_engine.kill_switch_active else "OK"}'
    )

    # Record daily return every 40 cycles (~1 hour)
    if cycle_count % 40 == 0:
        learner.record_daily_return(daily_pnl, portfolio)

    # Check kill switch
    if risk_engine.kill_switch_active:
        log.warning(f'KILL SWITCH ACTIVE: {risk_engine.kill_switch_reason}')
        if cycle_count % 20 == 0:  # Remind every ~30 min
            send_telegram(f'<b>KILL SWITCH</b>: {risk_engine.kill_switch_reason}\nPnL: {daily_pnl:.2%}')
        return

    # Check daily target reached
    if daily_pnl >= DAILY_PROFIT_TARGET:
        log.info(f'DAILY TARGET REACHED: {daily_pnl:.2%}')
        send_telegram(f'<b>TARGET REACHED!</b> PnL={daily_pnl:.2%} - holding positions')
        # Don't close all - just stop opening new ones
        return

    # Refresh engines
    refresh_regime()
    refresh_intelligence()
    run_learning()

    # Check stops on existing positions
    check_risk_managed_stops()

    # Trade crypto symbols
    for sym in CRYPTO_SYMBOLS:
        try:
            trade_symbol(sym, True, portfolio)
        except Exception as e:
            log.error(f'{sym}: {e}')

    # Trade stock symbols (only during market hours)
    hour = datetime.utcnow().hour
    if 14 <= hour <= 20:  # ~9:30 AM - 4 PM ET
        for sym in STOCK_SYMBOLS:
            try:
                trade_symbol(sym, False, portfolio)
            except Exception as e:
                log.error(f'{sym}: {e}')

def main():
    log.info('=== V2 Institutional Trading Bot Started ===')
    send_telegram(
        '<b>Bot V2 INSTITUTIONAL Started</b>\n'
        'Target: 0.5-1% daily (conservative)\n'
        'Modules: SignalEngine + RegimeEngine + RiskEngine\n'
        'Risk: Kill switch, daily/weekly limits\n'
        'Regime: uptrend/sideways/stress detection\n'
        'Signals: Multi-factor (RSI+MACD+EMA+BB+VWAP+Volume)'
    )
    while True:
        try:
            trade_cycle()
        except Exception as e:
            log.error(f'Main loop: {e}')
            send_telegram(f'<b>ERROR</b>: {e}')
        time.sleep(TRADE_INTERVAL)

if __name__ == '__main__':
    main()

"""
Risk Engine - Position sizing, limits and kill switches
V2 Institutional Upgrade - April 2026

Manages:
- Per-trade risk sizing (max 0.5% NAV per trade)
- Per-instrument exposure limits
- Daily/weekly loss limits
- Kill switches for drawdown protection
- Trade frequency controls
"""

import logging
from datetime import datetime

log = logging.getLogger(__name__)

# Leveraged ETFs that need special treatment
LEVERAGED_ETFS = ['TQQQ', 'SOXL', 'SPXL', 'SQQQ', 'SPXS']


class RiskEngine:
    """
    Institutional-grade risk management engine.
    
    Enforces hard limits on position size, daily losses, 
    weekly losses, trade frequency, and aggregate exposure.
    """

    def __init__(self):
        # Per-trade risk
        self.max_risk_per_trade_pct = 0.005  # 0.5% of NAV
        
        # Per-instrument limits
        self.max_position_pct_core = 0.20    # 20% for QQQ/SPY
        self.max_position_pct_tactical = 0.10  # 10% for TQQQ
        self.max_position_pct_crypto = 0.15  # 15% for BTC/USD
        
        # Aggregate limits
        self.max_total_exposure_pct = 0.70   # 70% max invested
        self.max_leveraged_exposure_pct = 0.15  # 15% max in leveraged ETFs
        self.max_crypto_exposure_pct = 0.25  # 25% max in crypto
        
        # Daily/weekly loss limits
        self.daily_loss_limit_pct = -0.015   # -1.5% daily
        self.weekly_loss_limit_pct = -0.04   # -4% weekly
        
        # Trade frequency
        self.max_trades_per_day = 6
        self.cooldown_after_stop_bars = 30   # 30 minute cooldown after stop
        
        # State tracking
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.last_trade_date = None
        self.kill_switch_active = False
        self.kill_switch_reason = ''
        self.stopped_symbols = {}  # symbol -> cooldown expiry
        self.trade_log = []

    def reset_daily(self):
        """Reset daily counters. Call at start of each trading day."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        if self.last_trade_date != today:
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today
            log.info(f'Risk engine: daily reset for {today}')

    def reset_weekly(self):
        """Reset weekly counters. Call at start of each trading week."""
        self.weekly_pnl = 0.0
        log.info('Risk engine: weekly reset')

    def can_trade(self, symbol, portfolio_value, current_positions):
        """
        Check if a new trade is allowed.
        
        Returns: (allowed: bool, reason: str)
        """
        # Kill switch check
        if self.kill_switch_active:
            return False, f'Kill switch active: {self.kill_switch_reason}'

        # Daily loss limit
        if self.daily_pnl <= self.daily_loss_limit_pct:
            return False, f'Daily loss limit reached: {self.daily_pnl:.2%}'

        # Weekly loss limit
        if self.weekly_pnl <= self.weekly_loss_limit_pct:
            self.activate_kill_switch('Weekly loss limit reached')
            return False, f'Weekly loss limit reached: {self.weekly_pnl:.2%}'

        # Trade frequency
        if self.trades_today >= self.max_trades_per_day:
            return False, f'Max trades per day reached: {self.trades_today}'

        # Symbol cooldown after stop
        if symbol in self.stopped_symbols:
            return False, f'{symbol} in cooldown after stop-loss'

        # Aggregate exposure check
        total_exposure = self._calc_total_exposure(current_positions, portfolio_value)
        if total_exposure >= self.max_total_exposure_pct:
            return False, f'Max total exposure reached: {total_exposure:.2%}'

        # Leveraged ETF exposure check
        sym_clean = symbol.replace('/', '')
        if sym_clean in LEVERAGED_ETFS:
            lev_exposure = self._calc_leveraged_exposure(current_positions, portfolio_value)
            if lev_exposure >= self.max_leveraged_exposure_pct:
                return False, f'Max leveraged exposure reached: {lev_exposure:.2%}'

        # Crypto exposure check
        if '/' in symbol:  # Crypto symbols have /
            crypto_exposure = self._calc_crypto_exposure(current_positions, portfolio_value)
            if crypto_exposure >= self.max_crypto_exposure_pct:
                return False, f'Max crypto exposure reached: {crypto_exposure:.2%}'

        return True, 'OK'

    def calculate_position_size(self, symbol, price, stop_price, portfolio_value):
        """
        Calculate position size based on risk per trade.
        
        Uses fixed fractional method:
        qty = (NAV * max_risk_pct) / (price - stop_price)
        
        Then caps by instrument-level limits.
        """
        if price <= 0 or stop_price <= 0 or price <= stop_price:
            return 0.0

        # Risk-based sizing
        risk_per_share = price - stop_price
        max_loss_dollars = portfolio_value * self.max_risk_per_trade_pct
        risk_qty = max_loss_dollars / risk_per_share

        # Cap by instrument position limit
        sym_clean = symbol.replace('/', '')
        if sym_clean in LEVERAGED_ETFS:
            max_pct = self.max_position_pct_tactical
        elif '/' in symbol:
            max_pct = self.max_position_pct_crypto
        else:
            max_pct = self.max_position_pct_core

        max_qty_by_limit = (portfolio_value * max_pct) / price
        final_qty = min(risk_qty, max_qty_by_limit)

        # Minimum order value check ($1 for Alpaca)
        if final_qty * price < 1.0:
            return 0.0

        log.info(
            f'SIZING {symbol}: risk_qty={risk_qty:.4f} limit_qty={max_qty_by_limit:.4f} '
            f'final={final_qty:.4f} (${final_qty * price:.2f})'
        )
        return round(final_qty, 6)

    def calculate_stop_price(self, price, atr_pct, is_leveraged=False):
        """
        Calculate stop-loss price based on ATR.
        
        For leveraged ETFs, use tighter stops.
        """
        if is_leveraged:
            stop_distance = max(atr_pct * 1.5, 0.01)  # Min 1% for leveraged
            stop_distance = min(stop_distance, 0.03)   # Max 3% for leveraged
        else:
            stop_distance = max(atr_pct * 2.0, 0.008)  # Min 0.8%
            stop_distance = min(stop_distance, 0.05)    # Max 5%

        return price * (1 - stop_distance)

    def calculate_take_profit(self, price, stop_price, risk_reward=2.0):
        """Calculate take-profit at specified risk:reward ratio."""
        risk = price - stop_price
        return price + (risk * risk_reward)

    def record_trade(self, symbol, side, pnl_pct=None):
        """Record a trade for tracking."""
        self.trades_today += 1
        if pnl_pct is not None:
            self.daily_pnl += pnl_pct
            self.weekly_pnl += pnl_pct

        self.trade_log.append({
            'ts': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'side': side,
            'pnl_pct': pnl_pct,
            'daily_pnl': self.daily_pnl,
        })
        self.trade_log = self.trade_log[-500:]

    def record_stop_loss(self, symbol):
        """Record a stop-loss hit and set cooldown."""
        self.stopped_symbols[symbol] = True
        log.warning(f'RISK: {symbol} stopped out, cooldown active')

    def clear_cooldown(self, symbol):
        """Clear cooldown for a symbol."""
        self.stopped_symbols.pop(symbol, None)

    def update_daily_pnl(self, pnl_pct):
        """Update daily PnL from account data."""
        self.daily_pnl = pnl_pct

        # Check daily limit
        if self.daily_pnl <= self.daily_loss_limit_pct:
            log.warning(f'RISK: Daily loss limit triggered: {self.daily_pnl:.2%}')

    def activate_kill_switch(self, reason):
        """Activate emergency kill switch."""
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        log.critical(f'KILL SWITCH ACTIVATED: {reason}')

    def deactivate_kill_switch(self):
        """Deactivate kill switch (requires manual review)."""
        self.kill_switch_active = False
        self.kill_switch_reason = ''
        log.info('Kill switch deactivated')

    def _calc_total_exposure(self, positions, portfolio):
        """Calculate total exposure as % of portfolio."""
        if not positions or portfolio <= 0:
            return 0.0
        total = sum(abs(float(p.market_value)) for p in positions)
        return total / portfolio

    def _calc_leveraged_exposure(self, positions, portfolio):
        """Calculate leveraged ETF exposure as % of portfolio."""
        if not positions or portfolio <= 0:
            return 0.0
        lev_total = sum(
            abs(float(p.market_value)) for p in positions 
            if p.symbol in LEVERAGED_ETFS
        )
        return lev_total / portfolio

    def _calc_crypto_exposure(self, positions, portfolio):
        """Calculate crypto exposure as % of portfolio."""
        if not positions or portfolio <= 0:
            return 0.0
        crypto_total = sum(
            abs(float(p.market_value)) for p in positions
            if any(c in p.symbol for c in ['BTC', 'ETH', 'SOL'])
        )
        return crypto_total / portfolio

    def get_status(self):
        """Get risk engine status summary."""
        return {
            'kill_switch': self.kill_switch_active,
            'kill_reason': self.kill_switch_reason,
            'trades_today': self.trades_today,
            'max_trades': self.max_trades_per_day,
            'daily_pnl': round(self.daily_pnl, 4),
            'daily_limit': self.daily_loss_limit_pct,
            'weekly_pnl': round(self.weekly_pnl, 4),
            'weekly_limit': self.weekly_loss_limit_pct,
            'cooldown_symbols': list(self.stopped_symbols.keys()),
        }

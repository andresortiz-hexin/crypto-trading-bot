"""
Regime Engine - Market regime classification module
V2 Institutional Upgrade - April 2026

Classifies market into UPTREND, SIDEWAYS, or STRESS states
based on underlying index (QQQ/SPY) behavior, not leveraged ETFs.
"""

import logging
import numpy as np
import pandas as pd
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

log = logging.getLogger(__name__)

# Regime states
UPTREND = 'uptrend'
SIDEWAYS = 'sideways'
STRESS = 'stress'


class RegimeEngine:
    """
    Institutional-grade market regime classifier.

    Uses QQQ/SPY daily data to determine market state:
    - UPTREND: Price above EMA, positive slope, moderate volatility
    - SIDEWAYS: Price near EMA, flat slope, normal volatility
    - STRESS: Price below EMA, high volatility, significant drawdown

    Regime determines which assets are tradeable and signal thresholds.
    """

    def __init__(self):
        self.current_regime = SIDEWAYS
        self.regime_history = []
        self.regime_change_count = 0

        # Regime-specific rules
        self.regime_rules = {
            UPTREND: {
                                'allowed_assets': ['QQQ', 'SPY', 'TQQQ', 'IWM', 'XLK', 'BTC/USD', 'ETH/USD', 'SOL/USD', 'GLD'],
                'max_leveraged_pct': 0.15,
                                'min_signal_score': 30,
                                'max_trades_per_day': 15,
                'description': 'Trend favorable - tactical leverage allowed'
            },
            SIDEWAYS: {
                                'allowed_assets': ['QQQ', 'SPY', 'TQQQ', 'IWM', 'BTC/USD', 'ETH/USD', 'SOL/USD', 'XLK'],
                'max_leveraged_pct': 0.0,
                                'min_signal_score': 25,
                                'max_trades_per_day': 10,
                'description': 'Range market - moderate mode, diversified crypto'
            },
            STRESS: {
                                'allowed_assets': ['SPY', 'QQQ', 'BTC/USD', 'ETH/USD', 'GLD'],
                'max_leveraged_pct': 0.0,
                                'min_signal_score': 35,
                                'max_trades_per_day': 6,
                'description': 'High risk - reduced exposure, defensive assets only'
            }
        }

    def classify(self, bars_daily, symbol='QQQ'):
        """
        Classify market regime using daily bars of underlying index.

        Args:
            bars_daily: DataFrame with OHLCV daily data (min 100 bars)
            symbol: Index symbol used for classification

        Returns:
            str: Current regime (UPTREND, SIDEWAYS, STRESS)
        """
        if bars_daily is None or len(bars_daily) < 50:
            log.warning('Insufficient data for regime classification')
            return self.current_regime

        try:
            close = bars_daily['close']
            high = bars_daily['high']
            low = bars_daily['low']

            # 1. EMA slope analysis
            ema50 = EMAIndicator(close, window=50).ema_indicator()
            ema200 = EMAIndicator(close, window=min(200, len(close) - 1)).ema_indicator()

            current_price = close.iloc[-1]
            current_ema50 = ema50.iloc[-1]

            # EMA slope: compare last value vs 10 bars ago
            if len(ema50) >= 10:
                ema_slope = (ema50.iloc[-1] - ema50.iloc[-10]) / ema50.iloc[-10]
            else:
                ema_slope = 0.0

            # 2. Volatility analysis (ATR as % of price)
            atr = AverageTrueRange(high, low, close, window=14).average_true_range()
            current_atr = atr.iloc[-1]
            atr_pct = current_atr / current_price if current_price > 0 else 0

            # Historical ATR for comparison
            atr_mean = atr.tail(50).mean()
            atr_ratio = current_atr / atr_mean if atr_mean > 0 else 1.0

            # 3. Drawdown from recent high
            rolling_max = close.rolling(window=50).max()
            drawdown = (close.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]

            # 4. Price position relative to EMAs
            above_ema50 = current_price > current_ema50
            if len(ema200.dropna()) > 0:
                above_ema200 = current_price > ema200.dropna().iloc[-1]
            else:
                above_ema200 = above_ema50

            # Classification logic
            old_regime = self.current_regime

            if (above_ema50 and ema_slope > 0.005 and
                    atr_ratio < 1.5 and drawdown > -0.05):
                self.current_regime = UPTREND
            elif (drawdown < -0.08 or atr_ratio > 1.8 or
                  (not above_ema50 and ema_slope < -0.01)):
                self.current_regime = STRESS
            else:
                self.current_regime = SIDEWAYS

            # Log regime change
            if self.current_regime != old_regime:
                self.regime_change_count += 1
                log.info(
                    f'REGIME CHANGE #{self.regime_change_count}: '
                    f'{old_regime} -> {self.current_regime} | '
                    f'EMA_slope={ema_slope:.4f} ATR_ratio={atr_ratio:.2f} '
                    f'DD={drawdown:.2%} Above_EMA50={above_ema50}'
                )

            self.regime_history.append({
                'regime': self.current_regime,
                'ema_slope': round(ema_slope, 5),
                'atr_ratio': round(atr_ratio, 3),
                'drawdown': round(drawdown, 4),
                'above_ema50': above_ema50,
                'above_ema200': above_ema200,
                'price': round(current_price, 2),
            })

            # Keep last 100 entries
            self.regime_history = self.regime_history[-100:]

            return self.current_regime

        except Exception as e:
            log.error(f'Regime classification error: {e}')
            return self.current_regime

    def is_asset_allowed(self, symbol):
        """Check if an asset is allowed to trade in current regime."""
        rules = self.regime_rules.get(self.current_regime, {})
        allowed = rules.get('allowed_assets', [])
        return symbol in allowed or symbol.replace('/', '') in [a.replace('/', '') for a in allowed]

    def get_min_signal_score(self):
        """Get minimum signal score required for current regime."""
        rules = self.regime_rules.get(self.current_regime, {})
        return rules.get('min_signal_score', 35)

    def get_max_leveraged_pct(self):
        """Get maximum portfolio % allowed in leveraged ETFs."""
        rules = self.regime_rules.get(self.current_regime, {})
        return rules.get('max_leveraged_pct', 0.0)

    def get_max_trades_per_day(self):
        """Get maximum trades allowed per day in current regime."""
        rules = self.regime_rules.get(self.current_regime, {})
        return rules.get('max_trades_per_day', 4)

    def get_status(self):
        """Get regime status summary."""
        rules = self.regime_rules.get(self.current_regime, {})
        last = self.regime_history[-1] if self.regime_history else {}
        return {
            'regime': self.current_regime,
            'description': rules.get('description', ''),
            'allowed_assets': rules.get('allowed_assets', []),
            'min_signal_score': rules.get('min_signal_score', 35),
            'max_leveraged_pct': rules.get('max_leveraged_pct', 0.0),
            'max_trades_per_day': rules.get('max_trades_per_day', 4),
            'last_metrics': last,
            'regime_changes': self.regime_change_count,
        }

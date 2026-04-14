"""
Signal Engine - Multi-factor signal scoring module
V4 Aggressive Upgrade - April 2026

Optimized for ~1% daily target.
Key changes from V2:
- Lower buy threshold (50 vs 60) for more trade opportunities
- Added MACD histogram momentum detection
- Added Stochastic RSI for better oversold/overbought detection
- Faster EMA periods (5/13/34 vs 9/21/50) for quicker signals
- Stronger momentum weighting for crypto volatility capture
- Multi-timeframe bonus increased to 15 pts
"""

import logging
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

log = logging.getLogger(__name__)


class SignalEngine:
    """
    Multi-factor signal engine producing a score 0-100.

    Score composition (V4 - aggressive):
    - Trend (20 pts): EMA 5/13/34 alignment + slope
    - Momentum (25 pts): Returns + MACD histogram direction
    - RSI reversal (20 pts): Stochastic RSI for precision entries
    - Volume (15 pts): Current volume vs 20-bar average
    - Volatility (10 pts): ATR within acceptable range
    - Multi-TF bonus (15 pts): Hourly confirmation
    """

    WEIGHT_TREND = 20
    WEIGHT_MOMENTUM = 25
    WEIGHT_RSI = 20
    WEIGHT_VOLUME = 15
    WEIGHT_VOLATILITY = 10
    WEIGHT_MTF = 15

    # V4: Lower thresholds for more opportunities
    BUY_THRESHOLD = 50    # Was 60 - now triggers on moderate signals
    SELL_THRESHOLD = 30   # Was 25 - faster exits on weakness

    def __init__(self):
        self.signal_history = []

    def compute_score(self, bars, bars_hourly=None, benchmark_return_20d=None):
        """
        Compute multi-factor signal score for an asset.
        Returns: (score 0-100, direction 1/0/-1, details dict)
        """
        if bars is None or len(bars) < 30:
            return 0, 0, {'error': 'insufficient_data'}

        try:
            close = bars['close']
            high = bars['high']
            low = bars['low']
            volume = bars['volume']
            price = close.iloc[-1]
            details = {'price': price}

            trend_score = self._score_trend(close, details)
            momentum_score = self._score_momentum(close, benchmark_return_20d, details)
            rsi_score = self._score_rsi(close, details)
            volume_score = self._score_volume(volume, details)
            volatility_score = self._score_volatility(high, low, close, price, details)

            mtf_bonus = 0
            if bars_hourly is not None and len(bars_hourly) >= 30:
                mtf_bonus = self._multi_tf_bonus(bars_hourly, details)

            # V4: Add MACD histogram bonus
            macd_bonus = self._score_macd(close, details)

            raw_score = (trend_score + momentum_score + rsi_score +
                        volume_score + volatility_score + mtf_bonus + macd_bonus)
            total_score = max(0, min(100, raw_score))

            # V4: Lower buy threshold, faster sell detection
            if total_score >= self.BUY_THRESHOLD and details.get('trend_bullish', False):
                direction = 1
            elif total_score >= self.BUY_THRESHOLD and details.get('macd_bullish', False):
                direction = 1  # V4: MACD can override weak trend
            elif total_score <= self.SELL_THRESHOLD and not details.get('trend_bullish', True):
                direction = -1
            else:
                direction = 0

            details.update({
                'total_score': total_score,
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'rsi_score': rsi_score,
                'volume_score': volume_score,
                'volatility_score': volatility_score,
                'mtf_bonus': mtf_bonus,
                'macd_bonus': macd_bonus,
                'direction': direction,
            })

            self.signal_history.append({
                'score': total_score,
                'direction': direction,
                'price': round(price, 2),
            })
            self.signal_history = self.signal_history[-200:]

            return total_score, direction, details

        except Exception as e:
            log.error(f'Signal computation error: {e}')
            return 0, 0, {'error': str(e)}

    def _score_trend(self, close, details):
        """Score trend alignment (0-20). V4: Faster EMAs."""
        score = 0
        # V4: Faster EMAs for quicker signal detection
        ema5 = EMAIndicator(close, window=5).ema_indicator().iloc[-1]
        ema13 = EMAIndicator(close, window=13).ema_indicator().iloc[-1]
        ema34 = EMAIndicator(close, window=min(34, len(close) - 1)).ema_indicator().iloc[-1]
        price = close.iloc[-1]

        if price > ema5 > ema13 > ema34:
            score += 17
            details['trend_bullish'] = True
        elif price > ema5 > ema13:
            score += 13
            details['trend_bullish'] = True
        elif price > ema13:
            score += 8
            details['trend_bullish'] = True
        elif price < ema5 < ema13:
            score += 0
            details['trend_bullish'] = False
        else:
            score += 4
            details['trend_bullish'] = price > ema13

        # EMA slope - use faster period
        ema13_series = EMAIndicator(close, window=13).ema_indicator()
        if len(ema13_series) >= 3:
            slope = (ema13_series.iloc[-1] - ema13_series.iloc[-3]) / ema13_series.iloc[-3]
            if slope > 0.001:
                score += 3
            details['ema_slope'] = round(slope, 5)

        details['ema5'] = round(ema5, 2)
        details['ema13'] = round(ema13, 2)
        return min(score, self.WEIGHT_TREND)

    def _score_momentum(self, close, benchmark_return=None, details=None):
        """Score momentum (0-25). V4: More sensitive."""
        score = 0

        if len(close) >= 20:
            ret_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            if ret_20 > 0.015:
                score += 12
            elif ret_20 > 0:
                score += 8
            elif ret_20 > -0.005:
                score += 5  # V4: Small score even for flat
            details['return_20'] = round(ret_20, 4)

        if len(close) >= 5:
            ret_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
            if ret_5 > 0.003:
                score += 6  # V4: Increased from 5
            elif ret_5 > 0:
                score += 4
            elif ret_5 > -0.003:
                score += 2  # V4: Small positive even for slight negative
            details['return_5'] = round(ret_5, 4)

        # V4: Short-term momentum (3 bars)
        if len(close) >= 3:
            ret_3 = (close.iloc[-1] - close.iloc[-3]) / close.iloc[-3]
            if ret_3 > 0.002:
                score += 4
            details['return_3'] = round(ret_3, 4)

        if benchmark_return is not None and len(close) >= 20:
            ret_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            if ret_20 > benchmark_return:
                score += 3
            details['rel_momentum'] = round(ret_20 - benchmark_return, 4)

        return min(score, self.WEIGHT_MOMENTUM)

    def _score_rsi(self, close, details):
        """Score RSI mean-reversion signal (0-20). V4: StochRSI."""
        score = 0
        rsi = RSIIndicator(close, window=14).rsi()
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) >= 2 else current_rsi
        details['rsi'] = round(current_rsi, 1)

        # V4: Use Stochastic RSI for precision
        try:
            stoch_rsi = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
            stoch_k = stoch_rsi.stochrsi_k().iloc[-1]
            stoch_d = stoch_rsi.stochrsi_d().iloc[-1]
            details['stoch_k'] = round(stoch_k * 100, 1)
            details['stoch_d'] = round(stoch_d * 100, 1)
        except Exception:
            stoch_k, stoch_d = 0.5, 0.5

        trend_bullish = details.get('trend_bullish', True)

        if trend_bullish:
            # V4: More generous scoring in uptrend
            if 30 <= current_rsi <= 50 and prev_rsi < 30:
                score += 20  # Perfect oversold bounce
            elif current_rsi < 40 and stoch_k < 0.25:
                score += 17  # V4: StochRSI oversold in uptrend
            elif current_rsi < 45:
                score += 14  # V4: Increased from 10
            elif 45 <= current_rsi <= 60:
                score += 10  # Healthy momentum zone
            elif current_rsi > 70:
                score += 3
            else:
                score += 8
        else:
            if current_rsi < 25:
                score += 10  # V4: Increased from 8
            elif current_rsi > 70:
                score += 0
            else:
                score += 5

        return min(score, self.WEIGHT_RSI)

    def _score_macd(self, close, details):
        """V4 NEW: Score MACD histogram momentum (0-10 bonus)."""
        try:
            macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
            hist = macd.macd_diff()
            current_hist = hist.iloc[-1]
            prev_hist = hist.iloc[-2] if len(hist) >= 2 else 0

            details['macd_hist'] = round(current_hist, 4)
            details['macd_bullish'] = current_hist > 0

            score = 0
            # Histogram turning positive (bullish crossover)
            if current_hist > 0 and prev_hist <= 0:
                score += 8  # Strong buy signal
            # Histogram positive and increasing
            elif current_hist > 0 and current_hist > prev_hist:
                score += 6
            # Histogram positive but decreasing
            elif current_hist > 0:
                score += 3
            # Histogram negative but improving
            elif current_hist < 0 and current_hist > prev_hist:
                score += 2  # Recovering

            return min(score, 10)
        except Exception:
            details['macd_hist'] = 0
            details['macd_bullish'] = False
            return 0

    def _score_volume(self, volume, details):
        """Score volume confirmation (0-15)."""
        if len(volume) < 20:
            return 0
        avg_vol = volume.tail(20).mean()
        current_vol = volume.iloc[-1]
        if avg_vol > 0:
            vol_ratio = current_vol / avg_vol
            details['volume_ratio'] = round(vol_ratio, 2)
            if vol_ratio > 1.5:
                return 15
            elif vol_ratio > 1.2:
                return 12
            elif vol_ratio > 0.8:
                return 8
            elif vol_ratio > 0.5:
                return 5  # V4: Give some points even for low volume
            else:
                return 2
        return 0

    def _score_volatility(self, high, low, close, price, details):
        """Score volatility acceptability (0-10). V4: Reduced weight."""
        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        current_atr = atr.iloc[-1]
        atr_pct = current_atr / price if price > 0 else 0
        details['atr_pct'] = round(atr_pct, 4)

        atr_mean = atr.tail(50).mean() if len(atr) >= 50 else atr.mean()
        atr_ratio = current_atr / atr_mean if atr_mean > 0 else 1.0
        details['atr_ratio'] = round(atr_ratio, 2)

        # V4: More permissive volatility scoring
        if 0.5 <= atr_ratio <= 1.5:
            return 10
        elif 0.3 <= atr_ratio <= 2.0:
            return 7
        elif atr_ratio <= 2.5:
            return 4
        return 2  # V4: Always give minimum points

    def _multi_tf_bonus(self, bars_hourly, details):
        """Bonus for multi-timeframe confirmation (0-15). V4: Increased."""
        try:
            close_h = bars_hourly['close']
            ema5_h = EMAIndicator(close_h, window=5).ema_indicator().iloc[-1]
            ema13_h = EMAIndicator(close_h, window=13).ema_indicator().iloc[-1]
            price_h = close_h.iloc[-1]
            rsi_h = RSIIndicator(close_h, window=14).rsi().iloc[-1]

            bonus = 0
            if price_h > ema5_h > ema13_h:
                bonus += 7  # V4: Increased from 5
            elif price_h > ema13_h:
                bonus += 4  # V4: Partial credit

            if 30 <= rsi_h <= 60:
                bonus += 4  # V4: Wider sweet spot
            if rsi_h < 35:
                bonus += 6  # V4: Oversold hourly is strong signal

            # V4: MACD hourly confirmation
            try:
                macd_h = MACD(close_h).macd_diff().iloc[-1]
                if macd_h > 0:
                    bonus += 3
            except Exception:
                pass

            details['hourly_aligned'] = price_h > ema5_h > ema13_h
            details['hourly_rsi'] = round(rsi_h, 1)
            return min(bonus, self.WEIGHT_MTF)
        except Exception:
            return 0

    def get_last_signals(self, n=10):
        """Get last N signal scores."""
        return self.signal_history[-n:]

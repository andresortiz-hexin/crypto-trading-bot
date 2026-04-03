"""
Signal Engine - Multi-factor signal scoring module
V2 Institutional Upgrade - April 2026

Computes a 0-100 score per asset based on:
- Trend (EMA alignment)
- Momentum (relative returns)
- Mean reversion (RSI from oversold in trend direction)
- Volume confirmation
- Volatility filter
"""

import logging
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

log = logging.getLogger(__name__)


class SignalEngine:
    """
    Multi-factor signal engine producing a score 0-100.
    
    Score composition:
    - Trend (25 pts): EMA 9/21/50 alignment + slope
    - Momentum (25 pts): 20-bar return, relative strength
    - RSI reversal (20 pts): RSI recovering from oversold in trend direction
    - Volume (15 pts): Current volume vs 20-day average
    - Volatility (15 pts): ATR within acceptable range
    """

    WEIGHT_TREND = 25
    WEIGHT_MOMENTUM = 25
    WEIGHT_RSI = 20
    WEIGHT_VOLUME = 15
    WEIGHT_VOLATILITY = 15

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

            raw_score = trend_score + momentum_score + rsi_score + volume_score + volatility_score + mtf_bonus
            total_score = max(0, min(100, raw_score))

            if total_score >= 60 and details.get('trend_bullish', False):
                direction = 1
            elif total_score <= 25 and not details.get('trend_bullish', True):
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
        """Score trend alignment (0-25)."""
        score = 0
        ema9 = EMAIndicator(close, window=9).ema_indicator().iloc[-1]
        ema21 = EMAIndicator(close, window=21).ema_indicator().iloc[-1]
        ema50 = EMAIndicator(close, window=min(50, len(close) - 1)).ema_indicator().iloc[-1]
        price = close.iloc[-1]

        if price > ema9 > ema21 > ema50:
            score += 20
            details['trend_bullish'] = True
        elif price > ema9 > ema21:
            score += 15
            details['trend_bullish'] = True
        elif price > ema21:
            score += 8
            details['trend_bullish'] = True
        elif price < ema9 < ema21:
            score += 0
            details['trend_bullish'] = False
        else:
            score += 5
            details['trend_bullish'] = price > ema21

        ema21_series = EMAIndicator(close, window=21).ema_indicator()
        if len(ema21_series) >= 5:
            slope = (ema21_series.iloc[-1] - ema21_series.iloc[-5]) / ema21_series.iloc[-5]
            if slope > 0.002:
                score += 5
            details['ema_slope'] = round(slope, 5)

        details['ema9'] = round(ema9, 2)
        details['ema21'] = round(ema21, 2)
        return min(score, self.WEIGHT_TREND)

    def _score_momentum(self, close, benchmark_return=None, details=None):
        """Score momentum (0-25)."""
        score = 0
        if len(close) >= 20:
            ret_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            if ret_20 > 0.02:
                score += 12
            elif ret_20 > 0:
                score += 8
            elif ret_20 > -0.01:
                score += 4
            details['return_20'] = round(ret_20, 4)

        if len(close) >= 5:
            ret_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
            if ret_5 > 0.005:
                score += 5
            elif ret_5 > 0:
                score += 3
            details['return_5'] = round(ret_5, 4)

        if benchmark_return is not None and len(close) >= 20:
            ret_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            if ret_20 > benchmark_return:
                score += 8
            elif ret_20 > benchmark_return - 0.01:
                score += 4
            details['rel_momentum'] = round(ret_20 - benchmark_return, 4)

        return min(score, self.WEIGHT_MOMENTUM)

    def _score_rsi(self, close, details):
        """Score RSI mean-reversion signal (0-20)."""
        score = 0
        rsi = RSIIndicator(close, window=14).rsi()
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) >= 2 else current_rsi
        details['rsi'] = round(current_rsi, 1)
        trend_bullish = details.get('trend_bullish', True)

        if trend_bullish:
            if 30 <= current_rsi <= 50 and prev_rsi < 30:
                score += 20
            elif current_rsi < 35:
                score += 15
            elif 35 <= current_rsi <= 55:
                score += 10
            elif current_rsi > 70:
                score += 2
            else:
                score += 8
        else:
            if current_rsi < 25:
                score += 8
            elif current_rsi > 70:
                score += 0
            else:
                score += 4

        return min(score, self.WEIGHT_RSI)

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
            else:
                return 3
        return 0

    def _score_volatility(self, high, low, close, price, details):
        """Score volatility acceptability (0-15)."""
        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        current_atr = atr.iloc[-1]
        atr_pct = current_atr / price if price > 0 else 0
        details['atr_pct'] = round(atr_pct, 4)

        atr_mean = atr.tail(50).mean() if len(atr) >= 50 else atr.mean()
        atr_ratio = current_atr / atr_mean if atr_mean > 0 else 1.0
        details['atr_ratio'] = round(atr_ratio, 2)

        if 0.7 <= atr_ratio <= 1.3:
            return 15
        elif 0.5 <= atr_ratio <= 1.5:
            return 10
        elif atr_ratio <= 2.0:
            return 5
        return 0

    def _multi_tf_bonus(self, bars_hourly, details):
        """Bonus for multi-timeframe confirmation (0-10)."""
        try:
            close_h = bars_hourly['close']
            ema9_h = EMAIndicator(close_h, window=9).ema_indicator().iloc[-1]
            ema21_h = EMAIndicator(close_h, window=21).ema_indicator().iloc[-1]
            price_h = close_h.iloc[-1]
            rsi_h = RSIIndicator(close_h, window=14).rsi().iloc[-1]

            bonus = 0
            if price_h > ema9_h > ema21_h:
                bonus += 5
            if 35 <= rsi_h <= 65:
                bonus += 3
            if rsi_h < 30:
                bonus += 5

            details['hourly_aligned'] = price_h > ema9_h > ema21_h
            details['hourly_rsi'] = round(rsi_h, 1)
            return min(bonus, 10)
        except Exception:
            return 0

    def get_last_signals(self, n=10):
        """Get last N signal scores."""
        return self.signal_history[-n:]

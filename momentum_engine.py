"""
Momentum Engine - Absolute and Relative momentum computation
V3 Hexin Systematic Global Allocation Strategy
April 2026

Computes:
- Absolute momentum (time-series): is the asset trending up?
- Relative momentum (cross-sectional): which assets are strongest within class?
- Composite scores for allocation decisions
"""

import logging
import numpy as np
import pandas as pd
from universe import UNIVERSE, get_asset_class, get_symbols_by_class

log = logging.getLogger(__name__)

# Lookback weights for composite momentum
LOOKBACK_WEIGHTS = {
        21: 0.35,    # 1 month (~21 trading days)
        5: 0.15,     # 1 week (5 trading days) - short-term reactivity
        63: 0.20,    # 3 months
    126: 0.20,   # 6 months
    252: 0.10,   # 12 months
}


class MomentumEngine:
    """
    Dual momentum engine: absolute + relative.
    
    Absolute momentum filters out assets with negative trends.
    Relative momentum ranks assets within each class to find the strongest.
    """

    def __init__(self):
        self.scores = {}      # symbol -> composite score
        self.abs_signals = {} # symbol -> True/False (positive momentum)
        self.rankings = {}    # asset_class -> [(symbol, score), ...]

    def compute_returns(self, closes_series, lookback):
        """Compute return over lookback period."""
        if len(closes_series) < lookback + 1:
            return np.nan
        current = closes_series.iloc[-1]
        past = closes_series.iloc[-(lookback + 1)]
        if past <= 0 or np.isnan(past):
            return np.nan
        return (current - past) / past

    def compute_composite_momentum(self, closes_series):
        """
        Compute weighted composite momentum score.
        Returns float: positive = uptrend, negative = downtrend.
        """
        score = 0.0
        total_weight = 0.0
        for lookback, weight in LOOKBACK_WEIGHTS.items():
            ret = self.compute_returns(closes_series, lookback)
            if not np.isnan(ret):
                score += ret * weight
                total_weight += weight
        if total_weight > 0:
            return score / total_weight
        return 0.0

    def compute_volatility(self, closes_series, window=20):
        """Compute annualized volatility from daily returns."""
        if len(closes_series) < window + 1:
            return 0.0
        returns = closes_series.pct_change().dropna().tail(window)
        if len(returns) < window:
            return 0.0
        return float(returns.std() * np.sqrt(252))

    def compute_all(self, data):
        """
        Compute momentum for all symbols.
        
        Args:
            data: dict of {symbol: DataFrame with 'close' column}
        
        Returns:
            dict with keys:
                'scores': {symbol: composite_score}
                'abs_signals': {symbol: bool (positive momentum)}
                'rankings': {asset_class: [(symbol, score), ...]}
                'volatilities': {symbol: annualized_vol}
        """
        scores = {}
        abs_signals = {}
        volatilities = {}

        for sym, bars in data.items():
            if 'close' not in bars.columns:
                continue
            close = bars['close']
            
            # Composite momentum
            mom = self.compute_composite_momentum(close)
            scores[sym] = mom
            abs_signals[sym] = mom > 0
            
            # Volatility
            vol = self.compute_volatility(close)
            volatilities[sym] = vol

        # Relative rankings within each asset class
        rankings = {}
        for ac in set(get_asset_class(s) for s in scores):
            class_symbols = get_symbols_by_class(ac)
            class_scores = [(s, scores[s]) for s in class_symbols if s in scores]
            class_scores.sort(key=lambda x: x[1], reverse=True)
            rankings[ac] = class_scores

        self.scores = scores
        self.abs_signals = abs_signals
        self.rankings = rankings

        # Log summary
        pos_count = sum(1 for v in abs_signals.values() if v)
        log.info(
            f'Momentum: {pos_count}/{len(abs_signals)} assets with positive momentum'
        )
        for ac, ranked in rankings.items():
            top = ranked[:3] if ranked else []
            top_str = ', '.join([f'{s}={sc:.2%}' for s, sc in top])
            log.info(f'  {ac}: {top_str}')

        return {
            'scores': scores,
            'abs_signals': abs_signals,
            'rankings': rankings,
            'volatilities': volatilities,
        }

    def get_eligible_symbols(self):
        """Get symbols with positive absolute momentum."""
        return [s for s, sig in self.abs_signals.items() if sig]

    def get_top_n_per_class(self, n=None):
        """
        Get top N symbols per asset class by relative momentum.
        If n is None, returns top 50%.
        """
        result = {}
        for ac, ranked in self.rankings.items():
            # Only include positive momentum
            positive = [(s, sc) for s, sc in ranked if self.abs_signals.get(s, False)]
            if n is None:
                                count = max(1, int(len(positive) * 0.75))  # top 75%
            else:
                count = n
            result[ac] = [s for s, _ in positive[:count]]
        return result

    def is_eligible(self, symbol):
        """Check if symbol has positive absolute momentum."""
        return self.abs_signals.get(symbol, False)

    def get_rank_within_class(self, symbol):
        """Get symbol's rank within its asset class (0 = best)."""
        ac = get_asset_class(symbol)
        ranked = self.rankings.get(ac, [])
        for i, (s, _) in enumerate(ranked):
            if s == symbol:
                return i
        return len(ranked)

    def get_summary(self):
        """Get a summary dict for reporting."""
        return {
            'total_symbols': len(self.scores),
            'positive_momentum': sum(1 for v in self.abs_signals.values() if v),
            'top_scores': sorted(
                self.scores.items(), key=lambda x: x[1], reverse=True
            )[:5],
            'bottom_scores': sorted(
                self.scores.items(), key=lambda x: x[1]
            )[:5],
        }

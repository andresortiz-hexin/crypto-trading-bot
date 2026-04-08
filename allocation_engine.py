"""
Allocation Engine - Institutional-grade portfolio allocation.
Volatility targeting, regime-aware allocation, and risk budgeting.
BlackRock/JPM-style systematic allocation framework.

Categories aligned with universe.py: equity, fixed_income, commodity, crypto
"""
import numpy as np
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

# Strategic Asset Allocation (SAA) - base weights by asset class
# Categories match universe.py: equity, fixed_income, commodity, crypto
STRATEGIC_WEIGHTS = {
    'equity': 0.30,
    'fixed_income': 0.15,
    'commodity': 0.15,
    'crypto': 0.40,
}

# Regime multipliers for tactical overlay
REGIME_MULTIPLIERS = {
    'uptrend': {
        'equity': 1.2, 'fixed_income': 0.5, 'commodity': 1.0, 'crypto': 1.3,
    },
    'sideways': {
        'equity': 0.9, 'fixed_income': 1.3, 'commodity': 1.0, 'crypto': 0.7,
    },
    'stress': {
        'equity': 0.5, 'fixed_income': 1.8, 'commodity': 1.2, 'crypto': 0.2,
    },
}

# Max allocation per asset class (hard limits)
MAX_CLASS_ALLOCATION = {
    'equity': 0.40,
    'fixed_income': 0.40,
    'commodity': 0.20,
    'crypto': 0.50,
}

# Max single position size
MAX_SINGLE_POSITION = 0.15
MIN_POSITION_SIZE = 0.02


class AllocationEngine:
    """Institutional allocation engine with volatility targeting and regime overlay."""

    def __init__(self, target_volatility=0.12, max_leverage=1.0, risk_free_rate=0.05):
        self.target_vol = target_volatility
        self.max_leverage = max_leverage
        self.risk_free_rate = risk_free_rate
        self.current_allocations = {}
        self.allocation_history = []

    def compute_allocations(self, regime, momentum_engine, price_data, portfolio_value):
        """
        Compute target allocations using multi-layer framework:
        1. Strategic Asset Allocation (SAA) base
        2. Regime tactical overlay
        3. Momentum signal filtering
        4. Volatility targeting
        5. Position sizing with risk budget
        """
        logger.info(f"Computing allocations | regime={regime} | portfolio=${portfolio_value:,.2f}")

        # Step 1: Get SAA base weights
        base_weights = STRATEGIC_WEIGHTS.copy()

        # Step 2: Apply regime tactical overlay
        regime_mults = REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS['sideways'])
        tactical_weights = {}
        for ac, w in base_weights.items():
            tactical_weights[ac] = w * regime_mults.get(ac, 1.0)

        # Normalize tactical weights
        total_tactical = sum(tactical_weights.values())
        if total_tactical > 0:
            tactical_weights = {k: v / total_tactical for k, v in tactical_weights.items()}

        # Step 3: Get momentum-eligible symbols per asset class
        # Uses get_top_n_per_class() which returns {asset_class: [symbols]}
        eligible = momentum_engine.get_top_n_per_class() if momentum_engine else {}

        # Step 4: Volatility targeting - scale total exposure
        vol_scalar = self._compute_vol_scalar(price_data)

        # Step 5: Build final allocations
        allocations = {}
        for ac, class_weight in tactical_weights.items():
            # Apply vol scalar and cap at max class allocation
            adjusted_weight = min(
                class_weight * vol_scalar,
                MAX_CLASS_ALLOCATION.get(ac, 0.25)
            )

            # Get eligible symbols for this class
            class_symbols = eligible.get(ac, [])
            if not class_symbols:
                continue

            # Equal weight within class
            n_symbols = len(class_symbols)
            per_symbol = adjusted_weight / n_symbols

            for symbol in class_symbols:
                # Cap individual position
                position_size = min(per_symbol, MAX_SINGLE_POSITION)
                if position_size >= MIN_POSITION_SIZE:
                    allocations[symbol] = round(position_size, 4)

        # Ensure total doesn't exceed max leverage
        total_alloc = sum(allocations.values())
        if total_alloc > self.max_leverage:
            scale = self.max_leverage / total_alloc
            allocations = {k: round(v * scale, 4) for k, v in allocations.items()}

        # Calculate cash position
        cash_pct = max(0, 1.0 - sum(allocations.values()))

        self.current_allocations = allocations
        self.allocation_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'regime': regime,
            'vol_scalar': round(vol_scalar, 3),
            'total_invested': round(1 - cash_pct, 4),
            'cash_pct': round(cash_pct, 4),
            'n_positions': len(allocations),
            'allocations': allocations.copy(),
        })

        logger.info(
            f"Allocation result: {len(allocations)} positions | "
            f"invested={1-cash_pct:.1%} | cash={cash_pct:.1%} | vol_scalar={vol_scalar:.2f}"
        )
        return allocations

    def _compute_vol_scalar(self, price_data, lookback=30):
        """
        Compute portfolio volatility scalar.
        If realized vol > target, scale down.
        If < target, scale up (max 1.0).
        """
        if not price_data or len(price_data) < 5:
            return 0.5  # Conservative default

        try:
            # Use BTC as proxy for overall market vol
            btc_key = None
            for key in ['BTCUSD', 'BTC/USD', 'BTC']:
                if key in price_data:
                    btc_key = key
                    break

            if btc_key and len(price_data[btc_key]) >= lookback:
                prices = price_data[btc_key][-lookback:]
                returns = np.diff(np.log(prices))
                realized_vol = np.std(returns) * np.sqrt(252)

                if realized_vol > 0:
                    scalar = self.target_vol / realized_vol
                    # Clamp between 0.2 and 1.0 (never leverage)
                    return max(0.2, min(scalar, self.max_leverage))

            return 0.7  # Moderate default if no data
        except Exception as e:
            logger.warning(f"Vol scalar calculation failed: {e}")
            return 0.5

    def compute_rebalance_trades(self, current_positions, target_allocations, portfolio_value):
        """
        Compute trades needed to move from current to target allocations.
        Returns list of (symbol, action, notional_amount).
        """
        trades = []

        # Current position weights
        current_weights = {}
        for symbol, pos in current_positions.items():
            if portfolio_value > 0:
                current_weights[symbol] = pos.get('market_value', 0) / portfolio_value

        all_symbols = set(list(current_weights.keys()) + list(target_allocations.keys()))

        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0)
            target_w = target_allocations.get(symbol, 0)
            diff = target_w - current_w

            # Only trade if difference exceeds threshold (reduce turnover)
            if abs(diff) < 0.01:  # 1% threshold
                continue

            notional = abs(diff) * portfolio_value
            action = 'buy' if diff > 0 else 'sell'

            trades.append({
                'symbol': symbol,
                'action': action,
                'weight_change': round(diff, 4),
                'notional': round(notional, 2),
            })

        # Sort: sells first (free up capital), then buys
        trades.sort(key=lambda t: (0 if t['action'] == 'sell' else 1, -t['notional']))

        logger.info(f"Rebalance: {len(trades)} trades needed")
        return trades

    def get_regime_cash_floor(self, regime):
        """Minimum cash percentage by regime."""
        floors = {
            'uptrend': 0.05,
            'sideways': 0.20,
            'stress': 0.40,
        }
        return floors.get(regime, 0.25)

    def should_rebalance(self, current_positions, portfolio_value, threshold=0.05):
        """
        Check if rebalance is needed based on drift from target.
        Uses 5% absolute drift threshold (institutional standard).
        """
        if not self.current_allocations:
            return True

        for symbol, target_w in self.current_allocations.items():
            current_val = current_positions.get(symbol, {}).get('market_value', 0)
            current_w = current_val / portfolio_value if portfolio_value > 0 else 0
            if abs(current_w - target_w) > threshold:
                return True

        return False

    def get_summary(self):
        """Get allocation summary for reporting."""
        if not self.current_allocations:
            return {'status': 'no_allocations', 'n_positions': 0}

        total = sum(self.current_allocations.values())
        return {
            'n_positions': len(self.current_allocations),
            'total_invested_pct': round(total, 4),
            'cash_pct': round(1 - total, 4),
            'allocations': self.current_allocations.copy(),
            'last_rebalance': self.allocation_history[-1] if self.allocation_history else None,
        }

"""
Execution Engine - Smart order execution for Alpaca.
Handles order routing, position management, and rebalancing.
Integrates with AllocationEngine for target-based execution.
"""
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Smart execution engine for systematic trading."""

    def __init__(self, trade_client, min_order_value=1.0, max_slippage_pct=0.005):
        self.client = trade_client
        self.min_order_value = min_order_value
        self.max_slippage = max_slippage_pct
        self.execution_log = []
        self.pending_orders = []

    def execute_rebalance(self, trades, prices, dry_run=False):
        """
        Execute a list of rebalance trades from AllocationEngine.
        Sells first to free capital, then buys.
        """
        results = []
        sells = [t for t in trades if t['action'] == 'sell']
        buys = [t for t in trades if t['action'] == 'buy']

        # Execute sells first
        for trade in sells:
            result = self._execute_single(trade, prices, dry_run)
            results.append(result)

        # Small delay between sells and buys for settlement
        if sells and buys and not dry_run:
            time.sleep(2)

        # Execute buys
        for trade in buys:
            result = self._execute_single(trade, prices, dry_run)
            results.append(result)

        summary = {
            'total_trades': len(results),
            'executed': sum(1 for r in results if r.get('status') == 'filled'),
            'failed': sum(1 for r in results if r.get('status') == 'error'),
            'skipped': sum(1 for r in results if r.get('status') == 'skipped'),
            'trades': results,
        }

        logger.info(
            f"Rebalance complete: {summary['executed']}/{summary['total_trades']} executed, "
            f"{summary['failed']} failed, {summary['skipped']} skipped"
        )
        return summary

    def _execute_single(self, trade, prices, dry_run=False):
        """Execute a single trade."""
        symbol = trade['symbol']
        action = trade['action']
        notional = trade.get('notional', 0)

        # Get current price
        price = prices.get(symbol, 0)
        if price <= 0:
            return {'symbol': symbol, 'status': 'error', 'reason': 'no_price'}

        # Skip small orders
        if notional < self.min_order_value:
            return {'symbol': symbol, 'status': 'skipped', 'reason': 'below_minimum'}

        qty = notional / price

        if dry_run:
            logger.info(f"[DRY RUN] {action.upper()} {qty:.6f} {symbol} (~${notional:.2f})")
            return {'symbol': symbol, 'status': 'dry_run', 'qty': qty, 'notional': notional}

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            side = OrderSide.BUY if action == 'buy' else OrderSide.SELL
            sym = symbol.replace('/', '')

            # For sells, check we have enough position
            if action == 'sell':
                try:
                    pos = self.client.get_open_position(sym)
                    available = float(pos.qty)
                    qty = min(qty, available)
                except Exception:
                    return {'symbol': symbol, 'status': 'error', 'reason': 'no_position'}

            req = MarketOrderRequest(
                symbol=sym,
                qty=round(qty, 6),
                side=side,
                time_in_force=TimeInForce.GTC
            )
            order = self.client.submit_order(req)

            result = {
                'symbol': symbol,
                'status': 'filled',
                'action': action,
                'qty': round(qty, 6),
                'notional': round(notional, 2),
                'order_id': str(order.id),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }

            self.execution_log.append(result)
            logger.info(f"{action.upper()} {qty:.6f} {symbol} (~${notional:.2f}) order={order.id}")
            return result

        except Exception as e:
            logger.error(f"Execution error {symbol}: {e}")
            return {'symbol': symbol, 'status': 'error', 'reason': str(e)}

    def close_position(self, symbol):
        """Close entire position for a symbol."""
        sym = symbol.replace('/', '')
        try:
            self.client.close_position(sym)
            logger.info(f"Closed position: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Close position error {symbol}: {e}")
            return False

    def close_all_positions(self):
        """Emergency: close all positions."""
        try:
            self.client.close_all_positions(cancel_orders=True)
            logger.warning("ALL POSITIONS CLOSED")
            return True
        except Exception as e:
            logger.error(f"Close all error: {e}")
            return False

    def get_current_positions(self):
        """Get current positions as dict for allocation comparison."""
        positions = {}
        try:
            for pos in self.client.get_all_positions():
                positions[pos.symbol] = {
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'unrealized_pnl_pct': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'avg_entry': float(pos.avg_entry_price),
                }
        except Exception as e:
            logger.error(f"Get positions error: {e}")
        return positions

    def get_execution_summary(self):
        """Get execution statistics."""
        if not self.execution_log:
            return {'total_executions': 0}

        buys = [e for e in self.execution_log if e.get('action') == 'buy']
        sells = [e for e in self.execution_log if e.get('action') == 'sell']

        return {
            'total_executions': len(self.execution_log),
            'buys': len(buys),
            'sells': len(sells),
            'total_buy_notional': sum(e.get('notional', 0) for e in buys),
            'total_sell_notional': sum(e.get('notional', 0) for e in sells),
            'last_execution': self.execution_log[-1] if self.execution_log else None,
        }

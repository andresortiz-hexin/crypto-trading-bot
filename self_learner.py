"""V2 Self-Learning Engine.
Tracks trades, benchmarks performance, and generates reports.
V2 changes: Removed dangerous auto-parameter mutation.
Parameters are now managed by regime_engine and risk_engine.
This module focuses on analytics and reporting only.
"""
import os
import time
import json
import logging
import requests
import numpy as np
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

DATA_FILE = '/tmp/learner_state.json'


class SelfLearner:
    """V2 Analytics and learning engine.
    Key change from v4: No longer mutates trading parameters in production.
    Instead, tracks performance metrics and generates actionable reports.
    """

    def __init__(self):
        self.trades = []
        self.daily_returns = []
        self.benchmark_data = {}
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self.best_sharpe = -999
        self.regime = 'unknown'
        self.load_state()

    def load_state(self):
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r') as f:
                    data = json.load(f)
                self.trades = data.get('trades', [])
                self.daily_returns = data.get('daily_returns', [])
                self.win_count = data.get('win_count', 0)
                self.loss_count = data.get('loss_count', 0)
                self.total_pnl = data.get('total_pnl', 0.0)
                self.best_sharpe = data.get('best_sharpe', -999)
                self.regime = data.get('regime', 'unknown')
                log.info(f'Loaded learner: {len(self.trades)} trades, regime={self.regime}')
        except Exception as e:
            log.warning(f'Could not load learner state: {e}')

    def save_state(self):
        try:
            data = {
                'trades': self.trades[-500:],
                'daily_returns': self.daily_returns[-90:],
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'total_pnl': self.total_pnl,
                'best_sharpe': self.best_sharpe,
                'regime': self.regime,
            }
            with open(DATA_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            log.warning(f'Could not save learner state: {e}')

    def fetch_benchmarks(self):
        """Fetch benchmark returns for comparison."""
        benchmarks = {}
        try:
            r = requests.get(
                'https://api.coingecko.com/api/v3/simple/price'
                '?ids=bitcoin,ethereum,solana&vs_currencies=usd'
                '&include_24hr_change=true',
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                for coin, key in [('bitcoin', 'BTC'), ('ethereum', 'ETH'), ('solana', 'SOL')]:
                    if coin in data:
                        benchmarks[key] = {
                            '24h': data[coin].get('usd_24h_change', 0) / 100,
                        }
        except Exception as e:
            log.warning(f'Benchmark fetch: {e}')
        try:
            for idx, name in [('%5EGSPC', 'SP500'), ('%5EIXIC', 'NASDAQ')]:
                r = requests.get(
                    f'https://query1.finance.yahoo.com/v8/finance/chart/{idx}?range=5d&interval=1d',
                    timeout=10, headers={'User-Agent': 'Mozilla/5.0'}
                )
                if r.status_code == 200:
                    data = r.json()
                    closes = data.get('chart', {}).get('result', [{}])[0].get(
                        'indicators', {}).get('quote', [{}])[0].get('close', [])
                    closes = [c for c in closes if c is not None]
                    if len(closes) >= 2:
                        benchmarks[name] = {'24h': (closes[-1] - closes[-2]) / closes[-2]}
        except Exception as e:
            log.warning(f'Benchmark fetch indices: {e}')
        self.benchmark_data = benchmarks
        return benchmarks

    def record_trade(self, symbol, side, qty, price, pnl_pct=None):
        """Record a trade for analytics."""
        trade = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'pnl_pct': pnl_pct,
            'regime': self.regime,
        }
        self.trades.append(trade)
        if pnl_pct is not None:
            self.total_pnl += pnl_pct
            if pnl_pct > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
        self.save_state()

    def record_daily_return(self, pnl_pct, portfolio_value):
        """Record daily return for Sharpe calculation."""
        self.daily_returns.append({
            'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'pnl': pnl_pct,
            'portfolio': portfolio_value,
            'regime': self.regime,
        })
        self.save_state()

    def get_win_rate(self):
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.5

    def get_sharpe_ratio(self):
        if len(self.daily_returns) < 3:
            return 0
        returns = [d['pnl'] for d in self.daily_returns]
        avg = np.mean(returns)
        std = np.std(returns)
        return (avg / std) * np.sqrt(365) if std > 0 else 0

    def get_max_drawdown(self):
        """Calculate maximum drawdown from daily returns."""
        if len(self.daily_returns) < 2:
            return 0
        cumulative = 0
        peak = 0
        max_dd = 0
        for d in self.daily_returns:
            cumulative += d['pnl']
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def get_performance_vs_benchmarks(self):
        """Compare bot vs buy-and-hold benchmarks."""
        if len(self.daily_returns) < 2:
            return {}
        bot_return = sum(d['pnl'] for d in self.daily_returns[-7:])
        comparison = {'bot_7d': bot_return}
        for key in ['BTC', 'ETH', 'SP500', 'NASDAQ']:
            bm = self.benchmark_data.get(key, {})
            comparison[f'{key}_7d'] = bm.get('24h', 0) * 7  # Approximate
        comparison['beating_btc'] = bot_return > comparison.get('BTC_7d', 0)
        comparison['beating_sp500'] = bot_return > comparison.get('SP500_7d', 0)
        return comparison

    def get_regime_stats(self):
        """Get win rate breakdown by regime."""
        stats = {}
        for trade in self.trades[-100:]:
            r = trade.get('regime', 'unknown')
            pnl = trade.get('pnl_pct')
            if pnl is None:
                continue
            if r not in stats:
                stats[r] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
            if pnl > 0:
                stats[r]['wins'] += 1
            else:
                stats[r]['losses'] += 1
            stats[r]['total_pnl'] += pnl
        return stats

    def generate_report(self):
        """Generate performance report for Telegram."""
        win_rate = self.get_win_rate()
        sharpe = self.get_sharpe_ratio()
        max_dd = self.get_max_drawdown()
        total_trades = self.win_count + self.loss_count
        perf = self.get_performance_vs_benchmarks()
        regime_stats = self.get_regime_stats()

        report = f'<b>V2 LEARNING REPORT</b>\n'
        report += f'Regime: {self.regime.upper()}\n'
        report += f'Trades: {total_trades} (W:{self.win_count} L:{self.loss_count})\n'
        report += f'Win Rate: {win_rate:.1%}\n'
        report += f'Sharpe: {sharpe:.2f}\n'
        report += f'Max DD: {max_dd:.2%}\n'
        report += f'Total PnL: {self.total_pnl:.2%}\n'

        if perf:
            report += f'\n<b>vs Benchmarks (7d)</b>\n'
            report += f'Bot: {perf.get("bot_7d", 0):.2%}\n'
            for key in ['BTC', 'ETH', 'SP500', 'NASDAQ']:
                val = perf.get(f'{key}_7d', 0)
                icon = '+' if perf.get('bot_7d', 0) > val else '-'
                report += f'{icon} {key}: {val:.2%}\n'

        if regime_stats:
            report += f'\n<b>By Regime</b>\n'
            for r, s in regime_stats.items():
                total = s['wins'] + s['losses']
                wr = s['wins'] / total if total > 0 else 0
                report += f'{r}: {wr:.0%} WR ({total} trades, PnL={s["total_pnl"]:.2%})\n'

        return report

    def run_learning_cycle(self):
        """Run analytics cycle: fetch benchmarks, generate report.
        V2: No longer mutates parameters. Analytics only."""
        log.info('=== LEARNING CYCLE ===')
        self.fetch_benchmarks()
        sharpe = self.get_sharpe_ratio()
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            log.info(f'New best Sharpe: {sharpe:.2f}')
        report = self.generate_report()
        self.save_state()
        log.info('Learning cycle complete')
        return {}, report

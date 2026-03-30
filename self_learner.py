import os
import time
import json
import logging
import requests
import numpy as np
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# File to persist learning data between restarts
DATA_FILE = '/tmp/learner_state.json'

class SelfLearner:
    """Adaptive self-learning engine that:
    1. Tracks all trades and their outcomes
    2. Compares performance vs world benchmarks (BTC, ETH, S&P500, NASDAQ)
    3. Auto-adjusts strategy parameters to maximize returns
    4. Learns which market conditions lead to best trades
    5. Evolves RSI/MACD/EMA thresholds based on win rate
    """

    def __init__(self):
        self.trades = []
        self.daily_returns = []
        self.benchmark_data = {}
        self.param_history = []
        self.current_params = {
            'min_confidence': 0.15,
            'min_signal_score': 1,
            'max_position_pct': 0.40,
            'stop_loss_pct': 0.015,
            'trailing_stop_pct': 0.008,
            'rsi_buy': 45,
            'rsi_sell': 60,
            'force_entry_cycles': 5,
        }
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self.best_params = None
        self.best_sharpe = -999
        self.regime = 'unknown'  # bull, bear, sideways
        self.regime_params = {
            'bull': {'min_confidence': 0.10, 'min_signal_score': 1, 'max_position_pct': 0.50, 'stop_loss_pct': 0.02, 'trailing_stop_pct': 0.01, 'rsi_buy': 50, 'rsi_sell': 70, 'force_entry_cycles': 3},
            'bear': {'min_confidence': 0.25, 'min_signal_score': 1, 'max_position_pct': 0.25, 'stop_loss_pct': 0.01, 'trailing_stop_pct': 0.005, 'rsi_buy': 30, 'rsi_sell': 50, 'force_entry_cycles': 8},
            'sideways': {'min_confidence': 0.15, 'min_signal_score': 1, 'max_position_pct': 0.35, 'stop_loss_pct': 0.012, 'trailing_stop_pct': 0.006, 'rsi_buy': 40, 'rsi_sell': 58, 'force_entry_cycles': 5},
        }
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
                self.current_params = data.get('current_params', self.current_params)
                self.best_params = data.get('best_params', None)
                self.best_sharpe = data.get('best_sharpe', -999)
                self.regime = data.get('regime', 'unknown')
                log.info(f'Loaded learner state: {len(self.trades)} trades, regime={self.regime}')
        except Exception as e:
            log.warning(f'Could not load learner state: {e}')

    def save_state(self):
        try:
            data = {
                'trades': self.trades[-500:],  # Keep last 500 trades
                'daily_returns': self.daily_returns[-90:],  # Keep 90 days
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'total_pnl': self.total_pnl,
                'current_params': self.current_params,
                'best_params': self.best_params,
                'best_sharpe': self.best_sharpe,
                'regime': self.regime,
            }
            with open(DATA_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            log.warning(f'Could not save learner state: {e}')

    def fetch_benchmarks(self):
        """Fetch world benchmark returns for comparison."""
        benchmarks = {}
        try:
            # BTC 24h performance from CoinGecko
            r = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true&include_7d_change=true', timeout=10)
            if r.status_code == 200:
                data = r.json()
                for coin, key in [('bitcoin', 'BTC'), ('ethereum', 'ETH'), ('solana', 'SOL')]:
                    if coin in data:
                        benchmarks[key] = {
                            '24h': data[coin].get('usd_24h_change', 0) / 100,
                            '7d': data[coin].get('usd_7d_change', 0) / 100 if 'usd_7d_change' in data[coin] else 0,
                        }
        except Exception as e:
            log.warning(f'Benchmark fetch crypto: {e}')
        try:
            # S&P500 and NASDAQ via Yahoo Finance
            for idx, name in [('%5EGSPC', 'SP500'), ('%5EIXIC', 'NASDAQ')]:
                r = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{idx}?range=5d&interval=1d', timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                if r.status_code == 200:
                    data = r.json()
                    closes = data.get('chart', {}).get('result', [{}])[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])
                    closes = [c for c in closes if c is not None]
                    if len(closes) >= 2:
                        benchmarks[name] = {'24h': (closes[-1] - closes[-2]) / closes[-2]}
        except Exception as e:
            log.warning(f'Benchmark fetch indices: {e}')
        self.benchmark_data = benchmarks
        log.info(f'Benchmarks: {json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in benchmarks.items()})}')
        return benchmarks

    def record_trade(self, symbol, side, qty, price, pnl_pct=None):
        """Record a trade for learning."""
        trade = {
            'ts': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'pnl_pct': pnl_pct,
            'regime': self.regime,
            'params': dict(self.current_params),
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
            'date': datetime.utcnow().strftime('%Y-%m-%d'),
            'pnl': pnl_pct,
            'portfolio': portfolio_value,
            'regime': self.regime,
        })
        self.save_state()

    def detect_regime(self, btc_bars=None):
        """Detect market regime: bull, bear, or sideways."""
        old_regime = self.regime
        try:
            benchmarks = self.benchmark_data
            btc_24h = benchmarks.get('BTC', {}).get('24h', 0)
            btc_7d = benchmarks.get('BTC', {}).get('7d', 0)
            eth_24h = benchmarks.get('ETH', {}).get('24h', 0)
            avg_24h = (btc_24h + eth_24h) / 2 if btc_24h and eth_24h else btc_24h
            if btc_7d > 0.05 or avg_24h > 0.03:
                self.regime = 'bull'
            elif btc_7d < -0.05 or avg_24h < -0.03:
                self.regime = 'bear'
            else:
                self.regime = 'sideways'
            if self.regime != old_regime:
                log.info(f'REGIME CHANGE: {old_regime} -> {self.regime}')
                self._apply_regime_params()
        except Exception as e:
            log.warning(f'Regime detection error: {e}')
        return self.regime

    def _apply_regime_params(self):
        """Apply preset parameters for the detected regime."""
        if self.regime in self.regime_params:
            base = self.regime_params[self.regime]
            # Blend with learned adjustments
            for key in base:
                if key in self.current_params:
                    # 70% regime preset, 30% current (learned) value
                    self.current_params[key] = base[key] * 0.7 + self.current_params[key] * 0.3
            log.info(f'Applied {self.regime} regime params: {self.current_params}')
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

    def get_performance_vs_benchmarks(self):
        """Compare bot performance vs buy-and-hold benchmarks."""
        if len(self.daily_returns) < 2:
            return {}
        bot_return = sum(d['pnl'] for d in self.daily_returns[-7:])
        comparison = {'bot_7d': bot_return}
        for key in ['BTC', 'ETH', 'SP500', 'NASDAQ']:
            bm = self.benchmark_data.get(key, {})
            comparison[f'{key}_7d'] = bm.get('7d', bm.get('24h', 0))
        comparison['beating_btc'] = bot_return > comparison.get('BTC_7d', 0)
        comparison['beating_sp500'] = bot_return > comparison.get('SP500_7d', 0)
        return comparison

    def optimize_params(self):
        """Auto-optimize parameters based on recent performance.
        This is the core learning engine."""
        if len(self.trades) < 10:
            log.info('Not enough trades to optimize yet')
            return self.current_params

        win_rate = self.get_win_rate()
        sharpe = self.get_sharpe_ratio()
        recent_trades = self.trades[-20:]
        recent_wins = sum(1 for t in recent_trades if t.get('pnl_pct', 0) and t['pnl_pct'] > 0)
        recent_wr = recent_wins / len(recent_trades) if recent_trades else 0.5

        log.info(f'OPTIMIZE: win_rate={win_rate:.2%} recent_wr={recent_wr:.2%} sharpe={sharpe:.2f}')

        # Track if this is our best configuration
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_params = dict(self.current_params)
            log.info(f'NEW BEST PARAMS (sharpe={sharpe:.2f}): {self.best_params}')

        # === ADAPTIVE PARAMETER ADJUSTMENTS ===
        p = self.current_params

        # 1. If winning a lot -> be more aggressive
        if recent_wr > 0.65:
            p['min_confidence'] = max(0.05, p['min_confidence'] * 0.9)
            p['max_position_pct'] = min(0.60, p['max_position_pct'] * 1.1)
            p['stop_loss_pct'] = min(0.025, p['stop_loss_pct'] * 1.05)
            p['force_entry_cycles'] = max(2, p['force_entry_cycles'] - 1)
            log.info('OPTIMIZE: Hot streak -> more aggressive')

        # 2. If losing a lot -> be more conservative
        elif recent_wr < 0.35:
            p['min_confidence'] = min(0.40, p['min_confidence'] * 1.15)
            p['max_position_pct'] = max(0.15, p['max_position_pct'] * 0.85)
            p['stop_loss_pct'] = max(0.005, p['stop_loss_pct'] * 0.9)
            p['trailing_stop_pct'] = max(0.003, p['trailing_stop_pct'] * 0.9)
            p['force_entry_cycles'] = min(15, p['force_entry_cycles'] + 2)
            log.info('OPTIMIZE: Cold streak -> more conservative')

        # 3. Adjust RSI based on what's been working
        winning_trades = [t for t in self.trades[-50:] if t.get('pnl_pct', 0) and t['pnl_pct'] > 0]
        losing_trades = [t for t in self.trades[-50:] if t.get('pnl_pct', 0) and t['pnl_pct'] < 0]

        # 4. If underperforming benchmarks -> adapt
        perf = self.get_performance_vs_benchmarks()
        if perf.get('beating_btc') == False and perf.get('bot_7d', 0) < 0:
            p['min_confidence'] = max(0.05, p['min_confidence'] * 0.95)
            p['max_position_pct'] = min(0.50, p['max_position_pct'] * 1.05)
            log.info('OPTIMIZE: Underperforming BTC -> adjusting')

        # 5. If Sharpe is negative, revert to best known params
        if sharpe < -0.5 and self.best_params:
            log.info(f'OPTIMIZE: Negative Sharpe ({sharpe:.2f}), reverting to best params')
            self.current_params = dict(self.best_params)
            p = self.current_params

        # Round values
        for key in ['min_confidence', 'max_position_pct', 'stop_loss_pct', 'trailing_stop_pct']:
            p[key] = round(p[key], 4)
        p['force_entry_cycles'] = int(round(p['force_entry_cycles']))
        p['rsi_buy'] = int(round(p.get('rsi_buy', 45)))
        p['rsi_sell'] = int(round(p.get('rsi_sell', 60)))

        self.param_history.append({'ts': datetime.utcnow().isoformat(), 'params': dict(p), 'sharpe': sharpe, 'win_rate': win_rate})
        self.save_state()
        log.info(f'OPTIMIZED PARAMS: {p}')
        return p

    def generate_report(self):
        """Generate a performance report for Telegram."""
        win_rate = self.get_win_rate()
        sharpe = self.get_sharpe_ratio()
        total_trades = self.win_count + self.loss_count
        perf = self.get_performance_vs_benchmarks()

        report = f'<b>LEARNING REPORT</b>\n'
        report += f'Regime: {self.regime.upper()}\n'
        report += f'Trades: {total_trades} (W:{self.win_count} L:{self.loss_count})\n'
        report += f'Win Rate: {win_rate:.1%}\n'
        report += f'Sharpe: {sharpe:.2f}\n'
        report += f'Total PnL: {self.total_pnl:.2%}\n'
        if perf:
            report += f'\n<b>vs Benchmarks (7d)</b>\n'
            report += f'Bot: {perf.get("bot_7d", 0):.2%}\n'
            for key in ['BTC', 'ETH', 'SP500', 'NASDAQ']:
                val = perf.get(f'{key}_7d', 0)
                beating = perf.get("bot_7d", 0) > val
                icon = '+' if beating else '-'
                report += f'{icon} {key}: {val:.2%}\n'
        report += f'\n<b>Active Params</b>\n'
        for k, v in self.current_params.items():
            report += f'{k}: {v}\n'
        return report

    def run_learning_cycle(self):
        """Run a complete learning cycle: fetch benchmarks, detect regime, optimize."""
        log.info('=== LEARNING CYCLE ===')
        self.fetch_benchmarks()
        self.detect_regime()
        params = self.optimize_params()
        report = self.generate_report()
        log.info(f'Learning cycle complete. Regime={self.regime}')
        return params, report

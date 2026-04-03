"""
Universe - Asset universe definition and data fetching
V3 Hexin Systematic Global Allocation Strategy
April 2026

Defines the full multi-asset universe and provides unified data access.
"""

import logging
import pandas as pd
import numpy as np
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

log = logging.getLogger(__name__)

# Asset class definitions
EQUITY = 'equity'
FIXED_INCOME = 'fixed_income'
COMMODITY = 'commodity'
CRYPTO = 'crypto'

# Full universe with metadata
UNIVERSE = {
    # EQUITY (40% strategic)
    'SPY':     {'class': EQUITY, 'name': 'S&P 500', 'strategic_weight': 0.10},
    'QQQ':     {'class': EQUITY, 'name': 'Nasdaq 100', 'strategic_weight': 0.08},
    'IWM':     {'class': EQUITY, 'name': 'Russell 2000', 'strategic_weight': 0.05},
    'EFA':     {'class': EQUITY, 'name': 'MSCI EAFE', 'strategic_weight': 0.05},
    'EEM':     {'class': EQUITY, 'name': 'MSCI EM', 'strategic_weight': 0.04},
    'XLF':     {'class': EQUITY, 'name': 'Financials', 'strategic_weight': 0.03},
    'XLE':     {'class': EQUITY, 'name': 'Energy', 'strategic_weight': 0.03},
    'XLK':     {'class': EQUITY, 'name': 'Technology', 'strategic_weight': 0.02},
    # FIXED INCOME (20% strategic)
    'TLT':     {'class': FIXED_INCOME, 'name': '20Y+ Treasury', 'strategic_weight': 0.08},
    'IEF':     {'class': FIXED_INCOME, 'name': '7-10Y Treasury', 'strategic_weight': 0.05},
    'HYG':     {'class': FIXED_INCOME, 'name': 'High Yield', 'strategic_weight': 0.04},
    'TIP':     {'class': FIXED_INCOME, 'name': 'TIPS', 'strategic_weight': 0.03},
    # COMMODITIES (15% strategic)
    'GLD':     {'class': COMMODITY, 'name': 'Gold', 'strategic_weight': 0.06},
    'SLV':     {'class': COMMODITY, 'name': 'Silver', 'strategic_weight': 0.03},
    'USO':     {'class': COMMODITY, 'name': 'Crude Oil', 'strategic_weight': 0.03},
    'DBA':     {'class': COMMODITY, 'name': 'Agriculture', 'strategic_weight': 0.03},
    # CRYPTO (15% strategic)
    'BTC/USD': {'class': CRYPTO, 'name': 'Bitcoin', 'strategic_weight': 0.08},
    'ETH/USD': {'class': CRYPTO, 'name': 'Ethereum', 'strategic_weight': 0.04},
    'SOL/USD': {'class': CRYPTO, 'name': 'Solana', 'strategic_weight': 0.03},
}

# Asset class limits
CLASS_LIMITS = {
    EQUITY:       {'strategic': 0.40, 'max': 0.50},
    FIXED_INCOME: {'strategic': 0.20, 'max': 0.30},
    COMMODITY:    {'strategic': 0.15, 'max': 0.20},
    CRYPTO:       {'strategic': 0.15, 'max': 0.20},
}

# Position limits
MAX_SINGLE_POSITION = 0.15
MAX_SINGLE_CRYPTO = 0.08
MAX_SINGLE_COMMODITY = 0.10
MIN_CASH = 0.10


def is_crypto(symbol):
    return '/' in symbol


def get_asset_class(symbol):
    return UNIVERSE.get(symbol, {}).get('class', EQUITY)


def get_symbols_by_class(asset_class):
    return [s for s, m in UNIVERSE.items() if m['class'] == asset_class]


def get_all_symbols():
    return list(UNIVERSE.keys())


def get_stock_symbols():
    return [s for s in UNIVERSE if not is_crypto(s)]


def get_crypto_symbols():
    return [s for s in UNIVERSE if is_crypto(s)]


def get_max_position(symbol):
    """Get max position size for a symbol."""

    def get_tradeable_symbols():
    """Get all symbols with metadata for trading."""
    return [{'symbol': s, 'asset_class': m['class'], 'name': m['name']} for s, m in UNIVERSE.items()]
    ac = get_asset_class(symbol)
    if ac == CRYPTO:
        return MAX_SINGLE_CRYPTO
    elif ac == COMMODITY:
        return MAX_SINGLE_COMMODITY
    return MAX_SINGLE_POSITION


class UniverseDataFetcher:
    """Fetches and normalizes bar data for the full universe."""

    def __init__(self, crypto_client, stock_client):
        self.crypto_client = crypto_client
        self.stock_client = stock_client
        self._cache = {}  # symbol -> DataFrame

    def _normalize(self, df):
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

    def fetch(self, symbol, timeframe=TimeFrame.Day, limit=252):
        """Fetch bars for a single symbol."""
        try:
            if is_crypto(symbol):
                req = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    limit=limit
                )
                raw = self.crypto_client.get_crypto_bars(req).df
            else:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    limit=limit
                )
                raw = self.stock_client.get_stock_bars(req).df
            return self._normalize(raw)
        except Exception as e:
            log.error(f'fetch({symbol}): {e}')
            return None

    def fetch_all(self, timeframe=TimeFrame.Day, limit=252):
        """Fetch bars for all symbols in the universe. Returns dict."""
        data = {}
        for sym in get_all_symbols():
            bars = self.fetch(sym, timeframe, limit)
            if bars is not None and len(bars) >= 50:
                data[sym] = bars
            else:
                log.warning(f'Insufficient data for {sym}')
        self._cache = data
        log.info(f'Fetched data for {len(data)}/{len(UNIVERSE)} symbols')
        return data

    def get_closes(self, data=None):
        """Get a DataFrame of close prices for all symbols."""
        d = data or self._cache
        closes = {}
        for sym, bars in d.items():
            closes[sym] = bars['close'].values
        if not closes:
            return pd.DataFrame()
        max_len = max(len(v) for v in closes.values())
        aligned = {}
        for sym, vals in closes.items():
            if len(vals) < max_len:
                padded = np.full(max_len, np.nan)
                padded[-len(vals):] = vals
                aligned[sym] = padded
            else:
                aligned[sym] = vals
        return pd.DataFrame(aligned)

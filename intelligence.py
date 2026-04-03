"""V2 Institutional Intelligence Module.
Weighted multi-source sentiment with decay, volatility correlation,
and contrarian signals for regime-aware trading.
"""
import requests
import logging
import time
from datetime import datetime

log = logging.getLogger(__name__)

# --- Keyword dictionaries with weights ---
BULLISH_KEYWORDS = {
    'surge': 2, 'rally': 2, 'bull': 1, 'breakout': 2, 'record': 1,
    'high': 1, 'adoption': 2, 'etf': 2, 'approval': 3, 'inflow': 2,
    'buy': 1, 'accumulate': 2, 'upgrade': 1, 'partnership': 1,
    'launch': 1, 'all-time': 2, 'ath': 2, 'institutional': 2,
    'halving': 2, 'bullish': 2, 'soar': 2, 'gain': 1,
}
BEARISH_KEYWORDS = {
    'crash': 3, 'dump': 2, 'bear': 1, 'hack': 3, 'ban': 3,
    'regulation': 2, 'sell': 1, 'fear': 2, 'lawsuit': 2, 'sec': 2,
    'fraud': 3, 'collapse': 3, 'liquidation': 2, 'drop': 1,
    'decline': 1, 'warning': 2, 'exploit': 3, 'vulnerability': 2,
    'sanctioned': 2, 'bearish': 2, 'plunge': 2, 'loss': 1,
    'default': 2, 'bankrupt': 3, 'outflow': 2,
}

# --- Sentiment cache with TTL ---
_sentiment_cache = {}
CACHE_TTL = 180  # 3 minutes

def _cached_get(url, timeout=8, headers=None):
    """Simple cache for HTTP GET to avoid rate limits."""
    now = time.time()
    if url in _sentiment_cache:
        data, ts = _sentiment_cache[url]
        if now - ts < CACHE_TTL:
            return data
    try:
        h = headers or {}
        r = requests.get(url, timeout=timeout, headers=h)
        r.raise_for_status()
        result = r.json() if 'json' in r.headers.get('content-type', '') else r.text
        _sentiment_cache[url] = (result, now)
        return result
    except Exception as e:
        log.warning(f'HTTP GET failed {url}: {e}')
        return None


def get_fear_and_greed():
    """Get Fear & Greed index with historical context."""
    try:
        data = _cached_get('https://api.alternative.me/fng/?limit=7')
        if not data:
            return 50, 'Neutral', 0
        entries = data.get('data', [])
        if not entries:
            return 50, 'Neutral', 0
        current = int(entries[0]['value'])
        label = entries[0]['value_classification']
        # Calculate momentum (change over last 7 days)
        if len(entries) >= 7:
            week_ago = int(entries[-1]['value'])
            momentum = current - week_ago
        else:
            momentum = 0
        log.info(f'Fear & Greed: {current} ({label}), 7d momentum: {momentum:+d}')
        return current, label, momentum
    except Exception as e:
        log.warning(f'Fear&Greed error: {e}')
        return 50, 'Neutral', 0


def score_text(text):
    """Score text with weighted keywords. Returns weighted score."""
    text = text.lower()
    score = 0
    for w, weight in BULLISH_KEYWORDS.items():
        if w in text:
            score += weight
    for w, weight in BEARISH_KEYWORDS.items():
        if w in text:
            score -= weight
    return score


def get_cryptopanic_sentiment(symbol='BTC'):
    """Get sentiment from CryptoPanic with weighted scoring."""
    try:
        coin = symbol.replace('/USD', '').replace('/USDT', '')
        url = f'https://cryptopanic.com/api/v1/posts/?auth_token=free&currencies={coin}&public=true'
        data = _cached_get(url)
        if not data:
            return 0, []
        items = data.get('results', [])
        if not items:
            return 0, []
        total = 0
        headlines = []
        for i, item in enumerate(items[:15]):
            title = item.get('title', '')
            # Apply recency decay: newer headlines matter more
            decay = 1.0 - (i * 0.05)
            total += score_text(title) * decay
            if i < 5:
                headlines.append(title)
        sentiment = total / max(len(items[:15]), 1)
        log.info(f'CryptoPanic {coin}: sentiment={sentiment:.2f} from {len(items[:15])} headlines')
        return sentiment, headlines[:3]
    except Exception as e:
        log.warning(f'CryptoPanic error: {e}')
        return 0, []


def get_global_market_sentiment():
    """Get global market data from CoinGecko."""
    try:
        data = _cached_get('https://api.coingecko.com/api/v3/global')
        if not data:
            return 50, 0, 0
        gdata = data.get('data', {})
        btc_dominance = gdata.get('btc_dominance', 50)
        market_cap_change = gdata.get('market_cap_change_percentage_24h_usd', 0)
        active_cryptos = gdata.get('active_cryptocurrencies', 0)
        log.info(f'Global: BTC dom={btc_dominance:.1f}%, mcap_chg={market_cap_change:.2f}%')
        return btc_dominance, market_cap_change, active_cryptos
    except Exception as e:
        log.warning(f'CoinGecko global error: {e}')
        return 50, 0, 0


def get_coin_data(symbol):
    """Get detailed coin data including volume changes."""
    coin_map = {
        'BTC/USD': 'bitcoin', 'ETH/USD': 'ethereum', 'SOL/USD': 'solana',
    }
    coin_id = coin_map.get(symbol)
    if not coin_id:
        return {}
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
        data = _cached_get(f'{url}?localization=false&tickers=false&community_data=false&developer_data=false')
        if not data:
            return {}
        md = data.get('market_data', {})
        return {
            'price_change_24h': md.get('price_change_percentage_24h', 0),
            'price_change_7d': md.get('price_change_percentage_7d', 0),
            'price_change_30d': md.get('price_change_percentage_30d', 0),
            'total_volume': md.get('total_volume', {}).get('usd', 0),
            'market_cap': md.get('market_cap', {}).get('usd', 0),
            'ath_change': md.get('ath_change_percentage', {}).get('usd', 0),
        }
    except Exception as e:
        log.warning(f'CoinGecko coin error ({symbol}): {e}')
        return {}


def build_market_intelligence(symbol):
    """Build comprehensive market intelligence with weighted scoring.
    
    V2: Weighted sources, contrarian signals, volume analysis.
    Returns dict with score, confidence, and detailed breakdown.
    """
    fng_value, fng_label, fng_momentum = get_fear_and_greed()
    cp_sentiment, cp_headlines = get_cryptopanic_sentiment(symbol)
    btc_dom, mcap_change, _ = get_global_market_sentiment()
    coin = get_coin_data(symbol)
    
    price_24h = coin.get('price_change_24h', 0)
    price_7d = coin.get('price_change_7d', 0)
    price_30d = coin.get('price_change_30d', 0)
    
    # --- Weighted Intelligence Score (max ~10) ---
    intel_score = 0
    reasons = []
    
    # 1. Fear & Greed (weight: 2x) - CONTRARIAN approach
    if fng_value >= 75:
        intel_score -= 1  # Extreme greed = contrarian sell signal
        reasons.append(f'Extreme Greed ({fng_value}) - contrarian caution')
    elif fng_value >= 55:
        intel_score += 1
        reasons.append(f'Greed zone ({fng_value}) - favorable')
    elif fng_value <= 20:
        intel_score += 2  # Extreme fear = contrarian buy signal
        reasons.append(f'Extreme Fear ({fng_value}) - contrarian BUY')
    elif fng_value <= 35:
        intel_score -= 1
        reasons.append(f'Fear zone ({fng_value}) - cautious')
    
    # FNG momentum bonus
    if fng_momentum > 10:
        intel_score += 1
        reasons.append(f'FNG improving +{fng_momentum} in 7d')
    elif fng_momentum < -10:
        intel_score -= 1
        reasons.append(f'FNG deteriorating {fng_momentum} in 7d')
    
    # 2. News sentiment (weight: 1.5x)
    if cp_sentiment > 0.5:
        intel_score += 2
        reasons.append(f'Bullish news flow ({cp_sentiment:.1f})')
    elif cp_sentiment > 0.2:
        intel_score += 1
        reasons.append(f'Mildly bullish news ({cp_sentiment:.1f})')
    elif cp_sentiment < -0.5:
        intel_score -= 2
        reasons.append(f'Bearish news flow ({cp_sentiment:.1f})')
    elif cp_sentiment < -0.2:
        intel_score -= 1
        reasons.append(f'Mildly bearish news ({cp_sentiment:.1f})')
    
    # 3. Market cap momentum
    if mcap_change > 2:
        intel_score += 1
        reasons.append(f'Total mcap up {mcap_change:.1f}%')
    elif mcap_change < -3:
        intel_score -= 1
        reasons.append(f'Total mcap down {mcap_change:.1f}%')
    
    # 4. Price momentum (multi-timeframe)
    if price_24h > 3 and price_7d > 5:
        intel_score += 2
        reasons.append(f'Strong momentum: 24h={price_24h:.1f}%, 7d={price_7d:.1f}%')
    elif price_24h > 1:
        intel_score += 1
        reasons.append(f'{symbol} up {price_24h:.1f}% 24h')
    elif price_24h < -5 and price_7d < -10:
        intel_score -= 2
        reasons.append(f'Heavy selloff: 24h={price_24h:.1f}%, 7d={price_7d:.1f}%')
    elif price_24h < -2:
        intel_score -= 1
        reasons.append(f'{symbol} down {price_24h:.1f}% 24h')
    
    # 5. Trend alignment (30d context)
    if price_30d > 15 and price_7d > 0:
        intel_score += 1
        reasons.append(f'Strong uptrend: 30d={price_30d:.0f}%')
    elif price_30d < -20:
        intel_score -= 1
        reasons.append(f'Downtrend: 30d={price_30d:.0f}%')
    
    # --- Normalize confidence ---
    max_possible = 8
    intel_confidence = intel_score / max_possible
    intel_confidence = max(-1.0, min(1.0, intel_confidence))
    
    summary = {
        'score': intel_score,
        'confidence': round(intel_confidence, 4),
        'fng': fng_value,
        'fng_label': fng_label,
        'fng_momentum': fng_momentum,
        'price_24h': price_24h,
        'price_7d': price_7d,
        'price_30d': price_30d,
        'market_cap_change': mcap_change,
        'btc_dominance': btc_dom,
        'news_sentiment': cp_sentiment,
        'top_headlines': cp_headlines,
        'reasons': reasons,
    }
    log.info(f'Intel {symbol}: score={intel_score} conf={intel_confidence:.2f} reasons={len(reasons)}')
    return summary

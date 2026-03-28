import requests
import logging
import os
from datetime import datetime

log = logging.getLogger(__name__)

BULLISH_KEYWORDS = [
    'surge','rally','bull','breakout','record','high','adoption',
    'etf','approval','inflow','buy','accumulate','moon','pump',
    'upgrade','partnership','launch','all-time','ath'
]
BEARISH_KEYWORDS = [
    'crash','dump','bear','hack','ban','regulation','sell','fear',
    'lawsuit','sec','fraud','collapse','liquidation','drop','decline',
    'warning','exploit','vulnerability','sanctioned'
]

def get_fear_and_greed():
    try:
        r = requests.get('https://api.alternative.me/fng/?limit=1', timeout=8)
        data = r.json()['data'][0]
        value = int(data['value'])
        label = data['value_classification']
        log.info(f'Fear & Greed Index: {value} ({label})')
        return value, label
    except Exception as e:
        log.warning(f'Fear&Greed error: {e}')
        return 50, 'Neutral'

def score_text(text):
    text = text.lower()
    score = 0
    for w in BULLISH_KEYWORDS:
        if w in text:
            score += 1
    for w in BEARISH_KEYWORDS:
        if w in text:
            score -= 1
    return score

def get_cryptopanic_sentiment(symbol='BTC'):
    try:
        coin = symbol.replace('/USD','').replace('/USDT','')
        url = f'https://cryptopanic.com/api/v1/posts/?auth_token=free&currencies={coin}&public=true'
        r = requests.get(url, timeout=8)
        items = r.json().get('results', [])
        if not items:
            return 0, []
        total = 0
        headlines = []
        for item in items[:10]:
            title = item.get('title', '')
            total += score_text(title)
            headlines.append(title)
        sentiment = total / max(len(items[:10]), 1)
        log.info(f'CryptoPanic {coin}: sentiment={sentiment:.2f} from {len(headlines)} headlines')
        return sentiment, headlines[:3]
    except Exception as e:
        log.warning(f'CryptoPanic error: {e}')
        return 0, []

def get_rss_sentiment(symbol='BTC'):
    coin = symbol.replace('/USD','').replace('/USDT','')
    feeds = [
        'https://feeds.feedburner.com/CoinDesk',
        'https://cointelegraph.com/rss',
        'https://decrypt.co/feed',
    ]
    total_score = 0
    count = 0
    for feed_url in feeds:
        try:
            r = requests.get(feed_url, timeout=6, headers={'User-Agent': 'Mozilla/5.0'})
            text = r.text.lower()
            coin_lower = coin.lower()
            bitcoin_lower = 'bitcoin' if coin == 'BTC' else coin_lower
            ethereum_lower = 'ethereum' if coin == 'ETH' else coin_lower
            solana_lower = 'solana' if coin == 'SOL' else coin_lower
            if coin_lower in text or bitcoin_lower in text or ethereum_lower in text or solana_lower in text:
                total_score += score_text(text[:5000])
                count += 1
        except Exception as e:
            log.warning(f'RSS feed error ({feed_url}): {e}')
    if count == 0:
        return 0
    result = total_score / count
    log.info(f'RSS sentiment {coin}: {result:.2f} from {count} feeds')
    return result

def get_global_market_sentiment():
    try:
        r = requests.get('https://api.coingecko.com/api/v3/global', timeout=8)
        data = r.json()['data']
        btc_dominance = data.get('btc_dominance', 50)
        market_cap_change = data.get('market_cap_change_percentage_24h_usd', 0)
        log.info(f'Global: BTC dominance={btc_dominance:.1f}%, market_cap_change={market_cap_change:.2f}%')
        return btc_dominance, market_cap_change
    except Exception as e:
        log.warning(f'CoinGecko global error: {e}')
        return 50, 0

def get_coin_sentiment(symbol):
    coin_map = {
        'BTC/USD': 'bitcoin',
        'ETH/USD': 'ethereum',
        'SOL/USD': 'solana',
    }
    coin_id = coin_map.get(symbol)
    if not coin_id:
        return 0, 0
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
        params = {'localization': 'false', 'tickers': 'false', 'community_data': 'true', 'developer_data': 'false'}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        price_change_24h = data.get('market_data', {}).get('price_change_percentage_24h', 0)
        price_change_7d = data.get('market_data', {}).get('price_change_percentage_7d', 0)
        log.info(f'CoinGecko {symbol}: 24h={price_change_24h:.2f}%, 7d={price_change_7d:.2f}%')
        return price_change_24h, price_change_7d
    except Exception as e:
        log.warning(f'CoinGecko coin error ({symbol}): {e}')
        return 0, 0

def build_market_intelligence(symbol):
    fng_value, fng_label = get_fear_and_greed()
    cp_sentiment, cp_headlines = get_cryptopanic_sentiment(symbol)
    rss_score = get_rss_sentiment(symbol)
    btc_dom, mcap_change = get_global_market_sentiment()
    price_24h, price_7d = get_coin_sentiment(symbol)

    intel_score = 0
    reasons = []

    if fng_value >= 60:
        intel_score += 1
        reasons.append(f'Greed sentiment: {fng_value} ({fng_label})')
    elif fng_value <= 25:
        intel_score -= 2
        reasons.append(f'Extreme Fear: {fng_value} — avoiding buy')
    else:
        reasons.append(f'Neutral sentiment: {fng_value} ({fng_label})')

    if cp_sentiment > 0.3:
        intel_score += 1
        reasons.append(f'Bullish news flow (score={cp_sentiment:.1f})')
    elif cp_sentiment < -0.3:
        intel_score -= 1
        reasons.append(f'Bearish news flow (score={cp_sentiment:.1f})')

    if rss_score > 1:
        intel_score += 1
        reasons.append(f'Positive RSS feeds (score={rss_score:.1f})')
    elif rss_score < -1:
        intel_score -= 1
        reasons.append(f'Negative RSS feeds (score={rss_score:.1f})')

    if mcap_change > 1:
        intel_score += 1
        reasons.append(f'Market cap up {mcap_change:.1f}% in 24h')
    elif mcap_change < -2:
        intel_score -= 1
        reasons.append(f'Market cap down {mcap_change:.1f}% in 24h')

    if price_24h > 2:
        intel_score += 1
        reasons.append(f'{symbol} up {price_24h:.1f}% in 24h — momentum')
    elif price_24h < -3:
        intel_score -= 1
        reasons.append(f'{symbol} down {price_24h:.1f}% in 24h — caution')

    intel_confidence = intel_score / 5.0
    intel_confidence = max(-1.0, min(1.0, intel_confidence))

    summary = {
        'score': intel_score,
        'confidence': intel_confidence,
        'fng': fng_value,
        'fng_label': fng_label,
        'price_24h': price_24h,
        'price_7d': price_7d,
        'market_cap_change': mcap_change,
        'btc_dominance': btc_dom,
        'top_headlines': cp_headlines,
        'reasons': reasons,
    }
    return summary

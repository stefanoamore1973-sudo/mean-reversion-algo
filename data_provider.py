"""
data_provider.py v2.0
=====================
Mean Reversion Algoritmo — Modulo dati + indicatori.

Changelog vs v1.0:
- Aggiunto IBS (Internal Bar Strength) — Pagonidis 2013, ~58% win rate boost
- Aggiunto VWAP distance normalizzata (z-score rispetto a ATR)
- Aggiunto Relative Strength vs SPY (5-day return ratio)
- Aggiunto candle pattern detection (hammer, bullish engulfing, doji bottom)
- Aggiunto regime detection: VIX + SPY 20-EMA trend
- Aggiunto gap behavior (open vs prior close normalizzato)
- Caching con st.cache_data per SPY/VIX (risparmia API calls)
- Error handling robusto su ogni feature
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st


# ============================================================================
# TIINGO PROVIDER
# ============================================================================

class TiingoProvider:
    """Fetcher per Tiingo IEX real-time + daily OHLCV."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.tiingo.com"
    
    def get_quote(self, ticker):
        url = f"{self.base_url}/iex/{ticker}?token={self.api_key}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                return data
            else:
                st.error(f"Tiingo quote error {ticker}: HTTP {resp.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching quote {ticker}: {str(e)}")
            return None
    
    def get_intraday_bars(self, ticker, start_date, resample_freq='1min'):
        url = f"{self.base_url}/iex/{ticker}/prices?startDate={start_date}&resampleFreq={resample_freq}&token={self.api_key}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col not in df.columns:
                            df[col] = np.nan
                    return df
                return pd.DataFrame()
            else:
                st.warning(f"Tiingo intraday bars {ticker}: HTTP {resp.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error fetching intraday bars {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_daily_prices(self, ticker, start_date=None, end_date=None):
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        url = f"{self.base_url}/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={self.api_key}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col not in df.columns:
                            df[col] = np.nan
                    return df
                return pd.DataFrame()
            else:
                st.warning(f"Tiingo daily prices {ticker}: HTTP {resp.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error fetching daily prices {ticker}: {str(e)}")
            return pd.DataFrame()


# ============================================================================
# CACHED FETCHERS (for SPY/VIX - shared across tickers in same session)
# ============================================================================

@st.cache_data(ttl=600)  # Cache 10 min
def fetch_spy_daily(api_key, days=45):
    """Fetch SPY daily per regime detection e relative strength."""
    provider = TiingoProvider(api_key)
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    df = provider.get_daily_prices('SPY', start_date=start, end_date=end)
    return df


@st.cache_data(ttl=600)
def fetch_vix_daily(api_key, days=30):
    """Fetch VIX proxy (VIXY ETF) per regime detection.
    Tiingo non ha ^VIX diretto, uso VIXY come proxy ragionevole."""
    provider = TiingoProvider(api_key)
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    df = provider.get_daily_prices('VIXY', start_date=start, end_date=end)
    return df


# ============================================================================
# INDICATOR CALCULATOR
# ============================================================================

class IndicatorCalculator:
    """
    Calcoli indicatori tecnici per Mean Reversion intraday.
    
    Riferimenti empirici:
    - Wilder (1978): RSI formula originale
    - Conrad, Hameed, Niden (1994): Volume e short-term reversal
    - Pagonidis (2013): IBS effect in equity ETFs
    - Madhavan (2002): VWAP come ancora intraday istituzionale
    - Bollinger (2001): Band z-score come misura di stretch
    - Nison (1991): candle pattern reversal
    """
    
    @staticmethod
    def rsi(prices, period=14):
        """RSI Wilder. Range [0,100], <30=oversold, >70=overbought."""
        try:
            if len(prices) < period + 1:
                return np.nan
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
            loss = (-delta).where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except Exception:
            return np.nan
    
    @staticmethod
    def rsi_slope(prices, period=14, lookback=5):
        """RSI slope — rilevamento di inversione momentum."""
        try:
            if len(prices) < period + lookback:
                return np.nan
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
            loss = (-delta).where(delta < 0, 0).rolling(period, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_series = 100 - (100 / (1 + rs))
            recent = rsi_series.iloc[-lookback:]
            if len(recent) < lookback:
                return np.nan
            return (recent.iloc[-1] - recent.iloc[0]) / lookback
        except Exception:
            return np.nan
    
    @staticmethod
    def ibs(df):
        """
        Internal Bar Strength (Pagonidis 2013).
        IBS = (Close - Low) / (High - Low), calcolato sul daily PRECEDENTE.
        
        Interpretazione:
        - IBS < 0.2 → fortissimo segnale bullish MR (avg +0.35% il giorno dopo)
        - IBS > 0.8 → segnale bearish (avg -0.13% il giorno dopo)
        
        Evidenza: un filtro IBS aumenta i returns di una strategia RSI di ~10pp
        su ETF equity (Pagonidis 2013).
        """
        try:
            if len(df) < 2:
                return np.nan
            # Uso la barra più recente disponibile (daily)
            last = df.iloc[-1]
            high, low, close = last['high'], last['low'], last['close']
            if pd.isna(high) or pd.isna(low) or pd.isna(close):
                return np.nan
            if high == low:
                return 0.5  # No range → neutrale
            ibs_val = (close - low) / (high - low)
            return float(np.clip(ibs_val, 0, 1))
        except Exception:
            return np.nan
    
    @staticmethod
    def vwap(df):
        """VWAP cumulativo intraday."""
        try:
            if not all(c in df.columns for c in ['high','low','close','volume']) or len(df) == 0:
                return np.nan
            typical = (df['high'] + df['low'] + df['close']) / 3
            vol_sum = df['volume'].cumsum()
            if vol_sum.iloc[-1] == 0:
                return np.nan
            return ((typical * df['volume']).cumsum() / vol_sum).iloc[-1]
        except Exception:
            return np.nan
    
    @staticmethod
    def vwap_distance_atr(df, current_price, atr_value):
        """
        Distanza da VWAP normalizzata in ATR.
        Restituisce numero di ATR sotto (neg) o sopra (pos) VWAP.
        Setup MR long ottimale: -1.5 < dist < -0.5 ATR.
        """
        try:
            v = IndicatorCalculator.vwap(df)
            if pd.isna(v) or pd.isna(atr_value) or atr_value == 0 or pd.isna(current_price):
                return np.nan
            return (current_price - v) / atr_value
        except Exception:
            return np.nan
    
    @staticmethod
    def bollinger_zscore(prices, period=20):
        """Z-score rispetto a SMA20. |z| > 2 = stretch estremo."""
        try:
            if len(prices) < period:
                return np.nan
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            return (prices.iloc[-1] - sma.iloc[-1]) / (std.iloc[-1] + 1e-8)
        except Exception:
            return np.nan
    
    @staticmethod
    def atr(df, period=14):
        """Average True Range."""
        try:
            if len(df) < period or not all(c in df.columns for c in ['high','low','close']):
                return np.nan
            high, low, close = df['high'], df['low'], df['close']
            tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
            return tr.rolling(period).mean().iloc[-1]
        except Exception:
            return np.nan
    
    @staticmethod
    def adv(df, period=20):
        """Average Dollar Volume su `period` giorni."""
        try:
            if len(df) < period or 'volume' not in df.columns or 'close' not in df.columns:
                return np.nan
            return (df['volume'] * df['close']).tail(period).mean()
        except Exception:
            return np.nan
    
    @staticmethod
    def volume_ratio(df_intraday, adv20):
        """Volume corrente (cumulato oggi) / ADV20 daily.
        Ratio > 1.5 indica partecipazione anomala → conferma panic/accumulation."""
        try:
            if df_intraday is None or len(df_intraday) == 0 or pd.isna(adv20) or adv20 == 0:
                return np.nan
            if 'close' not in df_intraday.columns or 'volume' not in df_intraday.columns:
                return np.nan
            dollar_vol_today = (df_intraday['volume'] * df_intraday['close']).sum()
            return dollar_vol_today / adv20
        except Exception:
            return np.nan
    
    @staticmethod
    def candle_reversal_score(df_intraday, lookback=5):
        """
        Score [0-1] di presenza pattern reversal bullish nelle ultime N candele.
        Rileva: hammer, bullish engulfing, doji bottom.
        Nison (1991) + Bulkowski (2008) empirical patterns.
        """
        try:
            if df_intraday is None or len(df_intraday) < lookback:
                return 0.5  # Neutrale
            recent = df_intraday.tail(lookback).copy()
            if not all(c in recent.columns for c in ['open','high','low','close']):
                return 0.5
            
            score = 0
            max_score = 0
            
            for i in range(len(recent)):
                row = recent.iloc[i]
                o, h, l, c = row['open'], row['high'], row['low'], row['close']
                if any(pd.isna([o,h,l,c])):
                    continue
                body = abs(c - o)
                total_range = h - l
                if total_range == 0:
                    continue
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l
                max_score += 1
                
                # Hammer: lower wick >= 2x body, small upper wick, bullish close
                if body > 0 and lower_wick >= 2 * body and upper_wick <= body * 0.5 and c >= o:
                    score += 1
                    continue
                # Bullish engulfing (needs previous candle)
                if i > 0:
                    prev = recent.iloc[i-1]
                    po, pc = prev['open'], prev['close']
                    if not any(pd.isna([po,pc])):
                        if pc < po and c > o and c > po and o < pc:
                            score += 1
                            continue
                # Doji at bottom of range
                if body / total_range < 0.1 and (c - l) / total_range < 0.3:
                    score += 0.5
            
            if max_score == 0:
                return 0.5
            return min(1.0, score / max_score + 0.2)  # bias leggermente positivo se trovato
        except Exception:
            return 0.5
    
    @staticmethod
    def gap_behavior(df_daily, intraday_open):
        """
        Gap open vs prior close, normalizzato in ATR.
        Gap down >1 ATR in contesto MR può indicare panic → favorisce reversion.
        """
        try:
            if df_daily is None or len(df_daily) < 15 or pd.isna(intraday_open):
                return np.nan
            prior_close = df_daily['close'].iloc[-1]
            atr_val = IndicatorCalculator.atr(df_daily, 14)
            if pd.isna(atr_val) or atr_val == 0 or pd.isna(prior_close):
                return np.nan
            return (intraday_open - prior_close) / atr_val
        except Exception:
            return np.nan
    
    @staticmethod
    def relative_strength(ticker_daily, spy_daily, lookback=5):
        """
        Relative strength ticker vs SPY su `lookback` giorni.
        RS < 0: ticker underperform SPY → potenziale MR candidate.
        RS > 0: outperform → meno MR opportunity.
        """
        try:
            if ticker_daily is None or spy_daily is None:
                return np.nan
            if len(ticker_daily) < lookback + 1 or len(spy_daily) < lookback + 1:
                return np.nan
            t_ret = (ticker_daily['close'].iloc[-1] / ticker_daily['close'].iloc[-lookback-1]) - 1
            s_ret = (spy_daily['close'].iloc[-1] / spy_daily['close'].iloc[-lookback-1]) - 1
            return t_ret - s_ret
        except Exception:
            return np.nan


# ============================================================================
# REGIME DETECTION
# ============================================================================

def detect_market_regime(spy_daily, vix_daily):
    """
    Classifica il regime di mercato per calibrare RegimeMultiplier.
    
    Regimi:
    - 'MR_favorable': VIX alto + SPY flat/choppy → MR funziona meglio
    - 'MR_neutral': condizioni medie
    - 'MR_adverse': trend forte SPY o VIX molto basso → MR meno efficace
    
    Evidenza: Pagonidis 2013 mostra IBS performa meglio con VIX alto.
    Studi multipli confermano MR funziona male in trend forti.
    """
    regime = 'MR_neutral'
    multiplier = 1.0
    details = {}
    
    try:
        if spy_daily is not None and len(spy_daily) >= 21:
            spy_close = spy_daily['close']
            ema20 = spy_close.ewm(span=20, adjust=False).mean()
            trend_pct = (spy_close.iloc[-1] - ema20.iloc[-1]) / ema20.iloc[-1] * 100
            
            # Volatilità SPY ultimi 10 giorni
            spy_vol = spy_close.pct_change().tail(10).std() * 100
            
            details['spy_vs_ema20_pct'] = round(trend_pct, 2)
            details['spy_10d_vol_pct'] = round(spy_vol, 2)
            
            if abs(trend_pct) < 1.5 and spy_vol > 0.8:
                regime = 'MR_favorable'
                multiplier = 1.15
            elif abs(trend_pct) > 3.0:
                regime = 'MR_adverse'
                multiplier = 0.75
            else:
                regime = 'MR_neutral'
                multiplier = 1.0
        
        # VIX proxy adjustment
        if vix_daily is not None and len(vix_daily) >= 10:
            vix_close = vix_daily['close']
            vix_ma10 = vix_close.rolling(10).mean().iloc[-1]
            vix_now = vix_close.iloc[-1]
            details['vixy_vs_10dma'] = round((vix_now / vix_ma10 - 1) * 100, 2)
            if vix_now > vix_ma10 * 1.15:
                multiplier *= 1.05  # VIX spike → MR un po' più favorevole
            elif vix_now < vix_ma10 * 0.85:
                multiplier *= 0.95  # Complacenza → MR un po' meno affidabile
        
        multiplier = float(np.clip(multiplier, 0.5, 1.2))
    except Exception as e:
        details['error'] = str(e)
    
    return {
        'regime': regime,
        'multiplier': multiplier,
        'details': details,
    }


# ============================================================================
# FETCH + ANALYZE (per singolo ticker)
# ============================================================================

def fetch_and_analyze(ticker, api_key, spy_daily=None):
    """
    Fetch dati Tiingo per un ticker e restituisce tutti gli indicatori calcolati.
    """
    provider = TiingoProvider(api_key)
    calc = IndicatorCalculator()
    
    # --- Quote real-time ---
    quote = provider.get_quote(ticker)
    if quote is None:
        return None
    
    today = datetime.now().strftime('%Y-%m-%d')
    start_intraday = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    start_daily = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')
    
    intraday_df = provider.get_intraday_bars(ticker, start_intraday, resample_freq='1min')
    daily_df = provider.get_daily_prices(ticker, start_date=start_daily, end_date=today)
    
    last_price = quote.get('last') or quote.get('tngoLast') or quote.get('prevClose')
    
    result = {
        'ticker': ticker,
        'last_price': last_price,
        'timestamp': quote.get('timestamp', 'N/A'),
        'prev_close': quote.get('prevClose'),
        'open_today': quote.get('open'),
    }
    
    # --- Indicatori intraday ---
    if len(intraday_df) > 0 and 'close' in intraday_df.columns:
        try:
            close_s = pd.to_numeric(intraday_df['close'], errors='coerce').dropna()
            result['rsi'] = calc.rsi(close_s, 14)
            result['rsi_slope'] = calc.rsi_slope(close_s, 14, 5)
            result['bb_zscore'] = calc.bollinger_zscore(close_s, 20)
            result['atr_intraday'] = calc.atr(intraday_df, 14)
            result['vwap'] = calc.vwap(intraday_df)
            result['vwap_dist_atr'] = calc.vwap_distance_atr(
                intraday_df, last_price, result['atr_intraday']
            )
            result['candle_score'] = calc.candle_reversal_score(intraday_df, 5)
            result['intraday_bars_count'] = len(intraday_df)
        except Exception as e:
            st.warning(f"Errore intraday {ticker}: {e}")
            for k in ['rsi','rsi_slope','bb_zscore','atr_intraday','vwap','vwap_dist_atr','candle_score']:
                result[k] = np.nan
            result['intraday_bars_count'] = len(intraday_df)
    else:
        for k in ['rsi','rsi_slope','bb_zscore','atr_intraday','vwap','vwap_dist_atr','candle_score']:
            result[k] = np.nan
        result['intraday_bars_count'] = 0
    
    # --- Indicatori daily ---
    if len(daily_df) >= 2:
        try:
            result['adv20'] = calc.adv(daily_df, 20)
            result['atr_daily'] = calc.atr(daily_df, 14)
            result['ibs'] = calc.ibs(daily_df)
            result['vol_ratio'] = calc.volume_ratio(intraday_df, result['adv20'])
            result['gap_atr'] = calc.gap_behavior(daily_df, result.get('open_today'))
            if spy_daily is not None:
                result['rel_strength_5d'] = calc.relative_strength(daily_df, spy_daily, 5)
            else:
                result['rel_strength_5d'] = np.nan
        except Exception as e:
            st.warning(f"Errore daily {ticker}: {e}")
            for k in ['adv20','atr_daily','ibs','vol_ratio','gap_atr','rel_strength_5d']:
                result[k] = np.nan
    else:
        for k in ['adv20','atr_daily','ibs','vol_ratio','gap_atr','rel_strength_5d']:
            result[k] = np.nan
    
    return result

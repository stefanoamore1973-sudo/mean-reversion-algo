"""
data_provider.py v2.1
=====================
Changelog vs v2.0:
- FIX: filtro barre intraday solo sessione odierna (era includeva 3 giorni)
- FIX: VWAP riparte ogni giorno (intraday)
- AGGIUNTA: funzione diagnostica per debug dati grezzi
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
# CACHED FETCHERS
# ============================================================================

@st.cache_data(ttl=600)
def fetch_spy_daily(api_key, days=45):
    provider = TiingoProvider(api_key)
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    return provider.get_daily_prices('SPY', start_date=start, end_date=end)


@st.cache_data(ttl=600)
def fetch_vix_daily(api_key, days=30):
    provider = TiingoProvider(api_key)
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    return provider.get_daily_prices('VIXY', start_date=start, end_date=end)


# ============================================================================
# UTILS
# ============================================================================

def filter_today_bars(df_intraday):
    """
    Filtra il dataframe intraday per tenere SOLO le barre della sessione odierna.
    Tiingo restituisce dati dalla startDate richiesta, ma per indicatori
    come VWAP e Volume Ratio dobbiamo isolare la sessione corrente.
    """
    if df_intraday is None or len(df_intraday) == 0 or 'date' not in df_intraday.columns:
        return df_intraday
    try:
        # Data corrente (dominio US Eastern non necessario: il filtro sulla data
        # più recente presente nel df è sufficiente)
        df_intraday = df_intraday.copy()
        df_intraday['date'] = pd.to_datetime(df_intraday['date'])
        # Identifica la "sessione corrente" come data del timestamp più recente
        last_ts = df_intraday['date'].max()
        last_date = last_ts.date()
        today_mask = df_intraday['date'].dt.date == last_date
        return df_intraday[today_mask].reset_index(drop=True)
    except Exception:
        return df_intraday


# ============================================================================
# INDICATOR CALCULATOR
# ============================================================================

class IndicatorCalculator:
    
    @staticmethod
    def rsi(prices, period=14):
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
        """Internal Bar Strength — Pagonidis 2013."""
        try:
            if len(df) < 2:
                return np.nan
            last = df.iloc[-1]
            high, low, close = last['high'], last['low'], last['close']
            if pd.isna(high) or pd.isna(low) or pd.isna(close):
                return np.nan
            if high == low:
                return 0.5
            ibs_val = (close - low) / (high - low)
            return float(np.clip(ibs_val, 0, 1))
        except Exception:
            return np.nan
    
    @staticmethod
    def vwap(df):
        """VWAP cumulativo. NB: df deve essere già filtrato per la sessione odierna."""
        try:
            if not all(c in df.columns for c in ['high','low','close','volume']) or len(df) == 0:
                return np.nan
            typical = (df['high'] + df['low'] + df['close']) / 3
            # Converti volume a numerico per sicurezza
            vol = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            vol_sum = vol.cumsum()
            if vol_sum.iloc[-1] == 0:
                return np.nan
            return ((typical * vol).cumsum() / vol_sum).iloc[-1]
        except Exception:
            return np.nan
    
    @staticmethod
    def vwap_distance_atr(df, current_price, atr_value):
        try:
            v = IndicatorCalculator.vwap(df)
            if pd.isna(v) or pd.isna(atr_value) or atr_value == 0 or pd.isna(current_price):
                return np.nan
            return (current_price - v) / atr_value
        except Exception:
            return np.nan
    
    @staticmethod
    def bollinger_zscore(prices, period=20):
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
        try:
            if len(df) < period or 'volume' not in df.columns or 'close' not in df.columns:
                return np.nan
            return (df['volume'] * df['close']).tail(period).mean()
        except Exception:
            return np.nan
    
    @staticmethod
    def volume_ratio(df_intraday_today, adv20):
        """Volume $ oggi cumulato / ADV20. df deve essere già filtrato per oggi."""
        try:
            if df_intraday_today is None or len(df_intraday_today) == 0:
                return np.nan
            if pd.isna(adv20) or adv20 == 0:
                return np.nan
            if 'close' not in df_intraday_today.columns or 'volume' not in df_intraday_today.columns:
                return np.nan
            vol = pd.to_numeric(df_intraday_today['volume'], errors='coerce').fillna(0)
            cls = pd.to_numeric(df_intraday_today['close'], errors='coerce').fillna(0)
            dollar_vol_today = (vol * cls).sum()
            return dollar_vol_today / adv20
        except Exception:
            return np.nan
    
    @staticmethod
    def candle_reversal_score(df_intraday_today, lookback=5):
        try:
            if df_intraday_today is None or len(df_intraday_today) < lookback:
                return 0.5
            recent = df_intraday_today.tail(lookback).copy()
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
                if body > 0 and lower_wick >= 2 * body and upper_wick <= body * 0.5 and c >= o:
                    score += 1
                    continue
                if i > 0:
                    prev = recent.iloc[i-1]
                    po, pc = prev['open'], prev['close']
                    if not any(pd.isna([po,pc])):
                        if pc < po and c > o and c > po and o < pc:
                            score += 1
                            continue
                if body / total_range < 0.1 and (c - l) / total_range < 0.3:
                    score += 0.5
            if max_score == 0:
                return 0.5
            return min(1.0, score / max_score + 0.2)
        except Exception:
            return 0.5
    
    @staticmethod
    def gap_behavior(df_daily, intraday_open):
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
    regime = 'MR_neutral'
    multiplier = 1.0
    details = {}
    
    try:
        if spy_daily is not None and len(spy_daily) >= 21:
            spy_close = spy_daily['close']
            ema20 = spy_close.ewm(span=20, adjust=False).mean()
            trend_pct = (spy_close.iloc[-1] - ema20.iloc[-1]) / ema20.iloc[-1] * 100
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
        
        if vix_daily is not None and len(vix_daily) >= 10:
            vix_close = vix_daily['close']
            vix_ma10 = vix_close.rolling(10).mean().iloc[-1]
            vix_now = vix_close.iloc[-1]
            details['vixy_vs_10dma'] = round((vix_now / vix_ma10 - 1) * 100, 2)
            if vix_now > vix_ma10 * 1.15:
                multiplier *= 1.05
            elif vix_now < vix_ma10 * 0.85:
                multiplier *= 0.95
        
        multiplier = float(np.clip(multiplier, 0.5, 1.2))
    except Exception as e:
        details['error'] = str(e)
    
    return {
        'regime': regime,
        'multiplier': multiplier,
        'details': details,
    }


# ============================================================================
# FETCH + ANALYZE
# ============================================================================

def fetch_and_analyze(ticker, api_key, spy_daily=None, debug=False):
    """
    Fetch dati Tiingo per un ticker e calcola tutti gli indicatori.
    Se debug=True, include i dati grezzi per diagnostica.
    """
    provider = TiingoProvider(api_key)
    calc = IndicatorCalculator()
    
    quote = provider.get_quote(ticker)
    if quote is None:
        return None
    
    today = datetime.now().strftime('%Y-%m-%d')
    # Fetchiamo ultimi 3 giorni per avere contesto RSI/BB, ma filtriamo per oggi su VWAP/vol
    start_intraday = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    start_daily = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')
    
    intraday_df_all = provider.get_intraday_bars(ticker, start_intraday, resample_freq='1min')
    daily_df = provider.get_daily_prices(ticker, start_date=start_daily, end_date=today)
    
    # Filtro per la sessione odierna (per VWAP, vol ratio, candle pattern)
    intraday_df_today = filter_today_bars(intraday_df_all)
    
    last_price = quote.get('last') or quote.get('tngoLast') or quote.get('prevClose')
    
    result = {
        'ticker': ticker,
        'last_price': last_price,
        'timestamp': quote.get('timestamp', 'N/A'),
        'prev_close': quote.get('prevClose'),
        'open_today': quote.get('open'),
        'quote_volume_cumulative': quote.get('volume'),  # Volume cumulato dal quote endpoint
    }
    
    # ------------------------------------------------------------------
    # DIAGNOSTICA (se richiesta)
    # ------------------------------------------------------------------
    if debug:
        diag = {
            'quote_keys': list(quote.keys()) if quote else [],
            'quote_volume_field': quote.get('volume'),
            'intraday_df_all_rows': len(intraday_df_all),
            'intraday_df_all_cols': list(intraday_df_all.columns) if len(intraday_df_all) > 0 else [],
            'intraday_df_today_rows': len(intraday_df_today),
            'daily_df_rows': len(daily_df),
        }
        # Prime 3 righe del df intraday (oggi) come dict
        if len(intraday_df_today) > 0:
            diag['intraday_today_first_3'] = intraday_df_today.head(3).to_dict(orient='records')
            diag['intraday_today_last_1'] = intraday_df_today.tail(1).to_dict(orient='records')
            # Volume stats
            vol_series = pd.to_numeric(intraday_df_today['volume'], errors='coerce').fillna(0)
            diag['volume_sum_today'] = int(vol_series.sum())
            diag['volume_nonzero_bars'] = int((vol_series > 0).sum())
            diag['volume_max_bar'] = int(vol_series.max()) if len(vol_series) > 0 else 0
        result['_diagnostic'] = diag
    
    # ------------------------------------------------------------------
    # INDICATORI INTRADAY
    # ------------------------------------------------------------------
    # Uso intraday_df_all per RSI/BB (ha bisogno di più barre)
    # Uso intraday_df_today per VWAP/Volume/Candle (intraday puro)
    
    if len(intraday_df_all) > 0 and 'close' in intraday_df_all.columns:
        try:
            close_all = pd.to_numeric(intraday_df_all['close'], errors='coerce').dropna()
            result['rsi'] = calc.rsi(close_all, 14)
            result['rsi_slope'] = calc.rsi_slope(close_all, 14, 5)
            result['bb_zscore'] = calc.bollinger_zscore(close_all, 20)
            result['atr_intraday'] = calc.atr(intraday_df_all, 14)
        except Exception as e:
            st.warning(f"Errore indicatori intraday (tutti) {ticker}: {e}")
            for k in ['rsi','rsi_slope','bb_zscore','atr_intraday']:
                result[k] = np.nan
    else:
        for k in ['rsi','rsi_slope','bb_zscore','atr_intraday']:
            result[k] = np.nan
    
    if len(intraday_df_today) > 0 and 'close' in intraday_df_today.columns:
        try:
            result['vwap'] = calc.vwap(intraday_df_today)
            result['vwap_dist_atr'] = calc.vwap_distance_atr(
                intraday_df_today, last_price, result.get('atr_intraday')
            )
            result['candle_score'] = calc.candle_reversal_score(intraday_df_today, 5)
            result['intraday_bars_today'] = len(intraday_df_today)
        except Exception as e:
            st.warning(f"Errore indicatori intraday oggi {ticker}: {e}")
            for k in ['vwap','vwap_dist_atr','candle_score']:
                result[k] = np.nan
            result['intraday_bars_today'] = len(intraday_df_today)
    else:
        for k in ['vwap','vwap_dist_atr','candle_score']:
            result[k] = np.nan
        result['intraday_bars_today'] = 0
    
    # Alias per compatibilità
    result['intraday_bars_count'] = result.get('intraday_bars_today', 0)
    
    # ------------------------------------------------------------------
    # INDICATORI DAILY
    # ------------------------------------------------------------------
    if len(daily_df) >= 2:
        try:
            result['adv20'] = calc.adv(daily_df, 20)
            result['atr_daily'] = calc.atr(daily_df, 14)
            result['ibs'] = calc.ibs(daily_df)
            # Volume ratio: usa il quote endpoint se disponibile, fallback barre today
            quote_vol = quote.get('volume')
            if quote_vol is not None and last_price and result.get('adv20') and not pd.isna(result['adv20']):
                dollar_vol = quote_vol * last_price
                result['vol_ratio'] = dollar_vol / result['adv20']
                result['vol_source'] = 'quote_endpoint'
            else:
                result['vol_ratio'] = calc.volume_ratio(intraday_df_today, result.get('adv20'))
                result['vol_source'] = 'intraday_bars_sum'
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

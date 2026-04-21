import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

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
    
    def get_metadata(self, ticker):
        url = f"{self.base_url}/tiingo/daily/{ticker}?token={self.api_key}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            st.warning(f"Error fetching metadata {ticker}: {str(e)}")
            return None


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
            recent_rsi = rsi_series.iloc[-lookback:]
            if len(recent_rsi) < lookback:
                return np.nan
            slope = (recent_rsi.iloc[-1] - recent_rsi.iloc[0]) / lookback
            return slope
        except Exception:
            return np.nan
    
    @staticmethod
    def vwap(df):
        try:
            required_cols = ['high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return np.nan
            if len(df) == 0:
                return np.nan
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vol_sum = df['volume'].cumsum()
            if vol_sum.iloc[-1] == 0:
                return np.nan
            vwap = (typical_price * df['volume']).cumsum() / vol_sum
            return vwap.iloc[-1]
        except Exception:
            return np.nan
    
    @staticmethod
    def bollinger_zscore(prices, period=20, num_std=2):
        try:
            if len(prices) < period:
                return np.nan
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            zscore = (prices.iloc[-1] - sma.iloc[-1]) / (std.iloc[-1] + 1e-8)
            return zscore
        except Exception:
            return np.nan
    
    @staticmethod
    def atr(df, period=14):
        try:
            if len(df) < period:
                return np.nan
            required_cols = ['high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                return np.nan
            high = df['high']
            low = df['low']
            close = df['close']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr.iloc[-1]
        except Exception:
            return np.nan
    
    @staticmethod
    def adv(df, period=20):
        try:
            if len(df) < period:
                return np.nan
            if 'volume' not in df.columns or 'close' not in df.columns:
                return np.nan
            return (df['volume'] * df['close']).tail(period).mean()
        except Exception:
            return np.nan


def fetch_and_analyze(ticker, api_key):
    provider = TiingoProvider(api_key)
    calc = IndicatorCalculator()
    
    quote = provider.get_quote(ticker)
    if quote is None:
        return None
    
    today = datetime.now().strftime('%Y-%m-%d')
    intraday_df = provider.get_intraday_bars(ticker, today, resample_freq='1min')
    
    daily_df = provider.get_daily_prices(ticker, 
                                         start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                                         end_date=today)
    
    result = {
        'ticker': ticker,
        'last_price': quote.get('last') if quote.get('last') is not None else quote.get('tngoLast'),
        'timestamp': quote.get('timestamp', 'N/A'),
        'quote_raw': quote,
    }
    
    if len(intraday_df) > 0 and 'close' in intraday_df.columns:
        try:
            close_series = pd.to_numeric(intraday_df['close'], errors='coerce').dropna()
            result['rsi'] = calc.rsi(close_series, period=14)
            result['rsi_slope'] = calc.rsi_slope(close_series, period=14, lookback=5)
            result['vwap'] = calc.vwap(intraday_df)
            result['bb_zscore'] = calc.bollinger_zscore(close_series, period=20)
            result['atr'] = calc.atr(intraday_df, period=14)
            
            if 'volume' in intraday_df.columns and len(intraday_df) > 0:
                vol_val = intraday_df['volume'].iloc[-1]
                result['volume_last'] = vol_val if pd.notna(vol_val) else np.nan
            else:
                result['volume_last'] = np.nan
            
            result['intraday_bars_count'] = len(intraday_df)
        except Exception as e:
            st.warning(f"Errore calcolo indicatori intraday {ticker}: {str(e)}")
            result['rsi'] = np.nan
            result['rsi_slope'] = np.nan
            result['vwap'] = np.nan
            result['bb_zscore'] = np.nan
            result['atr'] = np.nan
            result['volume_last'] = np.nan
            result['intraday_bars_count'] = len(intraday_df)
    else:
        result['rsi'] = np.nan
        result['rsi_slope'] = np.nan
        result['vwap'] = np.nan
        result['bb_zscore'] = np.nan
        result['atr'] = np.nan
        result['volume_last'] = np.nan
        result['intraday_bars_count'] = 0
    
    if len(daily_df) > 0:
        try:
            result['adv20'] = calc.adv(daily_df, period=20)
            result['atr_daily'] = calc.atr(daily_df, period=14)
        except Exception as e:
            st.warning(f"Errore calcolo indicatori daily {ticker}: {str(e)}")
            result['adv20'] = np.nan
            result['atr_daily'] = np.nan
    else:
        result['adv20'] = np.nan
        result['atr_daily'] = np.nan
    
    return result

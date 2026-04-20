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
        """Fetch quote real-time IEX per un ticker."""
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
        """Fetch OHLCV intraday (1-min default) da Tiingo IEX."""
        url = f"{self.base_url}/iex/{ticker}/prices?startDate={start_date}&resampleFreq={resample_freq}&token={self.api_key}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                return pd.DataFrame()
            else:
                st.warning(f"Tiingo intraday bars {ticker}: HTTP {resp.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error fetching intraday bars {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_daily_prices(self, ticker, start_date=None, end_date=None):
        """Fetch daily OHLCV EOD da Tiingo daily."""
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
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                return pd.DataFrame()
            else:
                st.warning(f"Tiingo daily prices {ticker}: HTTP {resp.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error fetching daily prices {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_metadata(self, ticker):
        """Fetch metadata ticker (name, exchange, etc)."""
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
    """Calcoli indicatori tecnici."""
    
    @staticmethod
    def rsi(prices, period=14):
        """RSI (Relative Strength Index) — Wilder."""
        if len(prices) < period + 1:
            return np.nan
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    @staticmethod
    def rsi_slope(prices, period=14, lookback=5):
        """RSI slope — ultimi 5 periodi."""
        if len(prices) < period + lookback:
            return np.nan
        rsi_series = 100 - (100 / (1 + (prices.diff().where(lambda x: x > 0, 0).rolling(period, min_periods=1).mean() / 
                                          (-prices.diff()).where(lambda x: -prices.diff() > 0, 0).rolling(period, min_periods=1).mean())))
        recent_rsi = rsi_series.iloc[-lookback:]
        if len(recent_rsi) < lookback:
            return np.nan
        slope = (recent_rsi.iloc[-1] - recent_rsi.iloc[0]) / lookback
        return slope
    
    @staticmethod
    def vwap(df):
        """VWAP intraday (cumulative)."""
        if 'close' not in df.columns or 'volume' not in df.columns:
            return np.nan
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.iloc[-1]
    
    @staticmethod
    def bollinger_zscore(prices, period=20, num_std=2):
        """Bollinger Band z-score."""
        if len(prices) < period:
            return np.nan
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        zscore = (prices.iloc[-1] - sma.iloc[-1]) / (std.iloc[-1] + 1e-8)
        return zscore
    
    @staticmethod
    def atr(df, period=14):
        """ATR (Average True Range)."""
        if len(df) < period:
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
    
    @staticmethod
    def adv(df, period=20):
        """Average Daily Volume."""
        if len(df) < period:
            return np.nan
        return (df['volume'] * df['close']).tail(period).mean()


def fetch_and_analyze(ticker, api_key):
    """Fetch dati Tiingo e calcola indicatori base."""
    provider = TiingoProvider(api_key)
    calc = IndicatorCalculator()
    
    # Quote real-time
    quote = provider.get_quote(ticker)
    if quote is None:
        return None
    
    # Intraday bars (sessione odierna)
    today = datetime.now().strftime('%Y-%m-%d')
    intraday_df = provider.get_intraday_bars(ticker, today, resample_freq='1min')
    
    # Daily prices (ultimi 30 giorni per ADV, ATR)
    daily_df = provider.get_daily_prices(ticker, 
                                         start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                                         end_date=today)
    
    result = {
        'ticker': ticker,
        'last_price': quote.get('last'),
        'timestamp': quote.get('timestamp'),
        'quote_raw': quote,
    }
    
    # Indicatori intraday
    if len(intraday_df) > 0:
        result['rsi'] = calc.rsi(intraday_df['close'].astype(float), period=14)
        result['rsi_slope'] = calc.rsi_slope(intraday_df['close'].astype(float), period=14, lookback=5)
        result['vwap'] = calc.vwap(intraday_df)
        result['bb_zscore'] = calc.bollinger_zscore(intraday_df['close'].astype(float), period=20)
        result['atr'] = calc.atr(intraday_df, period=14)
        result['volume_last'] = intraday_df['volume'].iloc[-1] if len(intraday_df) > 0 else np.nan
        result['intraday_bars_count'] = len(intraday_df)
    else:
        result['rsi'] = np.nan
        result['rsi_slope'] = np.nan
        result['vwap'] = np.nan
        result['bb_zscore'] = np.nan
        result['atr'] = np.nan
        result['volume_last'] = np.nan
        result['intraday_bars_count'] = 0
    
    # Indicatori daily
    if len(daily_df) > 0:
        result['adv20'] = calc.adv(daily_df, period=20)
        result['atr_daily'] = calc.atr(daily_df, period=14)
    else:
        result['adv20'] = np.nan
        result['atr_daily'] = np.nan
    
    return result

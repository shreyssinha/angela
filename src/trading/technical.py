import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Technical:
    def __init__(self, data):
        """
        Initialize with historical data from archive
        
        Parameters:
        data: DataFrame or dict
            Historical price data containing at minimum:
            - Close/close prices
            - High/high prices
            - Low/low prices
            - Volume/volume data
        """
        # Convert to DataFrame if dict is passed
        if isinstance(data, dict):
            self.df = pd.DataFrame(data)
        else:
            self.df = data.copy()
            
        # Standardize column names (handle both uppercase and lowercase)
        column_mapping = {
            'close': 'Close',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume',
            'timestamp': 'Date'
        }
        
        self.df.rename(columns={k: v for k, v in column_mapping.items() 
                              if k in self.df.columns}, inplace=True)
        
        # Ensure DataFrame is sorted by date
        if 'Date' in self.df.columns:
            self.df.sort_values('Date', inplace=True)
            self.df.set_index('Date', inplace=True)
        
    def fetch_data(self):
        """Fetch historical data for the ticker"""
        try:
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def calculate_sma(self, period=20):
        """Calculate Simple Moving Average"""
        self.df[f'SMA_{period}'] = self.df['Close'].rolling(window=period).mean()
        self.df['SMA_Signal'] = np.where(self.df['Close'] > self.df[f'SMA_{period}'], 1, -1)
        return self.df[f'SMA_{period}']

    def calculate_ema(self, period=20):
        """Calculate Exponential Moving Average"""
        self.df[f'EMA_{period}'] = self.df['Close'].ewm(span=period, adjust=False).mean()
        self.df['EMA_Signal'] = np.where(self.df['Close'] > self.df[f'EMA_{period}'], 1, -1)
        return self.df[f'EMA_{period}']

    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        self.df['EMA_fast'] = self.df['Close'].ewm(span=fast, adjust=False).mean()
        self.df['EMA_slow'] = self.df['Close'].ewm(span=slow, adjust=False).mean()
        self.df['MACD'] = self.df['EMA_fast'] - self.df['EMA_slow']
        self.df['MACD_Signal_Line'] = self.df['MACD'].ewm(span=signal, adjust=False).mean()
        self.df['MACD_Signal'] = np.where(self.df['MACD'] > self.df['MACD_Signal_Line'], 1, -1)
        return self.df['MACD']

    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        self.df['RSI_Signal'] = np.where(self.df['RSI'] > 50, 1, -1)
        return self.df['RSI']

    def calculate_mfi(self, period=14):
        """Calculate Money Flow Index"""
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow = typical_price * self.df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        mfi_ratio = positive_flow / negative_flow
        self.df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        self.df['MFI_Signal'] = np.where(self.df['MFI'] > 50, 1, -1)
        return self.df['MFI']

    def calculate_roc(self, period=10):
        """Calculate Rate of Change"""
        self.df['ROC'] = self.df['Close'].pct_change(periods=period) * 100
        self.df['ROC_Signal'] = np.where(self.df['ROC'] > 0, 1, -1)
        return self.df['ROC']

    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        self.df['BB_middle'] = self.df['Close'].rolling(window=period).mean()
        std = self.df['Close'].rolling(window=period).std()
        self.df['BB_upper'] = self.df['BB_middle'] + (std_dev * std)
        self.df['BB_lower'] = self.df['BB_middle'] - (std_dev * std)
        self.df['BB_Signal'] = np.where(self.df['Close'] > self.df['BB_middle'], 1, -1)
        return self.df[['BB_upper', 'BB_middle', 'BB_lower']]

    def calculate_atr(self, period=14):
        """Calculate Average True Range"""
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift())
        low_close = abs(self.df['Low'] - self.df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.df['ATR'] = true_range.rolling(window=period).mean()
        self.df['ATR_Signal'] = np.where(self.df['ATR'] < self.df['ATR'].rolling(window=10).mean(), 1, -1)
        return self.df['ATR']

    def calculate_obv(self):
        """Calculate On-Balance Volume"""
        self.df['OBV'] = (self.df['Volume'] * (~self.df['Close'].diff().le(0) * 2 - 1)).cumsum()
        self.df['OBV_Signal'] = np.where(self.df['OBV'] > self.df['OBV'].shift(1), 1, -1)
        return self.df['OBV']

    def calculate_volume_sma(self, period=20):
        """Calculate Volume SMA"""
        self.df['Volume_SMA'] = self.df['Volume'].rolling(window=period).mean()
        self.df['Volume_SMA_Signal'] = np.where(self.df['Volume'] > self.df['Volume_SMA'], 1, -1)
        return self.df['Volume_SMA']

# Initialize for a specific ticker
tech = Technical('AAPL')  # or any other ticker

# Calculate individual indicators
print(tech.calculate_rsi())
print(tech.calculate_macd())

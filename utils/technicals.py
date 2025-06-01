import pandas as pd
import numpy as np

def calculate_rsi(close, window=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, window=14):
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_ma(close, window=14):
    return close.rolling(window=window).mean()

def add_indicators(df, rsi_window=14, atr_window=14, ma_window=14):
    df = df.copy()
    df["RSI_14"] = calculate_rsi(df["Close"], window=rsi_window)
    df["ATR_14"] = calculate_atr(df["High"], df["Low"], df["Close"], window=atr_window)
    df["MA_14"] = calculate_ma(df["Close"], window=ma_window)
    return df

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
from io import StringIO

def fetch_crypto_data(symbol="BTC-USD", start=None, end=None, interval="1d"):
    """
    Ambil data historis crypto dari Yahoo Finance. Default: 400 hari terakhir.
    """
    if start is None:
        start = datetime.now() - pd.Timedelta(days=400)
    if end is None:
        end = datetime.now()

    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df



def fetch_fgi_data():
    """
    Ambil data FGI dari API Alternative.me
    """
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    try:
        response = requests.get(url)
        fgi_json = response.json()
        fgi_data = fgi_json["data"]
        df_fgi = pd.DataFrame(fgi_data)
        df_fgi['timestamp'] = df_fgi['timestamp'].astype(int)
        df_fgi['Date'] = pd.to_datetime(df_fgi['timestamp'], unit='s').dt.date
        df_fgi = df_fgi[["Date", "value"]]
        df_fgi.columns = ["Date", "FGI"]
        df_fgi["FGI"] = df_fgi["FGI"].astype(int)
        return df_fgi
    except Exception as e:
        print(f"Error fetching FGI data: {e}")
        return pd.DataFrame()

def fetch_latest_data(symbol="BTC-USD"):
    """
    Gabungkan data historis crypto dan FGI berdasarkan tanggal.
    """
    df_price = fetch_crypto_data(symbol=symbol)
    df_fgi = fetch_fgi_data()

    if df_fgi.empty:
        # Jika FGI gagal diambil, isi dengan NaN
        df_price["FGI"] = None
        return df_price
    else:
        df_merged = pd.merge(df_price, df_fgi, on="Date", how="left")
        return df_merged

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import time
import sys

# === Konfigurasi ===
COIN_SYMBOL = "BTC-USD"  # Ubah ke simbol yang valid di Yahoo Finance
WINDOW_SIZE = 14
CLUSTER_K = 3
MODEL_FOLDER = os.path.join("2_models", "lstm_models")
LOG_FOLDER = "log_predict"
os.makedirs(LOG_FOLDER, exist_ok=True)

# === Logging helper ===
def log_print(msg):
    print(msg)
    with open(os.path.join(LOG_FOLDER, "log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# === Load scaler & model ===
try:
    model_path = os.path.join(MODEL_FOLDER, f"btc_combined_k{CLUSTER_K}_w{WINDOW_SIZE}_model.h5")
    scaler_path = os.path.join(MODEL_FOLDER, f"btc_combined_k{CLUSTER_K}_w{WINDOW_SIZE}_scaler.pkl")
    input_scaler_path = os.path.join(MODEL_FOLDER, f"btc_combined_k{CLUSTER_K}_w{WINDOW_SIZE}_xscaler.pkl")

    scaler_X = joblib.load(input_scaler_path)
    scaler_y = joblib.load(scaler_path)
    model = load_model(model_path)
except Exception as e:
    log_print(f"Gagal load model atau scaler LSTM: {e}")
    sys.exit(1)

# === Load model & scaler untuk klasterisasi ===
try:
    cluster_model_path = f"2_models/cluster_models/btc_combined/kmeans_k{CLUSTER_K}.pkl"
    scaler_cluster_path = f"2_models/cluster_models/btc_combined/scaler_k{CLUSTER_K}.pkl"

    kmeans = joblib.load(cluster_model_path)
    cluster_scaler = joblib.load(scaler_cluster_path)
except Exception as e:
    log_print(f"Gagal load model atau scaler klasterisasi: {e}")
    sys.exit(1)

# === Ambil data terbaru ===
def fetch_latest_data(period="60d", max_retries=3):
    for attempt in range(max_retries):
        try:
            df = yf.download(COIN_SYMBOL, period=period, interval="1d", progress=False)
            if df.empty:
                log_print(f"Data kosong, percobaan ke-{attempt+1}")
                time.sleep(3)
            else:
                df = df.reset_index()
                df = df[["Date", "Close", "Volume"]].copy()
                df["Date"] = pd.to_datetime(df["Date"])
                return df
        except Exception as e:
            log_print(f"Error fetch data (percobaan {attempt+1}): {e}")
            time.sleep(3)
    log_print("Gagal fetch data setelah beberapa percobaan.")
    sys.exit(1)

# === Hitung indikator ===
def calculate_indicators(df):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(WINDOW_SIZE).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(WINDOW_SIZE).mean()
    RS = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + RS))
    high_low = df["Close"].rolling(WINDOW_SIZE).max() - df["Close"].rolling(WINDOW_SIZE).min()
    df["ATR_14"] = high_low.rolling(WINDOW_SIZE).mean()
    df["FGI"] = np.random.randint(20, 80, len(df))  # Dummy FGI
    df = df.dropna()
    if len(df) < WINDOW_SIZE:
        log_print(f"Jumlah data ({len(df)}) tidak cukup untuk window size ({WINDOW_SIZE})")
        sys.exit(1)
    return df

# === Klasterisasi dan windowing ===
def create_window_for_prediction(df):
    window = df[-WINDOW_SIZE:].copy()
    if window.empty or len(window) < WINDOW_SIZE:
        raise ValueError("Window data tidak cukup.")

    last_row = window.iloc[-1]
    features = ["FGI", "ATR_14", "RSI_14"]
    scaled_features = cluster_scaler.transform([last_row[features].values])
    cluster_label = kmeans.predict(scaled_features)[0]

    row = {}
    for j in range(WINDOW_SIZE):
        row[f"Close_t-{WINDOW_SIZE - j - 1}"] = window.iloc[j]["Close"]
    row["RSI_14"] = last_row["RSI_14"]
    row["ATR_14"] = last_row["ATR_14"]
    row["FGI"] = last_row["FGI"]
    row["cluster_subset"] = cluster_label

    return pd.DataFrame([row])

# === Prediksi harga ===
def predict_next_price():
    df = fetch_latest_data()
    df = calculate_indicators(df)

    now_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(LOG_FOLDER, f"btc_{now_str}_raw.csv")
    df.to_csv(raw_path, index=False)
    log_print(f"Data mentah disimpan: {raw_path}")
    log_print(f"Harga terakhir: {df['Close'].iloc[-1]}")

    input_df = create_window_for_prediction(df)
    input_path = os.path.join(LOG_FOLDER, f"btc_{now_str}_input.csv")
    input_df.to_csv(input_path, index=False)
    log_print(f"Input model disimpan: {input_path}")

    feature_cols = [col for col in input_df.columns if col != "y"]
    X = input_df[feature_cols]

    X_scaled = scaler_X.transform(X)
    log_print(f"Bentuk sebelum reshape: {X_scaled.shape}")
    log_print(f"Sample scaled input: {X_scaled[0][:10]}")

    X_scaled = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    return y_pred[0][0]

# === Eksekusi utama ===
if __name__ == "__main__":
    try:
        predicted_price = predict_next_price()
        log_print(f"Prediksi harga berikutnya: {predicted_price:.2f}")
    except Exception as e:
        log_print(f"Terjadi error saat prediksi: {e}")

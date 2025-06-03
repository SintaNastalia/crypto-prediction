import os
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
import tensorflow as tf
import traceback

from utils.data_fetcher import fetch_crypto_data, fetch_fgi_data
from utils.technicals import add_indicators
from utils.cluster_loader import load_cluster_model_and_data

# --- Konfigurasi ---
COINS = {
    "btc": "BTC-USD",
    "eth": "ETH-USD",
    "ada": "ADA-USD"
}
WINDOW_SIZE = 7
N_CLUSTERS = 4
MODEL_DIR = "2_models/lstm_models"  # Sesuaikan folder model training-mu

# --- Fungsi ---
def predict_once(model, x):
    return model(x, training=False)

def ensure_dirs():
    dirs = [
        "1_data_realtime/historical",
        "1_data_realtime/processed",
        "1_data_realtime/fgi",
        "1_data_realtime/prediction"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def one_hot_encode_cluster(cluster_series, n_clusters):
    """Ubah kolom cluster integer jadi one-hot dataframe."""
    one_hot = pd.get_dummies(cluster_series, prefix='cluster')
    # Pastikan kolom lengkap (0..n_clusters-1)
    for i in range(n_clusters):
        col_name = f'cluster_{i}'
        if col_name not in one_hot.columns:
            one_hot[col_name] = 0
    return one_hot[[f'cluster_{i}' for i in range(n_clusters)]]

# --- Mulai Prediksi ---
ensure_dirs()

print("[INFO] Mengambil data FGI...")
df_fgi = fetch_fgi_data()
df_fgi.to_csv("1_data_realtime/fgi/fgi.csv", index=False)

for coin_name, symbol in COINS.items():
    print(f"\n[INFO] Memproses {coin_name()} ({symbol})...")

    try:
        # Ambil data harga
        df_price = fetch_crypto_data(symbol)
        df_price.to_csv(f"1_data_realtime/historical/{coin_name}.csv", index=False)

        # Gabungkan dengan FGI & hitung indikator
        df_merged = pd.merge(df_price, df_fgi, on="Date", how="left")
        df_merged = add_indicators(df_merged)

        # Load model klaster dan scaler
        kmeans, scaler_cluster, _ = load_cluster_model_and_data(coin_name(), N_CLUSTERS)
        cluster_features = list(scaler_cluster.feature_names_in_)

        # Validasi kolom tersedia untuk clustering
        missing_features = [f for f in cluster_features if f not in df_merged.columns]
        if missing_features:
            print(f"[ERROR] Kolom hilang setelah indikator: {missing_features}")
            continue

        # Prediksi klaster
        df_merged["cluster"] = -1
        valid_rows = df_merged.dropna(subset=cluster_features)
        if not valid_rows.empty:
            cluster_input = scaler_cluster.transform(valid_rows[cluster_features])
            cluster_labels = kmeans.predict(cluster_input)
            df_merged.loc[valid_rows.index, "cluster"] = cluster_labels
        else:
            print(f"[WARNING] Tidak ada data valid untuk clustering pada {coin_name()}")
            continue

        # One-hot encode cluster
        df_merged = df_merged.reset_index(drop=True)
        one_hot_cluster_df = one_hot_encode_cluster(df_merged["cluster"], N_CLUSTERS)
        df_merged = pd.concat([df_merged, one_hot_cluster_df], axis=1)
        df_merged.drop(columns=["cluster"], inplace=True)
        df_merged.dropna(inplace=True)
        df_merged = df_merged.reset_index(drop=True)

        # Simpan processed
        processed_path = f"1_data_realtime/processed/{coin_name}_processed.csv"
        df_merged.to_csv(processed_path, index=False)
        print(f"[INFO] Data processed disimpan ke: {processed_path}")

        # Load scaler dan model
        scaler_x_path = os.path.join(MODEL_DIR, f"{coin_name}_k{N_CLUSTERS}_w{WINDOW_SIZE}_xscaler.pkl")
        scaler_y_path = os.path.join(MODEL_DIR, f"{coin_name}_k{N_CLUSTERS}_w{WINDOW_SIZE}_scaler.pkl")
        model_path = os.path.join(MODEL_DIR, f"{coin_name}_k{N_CLUSTERS}_w{WINDOW_SIZE}_model.h5")

        if not all(map(os.path.exists, [scaler_x_path, scaler_y_path, model_path])):
            print(f"[ERROR] Model atau scaler tidak ditemukan untuk {coin_name()}. Lewati.")
            continue

        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        model = load_model(model_path)
        feature_names = list(scaler_X.feature_names_in_)

        # Debug fitur
        print(f"[DEBUG] Fitur model: {feature_names}")
        print(f"[DEBUG] Kolom df_merged: {list(df_merged.columns)}")

        # Validasi fitur & window
        if len(df_merged) < WINDOW_SIZE:
            print(f"[SKIP] {coin_name()}: Hanya {len(df_merged)} baris, kurang dari window ({WINDOW_SIZE})")
            continue

        if df_merged[feature_names].isnull().any().any():
            print(f"[ERROR] NaN ditemukan dalam fitur yang dibutuhkan untuk {coin_name()}")
            continue

        # Prediksi
        df_window = df_merged[feature_names].iloc[-WINDOW_SIZE:]
        X_input = df_window.values
        X_input_scaled = scaler_X.transform(X_input)
        X_input_scaled = X_input_scaled.reshape(1, WINDOW_SIZE, -1)

        print(f"[INFO] Bentuk input untuk prediksi: {X_input_scaled.shape}")

        predicted_scaled = predict_once(model, tf.convert_to_tensor(X_input_scaled, dtype=tf.float32))
        predicted_price = scaler_y.inverse_transform(predicted_scaled.numpy())[0, 0]

        print(f"[PREDIKSI] {coin_name()}: {predicted_price:.2f}")

        # Simpan hasil prediksi
        next_date = pd.to_datetime(df_price["Date"].max()) + timedelta(days=1)
        pred_df = pd.DataFrame({
            "Date": [next_date],
            "Predicted_Close": [predicted_price]
        })
        pred_df.to_csv(f"1_data_realtime/prediction/{coin_name}_predicted.csv", index=False)
        print(f"[SUCCESS] Prediksi {coin_name()} disimpan.")

    except Exception as e:
        print(f"[ERROR] Gagal memproses {coin_name()}: {e}")
        traceback.print_exc()

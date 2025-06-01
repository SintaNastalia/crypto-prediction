import os
import joblib
import pandas as pd

def load_cluster_model_and_data(coin_name: str, n_clusters: int = 4, base_dir: str = "2_models") -> tuple:
    """
    Memuat model clustering (KMeans dan Scaler) serta data hasil klasterisasi
    berdasarkan nama koin dan jumlah klaster. Default n_clusters = 4.
    """
    model_dir = os.path.join(base_dir, "cluster_models", f"{coin_name}_combined")
    data_file = os.path.join(base_dir, "clustered_data", f"{coin_name}_combined.csv")

    kmeans_file = os.path.join(model_dir, f"kmeans_k{n_clusters}.pkl")
    scaler_file = os.path.join(model_dir, f"scaler_k{n_clusters}.pkl")

    print(f"Memuat model dan data untuk: {coin_name}, klaster: {n_clusters}")
    
    if not os.path.exists(kmeans_file):
        raise FileNotFoundError(f"Model KMeans tidak ditemukan: {kmeans_file}")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Scaler tidak ditemukan: {scaler_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data klasterisasi tidak ditemukan: {data_file}")

    kmeans = joblib.load(kmeans_file)
    scaler = joblib.load(scaler_file)
    clustered_df = pd.read_csv(data_file)

    return kmeans, scaler, clustered_df

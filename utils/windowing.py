import os
import pandas as pd

def create_feature_window(df, window_size=14):
    result = []

    for i in range(len(df) - window_size):
        window = df.iloc[i:i + window_size]
        target = df.iloc[i + window_size]["Close"]

        row = {}

        # Tambahkan harga Close window sebelumnya sebagai fitur Close_t-(n)
        for j in range(window_size):
            row[f"Close_t-{window_size - j - 1}"] = window.iloc[j]["Close"]

        # Ambil semua fitur teknikal dari baris terakhir window (kecuali Date, Close, dan cluster)
        last_row = window.iloc[-1]
        feature_columns = [col for col in df.columns if col not in ["Date", "Close", "cluster"]]

        for col in feature_columns:
            row[col] = last_row[col]

        # Tambahkan cluster sebagai fitur numerik
        row["cluster"] = last_row["cluster"]

        # Target y = harga Close hari berikutnya
        row["y"] = target
        result.append(row)

    return pd.DataFrame(result)

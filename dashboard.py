import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import subprocess
import os
import json

# --- Warna Tema ---
THEME_BG = "#F0F8FF"          # Biru sangat lembut untuk latar
THEME_PRIMARY = "#007BFF"     # Biru utama
THEME_ACCENT = "#003a77"      # Biru tua

st.set_page_config(
    layout="wide",
    page_title="Crypto Price Prediction",
    page_icon="💹"
)

# --- Gaya CSS Background ---
st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat&display=swap" rel="stylesheet">
    <style>
        /* Background halaman utama */
        .stApp {{
            background-color: {THEME_BG};
        }}

        /* Font global */
        html, body, [class*="css"] {{
            font-family: 'Montserrat', sans-serif !important;
        }}

        /* Warna latar sidebar */
        [data-testid="stSidebar"] {{
            background-color: #E6F0FA;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #003366;
        }}

        /* Warna teks di sidebar */
        [data-testid="stSidebar"] * {{
            color: #003366;
            font-weight: 600;
        }}

        /* Hover highlight pada elemen sidebar */
        [data-testid="stSidebar"] .css-1n76uvr:hover {{
            background-color: #cce0f5;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Konfigurasi ---
COINS = {
    "Bitcoin (BTC)": "btc",
    "Ethereum (ETH)": "eth",
    "Cardano (ADA)": "ada"
}

TIME_FRAMES = {
    "1 Bulan": 30,
    "3 Bulan": 90,
    "6 Bulan": 180,
    "1 Tahun": 365,
}

INDICATORS = ["RSI_14", "ATR_14", "FGI"]

# --- Header ---
st.markdown(f"<h1 style='color:{THEME_ACCENT};'>📊 Prediksi Harga Cryptocurrency</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    coin_label = st.selectbox("Pilih Cryptocurrency:", list(COINS.keys()))
    tf_label = st.selectbox("Pilih Time Frame:", list(TIME_FRAMES.keys()))
    indicator = st.selectbox("Pilih Indikator:", INDICATORS)
    show_data = st.checkbox("Tampilkan Tabel Harga", value=False)

    st.markdown("---")
    # st.subheader("📤 Prediksi Real-Time")
    if st.button("🔁 Jalankan Prediksi Hari Ini"):
        with st.spinner("Menjalankan prediksi harian..."):
            try:
                subprocess.run(["python", "run_daily_prediction.py"], check=True)
                st.cache_data.clear()  # Tambahkan ini agar cache dibersihkan
                st.success("✅ Prediksi berhasil dijalankan dan cache diperbarui.")
            except subprocess.CalledProcessError as e:
                st.error(f"Terjadi kesalahan: {e}")



coin = COINS[coin_label]
days = TIME_FRAMES[tf_label]

# --- Load Data ---
@st.cache_data
def load_data(coin):
    hist_path = f"1_data_realtime/historical/{coin}.csv"
    pred_path = f"1_data_realtime/prediction/{coin}_predicted.csv"
    proc_path = f"1_data_realtime/processed/{coin}_processed.csv"

    df_hist = pd.read_csv(hist_path)
    df_pred = pd.read_csv(pred_path)
    df_proc = pd.read_csv(proc_path)

    # Pastikan kolom Date bertipe date
    df_pred["Date"] = pd.to_datetime(df_pred["Date"]).dt.date
    df_hist["Date"] = pd.to_datetime(df_hist["Date"]).dt.date

    if "MA_14" not in df_proc.columns and "Close" in df_proc.columns:
        df_proc["MA_14"] = df_proc["Close"].rolling(window=14).mean()

    for df in [df_hist, df_pred, df_proc]:
        df["Date"] = pd.to_datetime(df["Date"])

    return df_hist, df_pred, df_proc

try:
    df_hist, df_pred, df_proc = load_data(coin)
    df_hist.sort_values("Date", inplace=True)
    df_proc.sort_values("Date", inplace=True)
    df_pred.sort_values("Date", inplace=True)

    df_hist_tf = df_hist[df_hist["Date"] >= df_hist["Date"].max() - pd.Timedelta(days=days)]
    df_proc_tf = df_proc[df_proc["Date"] >= df_proc["Date"].max() - pd.Timedelta(days=days)]
    df_pred = df_pred[df_pred["Date"] > df_hist["Date"].max()]

    # --- Highlight Prediksi Harga Terbaru ---
    if not df_pred.empty:
        latest_pred = df_pred.iloc[-1]
        pred_value = latest_pred["Predicted_Close"]
        pred_date = latest_pred["Date"].strftime("%d %b %Y")
        st.markdown(
            f"<div style='background-color:{THEME_PRIMARY};padding:20px;border-radius:10px;'>"
            f"<h2 style='color:white;text-align:center;'>🎯 Prediksi Harga {coin_label} pada {pred_date}: <b>{pred_value:,.2f} USD</b></h2>"
            f"</div><br>",
            unsafe_allow_html=True
        )

    # --- Grafik Harga dan Prediksi ---
    st.subheader(f"📈 Grafik Harga & Prediksi {coin_label.split()[0]}")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_hist_tf["Date"], y=df_hist_tf["Close"],
        mode="lines", name="Harga Historis",
        line=dict(color="darkblue", width=2)
    ))

    if "MA_14" in df_proc_tf.columns:
        fig.add_trace(go.Scatter(
            x=df_proc_tf["Date"], y=df_proc_tf["MA_14"],
            mode="lines", name="MA 14",
            line=dict(color="orange", width=1.5, dash="solid")
        ))

    # Titik akhir harga historis
    last_date = df_hist_tf["Date"].max()
    last_price = df_hist_tf[df_hist_tf["Date"] == last_date]["Close"].values[0]

    # Titik awal prediksi
    first_pred_date = df_pred["Date"].min()
    first_pred_price = df_pred[df_pred["Date"] == first_pred_date]["Predicted_Close"].values[0]

    # Garis transisi putus-putus dari historis ke prediksi
    fig.add_trace(go.Scatter(
        x=[last_date, first_pred_date],
        y=[last_price, first_pred_price],
        mode="lines",
        name="Transisi ke Prediksi",
        line=dict(color="darkblue", width=2, dash="dot"),
        showlegend=False
    ))


    if not df_pred.empty:
        fig.add_trace(go.Scatter(
            x=df_pred["Date"], y=df_pred["Predicted_Close"],
            mode="markers+text", name="Prediksi",
            marker=dict(size=10, color="darkblue"),
            text=[f"{v:.2f}" for v in df_pred["Predicted_Close"]],
            textposition="top center",
            showlegend=False
        ))

    if df_pred.empty:
        st.warning("Data prediksi belum tersedia.")

    fig.update_layout(
        height=450,
        margin=dict(t=20, b=20),
        hovermode="x unified",
        plot_bgcolor="#F0F8FF",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Indikator Teknis ---
    st.subheader(f"📊 Indikator: {indicator}")
    fig_ind = go.Figure()

    if indicator == "ATR_14":
        max_val = df_proc_tf["ATR_14"].max()
        min_val = df_proc_tf["ATR_14"].min()
        if max_val > 100:
            df_proc_tf["ATR_14_scaled"] = 100 * (df_proc_tf["ATR_14"] - min_val) / (max_val - min_val)
            y_data = df_proc_tf["ATR_14_scaled"]
            y_label = "ATR_14 (scaled)"
        else:
            y_data = df_proc_tf["ATR_14"]
            y_label = "ATR_14"
    else:
        y_data = df_proc_tf[indicator]
        y_label = indicator

    fig_ind.add_trace(go.Scatter(
        x=df_proc_tf["Date"], y=y_data,
        mode="lines", name=y_label,
        line=dict(color=THEME_ACCENT)
    ))

    if indicator in ["RSI_14", "ATR_14"]:
        fig_ind.update_yaxes(range=[0, 100])
        if indicator == "RSI_14":
            fig_ind.add_shape(type="line", x0=df_proc_tf["Date"].min(), x1=df_proc_tf["Date"].max(), y0=70, y1=70, line=dict(color="red", dash="dot"))
            fig_ind.add_shape(type="line", x0=df_proc_tf["Date"].min(), x1=df_proc_tf["Date"].max(), y0=30, y1=30, line=dict(color="green", dash="dot"))

    fig_ind.update_layout(
        height=300, margin=dict(t=10, b=20),
        hovermode="x unified",
        plot_bgcolor="#F0F8FF" 
    )
    st.plotly_chart(fig_ind, use_container_width=True)

    # --- Unduh Data Historis ---
    st.subheader("📥 Unduh Data Historis")
    st.download_button("⬇️ Download CSV", df_hist.to_csv(index=False), f"{coin}_historical.csv", "text/csv")

    # --- Tabel Data Opsional ---
    if show_data:
        st.subheader("📄 Tabel Harga")
        st.dataframe(df_hist_tf.tail(30), use_container_width=True)

except FileNotFoundError:
    st.error("❌ Data belum tersedia untuk koin ini.")
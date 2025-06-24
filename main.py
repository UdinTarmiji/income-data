import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Setup Page ---
st.set_page_config(page_title="Prediksi Pendapatan", page_icon="💼")

st.title("💼 Prediksi Pendapatan")
st.write("""
Aplikasi ini menggunakan model sederhana (Linear Regression) 
untuk memprediksi pendapatan berdasarkan jam kerja dan jumlah klien tiap minggu.
""")

# --- Load Dataset ---
url = "https://raw.githubusercontent.com/UdinTarmiji/income-data/main/data/income_data.csv"
data = pd.read_csv(url)

# --- Train the Model ---
x = data[["jam_kerja", "jumlah_klien"]]
y = data["pendapatan"]
model = LinearRegression()
model.fit(x, y)

# --- User Inputs ---
st.header("🔢 Masukkan Data")
jam = st.slider("🕒 Jam kerja per minggu:", 0, 100, 40)
klien = st.slider("👥 Jumlah klien:", 0, 20, 4)

# --- Predict Button ---
if st.button("🎯 Prediksi Sekarang"):
    hasil = model.predict([[jam, klien]])
    prediksi = int(hasil[0])
    st.success(f"💵 Prediksi Pendapatan: Rp {prediksi:,}")

    if prediksi >= 10_000_000:
        st.balloons()
        st.write("🔥 Wah pendapatanmu luar biasa!")
    elif prediksi >= 5_000_000:
        st.write

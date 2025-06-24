import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset dari GitHub
url = "https://raw.githubusercontent.com/UdinTarmiji/income-data/main/data/income_data.csv"
data = pd.read_csv(url)

# Latih model
x = data[["jam_kerja", "jumlah_klien"]]
y = data["pendapatan"]
model = LinearRegression()
model.fit(x, y)

# UI
st.set_page_config(page_title="Prediksi Pendapatan", page_icon="💰")
st.title("💼 AI Prediksi Pendapatan")
st.write("Masukkan jumlah jam kerja dan klien untuk memprediksi penghasilanmu!")

# Input dari user
jam = st.number_input("Masukkan jam kerja per minggu:", min_value=0, max_value=100, step=1)
klien = st.number_input("Masukkan jumlah klien:", min_value=0, max_value=50, step=1)

# Tombol Prediksi
if st.button("Prediksi Sekarang"):
    hasil = model.predict([[jam, klien]])
    st.success(f"💵 Prediksi Pendapatan: Rp {int(hasil[0]):,}")

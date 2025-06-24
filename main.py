import streamlit as st
import pandas as pd
import matplotlib.pylot as plt
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
st.title("Prediksi Pendapatan")
st.write("Masukkan jumlah jam kerja dan klien untuk memprediksi penghasilanmu!")
st.write("""
Aplikasi ini menggunakan model AI sederhana (Linear Regression) untuk memprediksi pendapatan
berdasarkan jam kerja dan jumlah klien yang kamu tangani tiap minggu.
""")

# Input dari user
jam = st.slider("Masukkan jam kerja per minggu:", min_value=0, max_value=100, step=1)
klien = st.slider("Masukkan jumlah klien:", min_value=0, max_value=50, step=1)

# Tombol Prediksi
if st.button("Prediksi Sekarang"):
    hasil = model.predict([[jam, klien]])
    st.success(f"💵 Prediksi Pendapatan: Rp {int(hasil[0]):,}")
    with st.expander("📊 Lihat data yang digunakan untuk pelatihan AI"):
        st.dataframe(data)
    if hasil[0] >= 10_000_000:
        st.balloons()
        st.write("🔥 Wah pendapatanmu luar biasa!")

# Visualisasi sederhana
fig, ax = plt.subplots()
ax.scatter(data["jam_kerja"], data["pendapatan"], color='blue', label='Data')
ax.set_xlabel("Jam Kerja")
ax.set_ylabel("Pendapatan")
ax.set_title("Jam Kerja vs Pendapatan")
st.pyplot(fig)


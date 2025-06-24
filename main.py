import pandas as pd
from sklearn.linear_model import LinearRegression

#load dataset
url ="https://raw.githubusercontent.com/UdinTarmiji/income-data/main/data/income_data.csv"
data = pd.read_csv(url)

#siapkan data
x = data[["jam_kerja", "jumlah_klien"]]
y = data["pendapatan"]

#latih model ai
model = LinearRegression()
model.fit(x, y)

#prediksi pendapatan
jam = 40
klien = 4
prediksi = model.predict([[jam, klien]])


print(f"Prediksi Pendapatan: Ro {int(prediksi[0]):,}")

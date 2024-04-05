import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.linear_model import LinearRegression

# Erstellung DataFrame
start = dt.datetime(2019,1,1)
end = dt.datetime(2022,1,1)
prediction_days = 60
bitcoin = yf.download("BTC-USD", start=start, end=end)
bitcoin = bitcoin[["Adj Close"]]
bitcoin_data = bitcoin.values

# Skalierung Daten (0 bis 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(bitcoin)

# Aufteilung Trainings-Set / Test-Set (um Länge zu prüfen: len() nutzen!)
training_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_len]
test_data = scaled_data[training_len:]

# Trennung train_data in x (Features) und y (Labels) + Befüllung
x_train = []
y_train = []
for i in range(prediction_days, len(train_data)):
    x_train.append(train_data[i - prediction_days:i])
    y_train.append(train_data[i])

# Falls nötig: reshape der testdaten zur Vorbereitung auf das Model (reshape geht nur bei arrays, deswegen Umwandlung mit np.array)
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape((len(x_train), -1))

# Trennung test_data in x (Features) und y (Labels) + Befüllung
x_test = []
y_test = []
for i in range(prediction_days, len(test_data)):
    x_test.append(test_data[i - prediction_days:i])
    y_test.append(test_data[i])

# Falls nötig: reshape der testdaten zur Vorbereitung auf das Model (reshape geht nur bei arrays, deswegen Umwandlung mit np.array)
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape((len(x_test), -1))

# Auswahl Model + Training
model = LinearRegression()
model.fit(x_train, y_train)

# Validierung Model + Rückskalierung der predictions und label (y_test)
test_prediction = model.predict(x_test)
test_prediction = scaler.inverse_transform(test_prediction)

# Berechnung root mean squared error (gibt die durchschnittliche Abweichung zwischen Vorhersage und Label wieder)
rmse = np.sqrt(np.mean(test_prediction- y_test)**2)
print(rmse)

# Definition aller Variablen für Visualisierung (historische Daten (train), labeldaten (valid #1) und Vorhersage des Models (valid #2)
train = bitcoin[: training_len]
valid = bitcoin[training_len : -prediction_days] #1
valid["Test_Prediction"] = test_prediction #2

# Visualisierung der prediction vs. labeldaten
plt.figure(figsize=(16,8))
plt.title("Model")
plt.ylabel("BTC in $")
plt.xlabel("Date")
plt.plot(train["Adj Close"], label="Past")
plt.plot(valid[["Adj Close"]], label="Future")
plt.plot(valid[["Test_Prediction"]], label="Test")
plt.legend()
plt.show()
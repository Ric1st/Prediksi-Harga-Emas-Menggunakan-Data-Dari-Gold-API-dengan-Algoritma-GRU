import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/gold_price.csv")
df["price"] = df["price"].astype(float)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["price"]])

# simpan scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# =========================
# WINDOWING
# =========================
WINDOW = 5
X, y = [], []

for i in range(len(scaled_data) - WINDOW):
    X.append(scaled_data[i:i+WINDOW])
    y.append(scaled_data[i+WINDOW])

X = np.array(X)            # shape: (samples, 5, 1)
y = np.array(y).reshape(-1, 1)   # shape: (samples, 1)  <-- penting

# =========================
# BUILD GRU MODEL
# =========================
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(WINDOW, 1)),
    Dropout(0.2),

    GRU(32),
    Dropout(0.2),

    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

filename = "model.keras"

checkpoint = ModelCheckpoint(
    filename, monitor="loss", save_best_only=True, verbose=1
)


# =========================
# TRAIN
# =========================
model.fit(
    X, y,
    epochs=20,
    batch_size=16,
    callbacks=[checkpoint]
)

print("Training selesai! Model tersimpan → ", filename)

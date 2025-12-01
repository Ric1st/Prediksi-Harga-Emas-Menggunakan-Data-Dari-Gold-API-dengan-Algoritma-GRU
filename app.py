from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = "model.keras"
SCALER_PATH = "scaler.pkl"
CSV_PATH = "data/gold_price.csv"

model = load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

FORMAT_MAP = {
    "minute": "%Y-%m-%d %H:%M",
    "hour": "%Y-%m-%d %H",
    "day": "%Y-%m-%d",
    "week": "%Y-%m-%d",
    "month": "%Y-%m",
    "year": "%Y"
}

@app.route("/")
def index():
    return render_template("index.html")


def load_and_prepare_df():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        raise RuntimeError(f"CSV not found: {CSV_PATH}")

    if "timestamp" not in df.columns or "price" not in df.columns:
        raise RuntimeError("CSV harus memiliki kolom 'timestamp' dan 'price'")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    return df


@app.route("/api/data/<filter_type>")
def get_filtered_data(filter_type):
    try:
        df = load_and_prepare_df()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if filter_type == "minute":
        df_res = df.copy()
    else:
        df_idx = df.set_index("timestamp")
        if filter_type == "hour":
            df_res = df_idx.resample("1H").last().dropna()
        elif filter_type == "day":
            df_res = df_idx.resample("1D").last().dropna()
        elif filter_type == "week":
            df_res = df_idx.resample("1W").last().dropna()
        elif filter_type == "month":
            df_res = df_idx.resample("1M").last().dropna()
        elif filter_type == "year":
            df_res = df_idx.resample("1Y").last().dropna()
        else:
            df_res = df_idx.resample("1D").last().dropna()

    if "timestamp" in df_res.columns:
        timestamps_dt = pd.to_datetime(df_res["timestamp"])
        prices = df_res["price"].astype(float).tolist()
    else:
        timestamps_dt = pd.to_datetime(df_res.index.to_series())
        prices = df_res["price"].astype(float).tolist()

    if len(prices) == 0:
        return jsonify({"error": "Tidak ada data setelah resample"}), 400

    fmt = FORMAT_MAP.get(filter_type, "%Y-%m-%d")
    labels = [dt.strftime(fmt) for dt in list(timestamps_dt)]

    current_price = float(df["price"].iloc[-1])

    raw_prices = df["price"].astype(float).tolist()
    if len(raw_prices) >= 5:
        last_window = np.array(raw_prices[-5:]).reshape(-1, 1)
    else:
        pad_count = max(0, 5 - len(raw_prices))
        pad_vals = [raw_prices[0]] * pad_count + raw_prices
        last_window = np.array(pad_vals[-5:]).reshape(-1, 1)

    try:
        scaled_window = scaler.transform(last_window)
        X_input = np.array([scaled_window])
        pred_scaled = model.predict(X_input)[0][0]
        predicted_price = float(scaler.inverse_transform([[pred_scaled]])[0][0])
    except Exception as e:
        print("Prediction error:", e)
        predicted_price = current_price

    if len(raw_prices) >= 2:
        prev_price = float(raw_prices[-2])
        change_percent = ((current_price - prev_price) / prev_price) * 100
    else:
        change_percent = 0.0

    # Build response
    resp = {
        "timestamps": labels,
        "prices": [float(x) for x in prices],
        "current_price": float(current_price),
        "predicted_price": float(predicted_price),
        "change_percent": round(change_percent, 2)
    }

    return jsonify(resp)


if __name__ == "__main__":
    app.run(debug=True)

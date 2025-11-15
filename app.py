# app.py — Render Web Service + Supabase + LSTM Inference
import os
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client

# -----------------------------
# CONFIG
# -----------------------------
WINDOW = 15
FEATURES = 4

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# LOAD MODEL AND SCALERS
# -----------------------------
model = tf.keras.models.load_model("saved_model")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

app = FastAPI()

# -----------------------------
# MANUAL PREDICTION INPUT
# -----------------------------
class Reading(BaseModel):
    air_quality: float
    temperature: float
    humidity: float
    anomaly_flag: int = 0

class PredictionRequest(BaseModel):
    window: list[Reading]

@app.post("/predict")
def manual_predict(req: PredictionRequest):

    if len(req.window) != WINDOW:
        return {"error": f"Expected {WINDOW} readings, got {len(req.window)}"}

    X = np.array([
        [
            r.air_quality,
            r.temperature,
            r.humidity,
            r.anomaly_flag
        ]
        for r in req.window
    ], dtype=np.float32)

    X_scaled = scaler_X.transform(X).reshape(1, WINDOW, FEATURES)

    pred_scaled = model.predict(X_scaled)[0][0]
    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]

    category = (
        "Low" if pred < 3600 else
        "High" if pred > 4200 else
        "Medium"
    )

    return {
        "predicted_eggs": float(pred),
        "category": category
    }

# -----------------------------
# LIVE SUPABASE PREDICTION
# -----------------------------
@app.get("/predict-live")
def predict_live():

    # Fetch latest 15 rows
    response = (
        supabase
        .table("sensor_readings")
        .select("*")
        .order("timestamp", desc=True)
        .limit(WINDOW)
        .execute()
    )

    data = response.data

    if not data or len(data) < WINDOW:
        return {"error": f"Not enough data. Needed {WINDOW}, found {len(data)}"}

    # Reverse to chronological
    data = data[::-1]

    X = np.array([
        [
            row["air_quality"],
            row["temperature"],
            row["humidity"],
            row["anomaly"]
        ]
        for row in data
    ], dtype=np.float32)

    X_scaled = scaler_X.transform(X).reshape(1, WINDOW, FEATURES)

    pred_scaled = model.predict(X_scaled)[0][0]
    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]

    category = (
        "Low" if pred < 3600 else
        "High" if pred > 4200 else
        "Medium"
    )

    return {
        "predicted_eggs": float(pred),
        "category": category
    }

# -----------------------------
# ROOT CHECK
# -----------------------------
@app.get("/")
def root():
    return {"status": "API is running", "model": "PURAIR LSTM"}

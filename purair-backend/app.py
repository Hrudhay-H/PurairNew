import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load SavedModel directory
model = tf.keras.models.load_model("saved_model_purair")

# Load scalers
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

WINDOW = 15
FEATURES = 4


class Reading(BaseModel):
    air_quality: float
    temperature: float
    humidity: float
    anomaly_flag: int = 0


class PredictionRequest(BaseModel):
    window: list[Reading]


@app.post("/predict")
def predict_egg_production(req: PredictionRequest):

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

    # Scale features
    X_scaled = scaler_X.transform(X)
    X_scaled = X_scaled.reshape(1, WINDOW, FEATURES)

    # Predict
    y_scaled = model.predict(X_scaled)[0][0]

    # Reverse scaling
    pred = scaler_y.inverse_transform([[y_scaled]])[0][0]

    # Categorize
    if pred < 3600:
        cat = "Low"
    elif pred > 4200:
        cat = "High"
    else:
        cat = "Medium"

    return {
        "predicted_eggs": float(pred),
        "category": cat,
        "confidence": 0.90
    }


@app.get("/")
def home():
    return {"message": "PURAIR LSTM API running on Render."}

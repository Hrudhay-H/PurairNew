import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ---------------------------
# Load your model + scalers
# ---------------------------
model = tf.keras.models.load_model("PURAIR_LSTM_final.keras", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

WINDOW = 15
FEATURES = 4  # MQ135, Temp, Humidity, Anomaly


# ---------------------------
# Request Models
# ---------------------------
class Reading(BaseModel):
    air_quality: float
    temperature: float
    humidity: float
    anomaly_flag: int = 0


class PredictionRequest(BaseModel):
    window: list[Reading]


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    return {"status": "PURAIR ML API Running 🚀"}


@app.post("/predict")
def predict(req: PredictionRequest):

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

    # Predict (scaled output)
    y_scaled = model.predict(X_scaled)[0][0]

    # Reverse scale
    pred = scaler_y.inverse_transform([[y_scaled]])[0][0]

    # Categorize
    if pred < 3600:
        category = "Low"
    elif pred > 4200:
        category = "High"
    else:
        category = "Medium"

    return {
        "predicted_eggs": float(pred),
        "category": category,
        "confidence": 0.92
    }

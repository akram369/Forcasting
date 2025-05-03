from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Champion Model API")

# Champion model path
CHAMPION_MODEL_PATH = "models/champion_model.pkl"

# Define input schema
class ForecastInput(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    lag_5: float
    lag_6: float
    lag_7: float
    is_promo: int
    is_holiday: int

@app.post("/predict")
def predict(input_data: ForecastInput):
    if not os.path.exists(CHAMPION_MODEL_PATH):
        return {"error": "Champion model not found."}

    model = joblib.load(CHAMPION_MODEL_PATH)

    # Input feature order must match model training
    features = [[
        input_data.lag_1,
        input_data.lag_2,
        input_data.lag_3,
        input_data.lag_4,
        input_data.lag_5,
        input_data.lag_6,
        input_data.lag_7,
        input_data.is_promo,
        input_data.is_holiday
    ]]

    forecast = model.predict(features)[0]

    return {
        "forecast": float(forecast),
        "model": "Champion XGBoost",
        "status": "success"
    }

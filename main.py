from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import os

# ---- Config ----
MODEL_DIR = "gym_ai_models"
TARGETS = ["exercises", "equipment", "diet", "recommendation"]

# ---- App ----
app = FastAPI(title="AI Fitness API", version="1.0.0")

# CORS: allow local dev and any deployed frontends you use
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # add your deployed frontend origins here (Netlify/Vercel/etc.)
    # "https://your-frontend.example.com"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + ["*"],  # keep "*" while debugging; tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load metadata at startup (small, fast) ----
input_features_path = os.path.join(MODEL_DIR, "input_features.pkl")
category_mappings_path = os.path.join(MODEL_DIR, "category_mappings.pkl")

if not os.path.exists(input_features_path) or not os.path.exists(category_mappings_path):
    raise RuntimeError("Missing metadata files in gym_ai_models/: input_features.pkl or category_mappings.pkl")

input_features = joblib.load(input_features_path)
category_mappings = joblib.load(category_mappings_path)

# ---- Lazy model registry ----
_models: dict[str, lgb.Booster] = {}

def get_model(target: str) -> lgb.Booster:
    if target not in _models:
        model_path = os.path.join(MODEL_DIR, f"{target}_model.txt")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Missing model file: {model_path}")
        booster = lgb.Booster(model_file=model_path)
        # reduce thread usage on small CPU instances
        booster.reset_parameter({"num_threads": 1})
        _models[target] = booster
        print(f"✅ Loaded model for {target}")
    return _models[target]

def decode_prediction(pred: int, target: str) -> str:
    return category_mappings[target].get(pred, "Unknown")

def generate_recommendations(user_data: dict) -> dict:
    # Ensure all expected features exist; fill unknowns with 0
    df = pd.DataFrame([user_data])
    df = df.reindex(columns=input_features, fill_value=0)

    out = {}
    for target in TARGETS:
        model = get_model(target)
        proba = model.predict(df)

        # binary vs multiclass handling
        if model.params.get("objective") == "binary":
            pred_idx = int(proba > 0.5)
        else:
            pred_idx = int(np.argmax(proba, axis=1)[0])

        out[target] = decode_prediction(pred_idx, target)
    return out

# ---- Schemas ----
class PredictIn(BaseModel):
    sex: str | None = None
    age: int | float | None = None
    height: int | float | None = None
    weight: int | float | None = None
    hypertension: str | int | None = None
    diabetes: str | int | None = None
    fitness_goal: str | None = None
    fitness_type: str | None = None

# ---- Routes ----
@app.get("/")
def root():
    return {"message": "AI Fitness API is running!"}

@app.get("/debug/features")
def debug_features():
    return {"input_features": input_features}

@app.post("/predict")
async def predict(body: PredictIn):
    try:
        # Convert to dict; keep keys as your training pipeline expects
        user_data = {k: v for k, v in body.dict().items() if v is not None}
        if not user_data:
            raise HTTPException(status_code=400, detail="Empty payload")

        preds = generate_recommendations(user_data)
        return {"success": True, "predictions": preds}
    except Exception as e:
        return {"success": False, "error": str(e)}

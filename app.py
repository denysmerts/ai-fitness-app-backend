from flask import Flask, request, jsonify
from flask_cors import CORS
import lightgbm as lgb
import pandas as pd
import joblib
import os
import numpy as np


app = Flask(__name__)

# ✅ Allow both local frontend & deployed frontend
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://ai-fitness-app-backend-3.onrender.com"]}})

# =========================
# Load Models & Metadata
# =========================
MODEL_DIR = "gym_ai_models"
target_columns = ["exercises", "equipment", "diet", "recommendation"]

models = {}
for target in target_columns:
    model_path = os.path.join(MODEL_DIR, f"{target}_model.txt")
    model = lgb.Booster(model_file=model_path)

    # ✅ Reduce CPU threads = less memory spikes on Render Free Tier
    model.reset_parameter({"num_threads": 1})

    models[target] = model
    print(f"✅ Loaded model for {target}")

input_features = joblib.load(os.path.join(MODEL_DIR, "input_features.pkl"))
category_mappings = joblib.load(os.path.join(MODEL_DIR, "category_mappings.pkl"))
print("✅ Loaded input features and category mappings")


# =========================
# Helper Functions
# =========================
def decode_prediction(pred, target_column):
    return category_mappings[target_column].get(pred, "Unknown") if target_column in category_mappings else pred


def generate_recommendations(user_data):
    user_df = pd.DataFrame([user_data])
    user_df = user_df.reindex(columns=input_features, fill_value=0)

    recommendations = {}
    for target, model in models.items():
        pred_prob = model.predict(user_df)
        if model.params.get('objective') == 'binary':
            pred = int(pred_prob > 0.5)
        else:
            pred = int(np.argmax(pred_prob, axis=1)[0])
        recommendations[target] = decode_prediction(pred, target)
    return recommendations


# =========================
# Flask Routes
# =========================
@app.route('/')
def home():
    return jsonify({"message": "AI Fitness API is running!"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = request.get_json()
        predictions = generate_recommendations(user_data)
        return jsonify({"success": True, "predictions": predictions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ✅ Debug route to inspect feature requirements
@app.route('/debug/features', methods=['GET'])
def debug_features():
    return jsonify({"input_features": input_features})


# =========================
# Run Flask locally
# =========================
if __name__ == '__main__':
    app.run()

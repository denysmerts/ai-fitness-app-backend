from flask import Flask, request, jsonify
from flask_cors import CORS
import lightgbm as lgb
import pandas as pd
import joblib
import os
import numpy as np
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)

# =========================
# Load Models & Metadata
# =========================
MODEL_DIR = "gym_ai_models"
target_columns = ["exercises", "equipment", "diet", "recommendation"]

# Load LightGBM models
models = {}
for target in target_columns:
    model_path = os.path.join(MODEL_DIR, f"{target}_model.txt")
    models[target] = lgb.Booster(model_file=model_path)
    print(f"✅ Loaded model for {target}")

# Load input features and category mappings
input_features = joblib.load(os.path.join(MODEL_DIR, "input_features.pkl"))
category_mappings = joblib.load(os.path.join(MODEL_DIR, "category_mappings.pkl"))
print("✅ Loaded input features and category mappings")

# =========================
# Helper Functions
# =========================
def decode_prediction(pred, target_column):
    """Convert numerical prediction back to original category label."""
    return category_mappings[target_column].get(pred, "Unknown") if target_column in category_mappings else pred

def generate_recommendations(user_data):
    """Generate recommendations from pre-trained models."""
    user_df = pd.DataFrame([user_data])
    user_df = user_df.reindex(columns=input_features, fill_value=0)

    recommendations = {}
    for target, model in models.items():
        pred_prob = model.predict(user_df, num_iteration=model.best_iteration)
        # Detect binary vs multiclass
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

# =========================
# Run Flask
# =========================
# if __name__ == "__main__":
#     public_url = ngrok.connect(5000)
#     print("Public URL:", public_url)
#     from app import app  # or just use your Flask instance directly
#     app.run(port=5000)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

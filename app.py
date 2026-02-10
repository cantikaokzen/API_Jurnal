from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flasgger import Swagger, swag_from

import logging
from datetime import datetime
from catboost import CatBoostClassifier
import json
import numpy as np

# =========================
# Load resources
# =========================
try:
    # Load Preprocessing Params
    with open("preprocessing.json", "r") as f:
        preproc_params = json.load(f)
    
    # Extract mean and scale for the selected features
    # We know the selected indices from the JSON
    selected_indices = preproc_params["selector"]["indices"]
    all_means = preproc_params["scaler"]["mean"]
    all_scales = preproc_params["scaler"]["scale"]
    
    # Filter means and scales for the 10 selected features
    MODEL_MEANS = [all_means[i] for i in selected_indices]
    MODEL_SCALES = [all_scales[i] for i in selected_indices]
    
    # Load Model
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    
except Exception as e:
    logging.error(f"Failed to load model or parameters: {e}")
    raise e

# =========================
# Logging
# =========================
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# Flask setup
# =========================
app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

@app.route("/ui")
def ui():
    return send_from_directory(".", "index.html")

@app.route("/")
def home():
    return jsonify({"message": "Breast Cancer API (Lite)"})

# =========================
# Swagger spec
# =========================
PREDICT_SWAGGER = {
    "tags": ["Prediction"],
    "summary": "Predict Breast Cancer Classification",
    "description": "Menerima input fitur dan mengembalikan hasil klasifikasi kanker payudara.",
    "consumes": ["application/json"],
    "produces": ["application/json"],
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "radius_mean": {"type": "number", "example": 11.2},
                    "perimeter_mean": {"type": "number", "example": 71.2},
                    "area_mean": {"type": "number", "example": 380.0},
                    "concavity_mean": {"type": "number", "example": 0.03},
                    "concave_points_mean": {"type": "number", "example": 0.015},
                    "radius_worst": {"type": "number", "example": 12.6},
                    "perimeter_worst": {"type": "number", "example": 80.0},
                    "area_worst": {"type": "number", "example": 470.0},
                    "concavity_worst": {"type": "number", "example": 0.08},
                    "concave_points_worst": {"type": "number", "example": 0.03},
                },
                "required": [
                    "radius_mean", "perimeter_mean", "area_mean",
                    "concavity_mean", "concave_points_mean",
                    "radius_worst", "perimeter_worst", "area_worst",
                    "concavity_worst", "concave_points_worst"
                ]
            }
        }
    ],
    "responses": {
        200: {
            "description": "Prediction result",
            "schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "example": True},
                    "predicted_label": {"type": "string", "example": "Malignant"},
                    "probabilities": {
                        "type": "object",
                        "properties": {
                            "Benign": {"type": "number", "example": 0.0877},
                            "Malignant": {"type": "number", "example": 0.9123}
                        }
                    },
                    "timestamp": {"type": "string", "example": "2025-06-15T12:34:56.789123"}
                }
            }
        },
        400: {"description": "Bad request"},
        500: {"description": "Internal server error"}
    }
}

@app.route("/predict", methods=["POST"])
@swag_from(PREDICT_SWAGGER)
def predict():
    try:
        data = request.get_json(force=True)
        logging.info(f"Received prediction request: {data}")

        # Exact order matching the selected indices (0, 2, 3, 6, 7, 20, 22, 23, 26, 27)
        # Matches the user input fields
        user_feature_names = [
            "radius_mean", "perimeter_mean", "area_mean",
            "concavity_mean", "concave_points_mean",
            "radius_worst", "perimeter_worst", "area_worst",
            "concavity_worst", "concave_points_worst"
        ]

        missing = [f for f in user_feature_names if f not in data]
        if missing:
            return jsonify({"success": False, "error": f"Missing fields: {missing}"}), 400

        # Construct vector
        raw_values = [float(data[f]) for f in user_feature_names]
        
        # Manual Scaling: (x - mean) / scale
        scaled_values = []
        for val, mean, scale in zip(raw_values, MODEL_MEANS, MODEL_SCALES):
            scaled_values.append((val - mean) / scale)

        # Predict
        # CatBoost expects a list of lists (samples)
        X = [scaled_values] 
        
        predicted = int(model.predict(X)[0])
        proba_all = model.predict_proba(X)[0]
        classes = list(model.classes_)

        idx_1 = classes.index(1) if 1 in classes else None
        if idx_1 is None:
            prob_malignant = float(max(proba_all))
        else:
            prob_malignant = float(proba_all[idx_1])

        prob_benign = float(1 - prob_malignant)

        label_map = {0: "Benign", 1: "Malignant"}
        pred_label = label_map.get(predicted, str(predicted))

        return jsonify({
            "success": True,
            "predicted_label": pred_label,
            "probabilities": {
                "Benign": round(prob_benign, 4),
                "Malignant": round(prob_malignant, 4)
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

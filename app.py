from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flasgger import Swagger, swag_from

import pandas as pd
import joblib
import logging
from datetime import datetime
import sys

from imblearn.base import BaseSampler
from sklearn.neighbors import LocalOutlierFactor

class LOFResampler(BaseSampler):
    _sampling_type = "clean-sampling"
    _parameter_constraints = {}

    def __init__(self, n_neighbors=20, contamination=0.05):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.sampling_strategy = "auto"

    def _fit_resample(self, X, y):
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination
        )
        pred = lof.fit_predict(X)
        mask = (pred == 1)
        return X[mask], y[mask]

# IMPORTANT: so pickle that references __main__.LOFResampler can find it
setattr(sys.modules["__main__"], "LOFResampler", LOFResampler)

# =========================
# Load model
# =========================
model = joblib.load("catboost_pipeline.joblib")
required_cols = list(model.feature_names_in_)

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
    return send_from_directory(".", "index.html")  # index.html sejajar app.py

@app.route("/")
def home():
    return jsonify({"message": "Test API"})

# =========================
# Swagger spec (same as yours)
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

        user_feature_names = [
            "radius_mean", "perimeter_mean", "area_mean",
            "concavity_mean", "concave_points_mean",
            "radius_worst", "perimeter_worst", "area_worst",
            "concavity_worst", "concave_points_worst"
        ]

        missing = [f for f in user_feature_names if f not in data]
        if missing:
            return jsonify({"success": False, "error": f"Missing fields: {missing}"}), 400

        X = pd.DataFrame([{
            f: float(data[f]) for f in user_feature_names
        }])

        # add missing cols required by model with 0.0
        for col in required_cols:
            if col not in X.columns:
                X[col] = 0.0

        X = X[required_cols]

        predicted = int(model.predict(X)[0])
        proba_all = model.predict_proba(X)[0]
        classes = list(model.classes_)

        idx_1 = classes.index(1) if 1 in classes else None
        if idx_1 is None:
            # fallback if classes are not [0,1]
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
    # For local only. On deploy, use gunicorn (see note below).
    app.run(host="0.0.0.0", port=5000, debug=True)
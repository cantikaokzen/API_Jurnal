from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flasgger import Swagger, swag_from
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import sys
from lof_resampler import LOFResampler

# Biar pickle yang nyari __main__.LOFResampler ketemu
setattr(sys.modules["__main__"], "LOFResampler", LOFResampler)

# Load Model
model = joblib.load('catboost_pipeline.joblib')
required_cols = list(model.feature_names_in_)

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Init Flask app
app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

@app.route("/ui")
def ui():
    return send_from_directory(".", "index.html")  # index.html sejajar app.py

@app.route('/')
def home():
    return jsonify({"message": "Test API"})

@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'summary': 'Predict Breast Cancer Classification',
    'description': 'Menerima input fitur dan mengembalikan hasil klasifikasi kanker payudara.',
    'consumes': ['application/json'],
    'produces': ['application/json'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'radius_mean': {'type': 'number', 'example': 11.2},
                    'perimeter_mean': {'type': 'number', 'example': 71.2},
                    'area_mean': {'type': 'number', 'example': 380.0},
                    'concavity_mean': {'type': 'number', 'example': 0.03},
                    'concave_points_mean': {'type': 'number', 'example': 0.015},
                    'radius_worst': {'type': 'number', 'example': 12.6},
                    'perimeter_worst': {'type': 'number', 'example': 80.0},
                    'area_worst': {'type': 'number', 'example': 470.0},
                    'concavity_worst': {'type': 'number', 'example': 0.08},
                    'concave_points_worst': {'type': 'number', 'example': 0.03},
                },
                'required': [
                    'radius_mean', 'perimeter_mean', 'area_mean',
                    'concavity_mean', 'concave_points_mean',
                    'radius_worst', 'perimeter_worst', 'area_worst', 
                    'concavity_worst', 'concave_points_worst'
                ]
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Prediction result',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean', 'example': True},
                    'prediction': {'type': 'integer', 'example': 1},
                    'probability': {
                        'type': 'object',
                        'properties': {
                            'positive': {'type': 'number', 'example': 0.9123},
                            'negative': {'type': 'number', 'example': 0.0877}
                        }
                    },
                    'timestamp': {'type': 'string', 'example': '2025-06-15T12:34:56.789123'}
                }
            }
        },
        500: {
            'description': 'Internal server error',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean', 'example': False},
                    'error': {'type': 'string', 'example': 'Internal server error'}
                }
            }
        }
    }
})

def predict():
    try:
        data = request.get_json(force=True)
        logging.info(f"Received prediction request: {data}")

        # 10 fitur yang kamu minta user isi
        user_feature_names = [
            'radius_mean', 'perimeter_mean', 'area_mean',
            'concavity_mean', 'concave_points_mean',
            'radius_worst', 'perimeter_worst', 'area_worst',
            'concavity_worst', 'concave_points_worst'
        ]

        # validasi: 10 fitur wajib harus ada
        missing = [f for f in user_feature_names if f not in data]
        if missing:
            return jsonify({"success": False, "error": f"Missing fields: {missing}"}), 400

        # buat DF dari input 10 fitur
        X = pd.DataFrame([{
            f: float(data[f]) for f in user_feature_names
        }])

        # tambahkan fitur lain yang dibutuhkan model (yang tidak dikirim) dengan 0.0
        for col in required_cols:
            if col not in X.columns:
                X[col] = 0.0

        # pastikan urutan kolom sama persis seperti training
        X = X[required_cols]

        predicted = int(model.predict(X)[0])
        proba_all = model.predict_proba(X)[0]
        classes = list(model.classes_)

        # ambil probabilitas kelas 1 dengan aman
        idx_1 = classes.index(1)
        prob_malignant = float(proba_all[idx_1])
        prob_benign = float(1 - prob_malignant)

        label_map = {0: "Benign", 1: "Malignant"}
        pred_label = label_map[predicted]

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

# ==========================================================
# ExoHabitAI Backend API
# Flask Application for Exoplanet Habitability Prediction
# ==========================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import prepare_features, FEATURE_ORDER

import joblib
import pandas as pd
import os
import logging


# --------------------------------
# Initialize Flask Application
# --------------------------------

app = Flask(__name__)
CORS(app)


# --------------------------------
# Project Paths  (relative to this file)
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_candidates = [
    os.path.join(BASE_DIR, "best_model.pkl"),
    os.path.join(BASE_DIR, "models", "best_model.pkl"),
    os.path.join(BASE_DIR, "..", "models", "best_model.pkl"),
]
MODEL_PATH = next((p for p in _candidates if os.path.exists(p)), _candidates[0])

_rank_candidates = [
    os.path.join(BASE_DIR, "habitability_ranked.csv"),
    os.path.join(BASE_DIR, "data", "processed", "habitability_ranked.csv"),
    os.path.join(BASE_DIR, "..", "data", "processed", "habitability_ranked.csv"),
]
RANK_DATA_PATH = next((p for p in _rank_candidates if os.path.exists(p)), _rank_candidates[0])

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")


# --------------------------------
# Logging Configuration
# --------------------------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.info("ExoHabitAI API starting...")


# --------------------------------
# Load Trained Model
# --------------------------------

model = None

try:
    model = joblib.load(MODEL_PATH)
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Model loading failed: {e}")


# --------------------------------
# Health Check Endpoint
# --------------------------------

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "ExoHabitAI Backend API Running",
        "model_loaded": model is not None
    })


# --------------------------------
# Model Information Endpoint
# --------------------------------

@app.route("/model-info", methods=["GET"])
def model_info():
    # Use the model's own feature list if available, otherwise fall back to utils.py list
    features = (
        list(model.feature_names_in_)
        if model is not None and hasattr(model, "feature_names_in_")
        else FEATURE_ORDER
    )
    return jsonify({
        "model_name": "ExoHabitAI",
        "version": "1.0",
        "features_used": features,
        "note": "Stellar_Flux is derived server-side from Stellar_Luminosity / Semi_Major_Axis^2. Do not send it as input."
    })


# --------------------------------
# Prediction Endpoint
# --------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    try:
        data = request.get_json()

        if data is None:
            return jsonify({"status": "error", "message": "Invalid or missing JSON body"}), 400

        logging.info(f"Prediction request received: {data}")

        # prepare_features validates input, computes Stellar_Flux, returns a DataFrame
        features = prepare_features(data)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        label = "Potentially Habitable" if int(prediction) == 1 else "Non-Habitable"

        logging.info(f"Prediction completed | prediction={prediction} | probability={probability:.6f}")

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "label": label,
            "habitability_probability": round(float(probability), 6)
        })

    except ValueError as e:
        logging.warning(f"Validation error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"status": "error", "message": "Internal prediction error"}), 500


# --------------------------------
# Planet Ranking Endpoint
# --------------------------------

@app.route("/rank", methods=["GET"])
def rank():
    try:
        if not os.path.exists(RANK_DATA_PATH):
            return jsonify({"status": "error", "message": "Ranking dataset not found"}), 404

        df = pd.read_csv(RANK_DATA_PATH)

        # Validate n parameter
        try:
            n = int(request.args.get("n", 10))
            n = max(1, min(n, 100))   # clamp between 1 and 100
        except (ValueError, TypeError):
            n = 10

        top_planets = df.head(n)

        logging.info(f"Top planet ranking requested | n={n}")

        return jsonify({
            "status": "success",
            "count": len(top_planets),
            "top_planets": top_planets.to_dict(orient="records")
        })

    except Exception as e:
        logging.error(f"Ranking error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# --------------------------------
# Run Flask Server
# --------------------------------

if __name__ == "__main__":
    logging.info("Starting Flask development server")
    # Set debug=False for production deployment
    # Use an environment variable to control this safely
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)

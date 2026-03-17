# ==========================================================
# ExoHabitAI Backend API
# Flask Application for Exoplanet Habitability Prediction
# ==========================================================

# --------------------------------
# Import Required Libraries
# --------------------------------

from flask import Flask, request, jsonify
from utils import prepare_features

import joblib
import pandas as pd
import os
import logging


# --------------------------------
# Initialize Flask Application
# --------------------------------

app = Flask(__name__)


# --------------------------------
# Project Paths
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "../models/best_model.pkl")
RANK_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/habitability_ranked.csv")

LOG_DIR = os.path.join(BASE_DIR, "../logs")
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
# Load Trained Machine Learning Model
# --------------------------------

model = None

try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully")

except Exception as e:
    logging.error(f"Model loading failed: {e}")


# --------------------------------
# Health Check Endpoint
# --------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "ExoHabitAI Backend API Running"
    })


# --------------------------------
# Model Information Endpoint
# --------------------------------

@app.route("/model-info", methods=["GET"])
def model_info():

    return jsonify({
        "model_name": "ExoHabitAI",
        "version": "1.0",
        "features_used": [
            "planet_radius",
            "planet_mass",
            "orbital_period"
        ]
    })


# --------------------------------
# Prediction Endpoint
# --------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 500

    try:

        # Get JSON request
        data = request.get_json()

        if data is None:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON input"
            }), 400

        logging.info(f"Prediction request received: {data}")

        # Prepare features using preprocessing pipeline
        features = prepare_features(data)

        # Run model prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        logging.info(
            f"Prediction completed | prediction={prediction} | probability={probability}"
        )

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "habitability_probability": float(probability)
        })

    except Exception as e:

        logging.error(f"Prediction error: {e}")

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# --------------------------------
# Planet Ranking Endpoint
# --------------------------------

@app.route("/rank", methods=["GET"])
def rank():

    try:

        if not os.path.exists(RANK_DATA_PATH):

            return jsonify({
                "status": "error",
                "message": "Ranking dataset not found"
            }), 404

        df = pd.read_csv(RANK_DATA_PATH)

        # Top 10 habitable planets
        top_planets = df.head(10)

        logging.info("Top planet ranking requested")

        return jsonify({
            "status": "success",
            "count": len(top_planets),
            "top_planets": top_planets.to_dict(orient="records")
        })

    except Exception as e:

        logging.error(f"Ranking endpoint error: {e}")

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# --------------------------------
# Run Flask Server
# --------------------------------

if __name__ == "__main__":

    logging.info("Starting Flask development server")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
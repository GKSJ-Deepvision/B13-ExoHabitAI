# ==========================================================
# ExoHabitAI Backend API  —  app.py  (DEPLOYMENT READY)
# Flask Application for Exoplanet Habitability Prediction
# ==========================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import prepare_features, FEATURE_ORDER
import joblib
import pandas as pd
import os
import logging


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')   # serves index.html


# ── Flask init — serves frontend as static files ───────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)


# ── Model & data paths ─────────────────────────────────────────────────────
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


# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "api.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.info("ExoHabitAI API starting...")


# ── Real feature importances from trained best_model.pkl ──────────────────
FEATURE_IMPORTANCE = {
    "Stellar_Flux":        0.2511,
    "Equilibrium_Temp":    0.2468,
    "Planet_Radius":       0.1106,
    "Orbital_Period":      0.0959,
    "Semi_Major_Axis":     0.0827,
    "Planet_Mass":         0.0759,
    "Planet_Density":      0.0447,
    "Stellar_Luminosity":  0.0407,
    "Stellar_Temp":        0.0240,
    "Stellar_Metallicity": 0.0156,
}


# ── Load model ─────────────────────────────────────────────────────────────
model = None
try:
    model = joblib.load(MODEL_PATH)
    logging.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Model load failed: {e}")


# ══════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════

# ── Serve frontend (index.html) at root ───────────────────────────────────
@app.route('/')
def serve_frontend():
    """Serve the frontend index.html — works on localhost AND on Render."""
    return app.send_static_file('index.html')


# ── Health check ──────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "success",
        "message": "ExoHabitAI Backend API Running",
        "model_loaded": model is not None
    })


# ── Model info ────────────────────────────────────────────────────────────
@app.route('/model-info', methods=['GET'])
def model_info():
    features = (
        list(model.feature_names_in_)
        if model is not None and hasattr(model, "feature_names_in_")
        else FEATURE_ORDER
    )
    return jsonify({
        "model_name": "ExoHabitAI",
        "version": "1.0",
        "features_used": features,
        "note": "Stellar_Flux is derived server-side. Do not send it as input."
    })


# ── Predict ───────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"status": "error", "message": "Invalid or missing JSON body"}), 400

        logging.info(f"Prediction request: {data}")

        features    = prepare_features(data)
        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        label       = "Potentially Habitable" if int(prediction) == 1 else "Non-Habitable"

        logging.info(f"Prediction={prediction} | Probability={probability:.6f}")

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "label": label,
            "habitability_probability": round(float(probability), 6),
            "feature_importance": FEATURE_IMPORTANCE
        })

    except ValueError as e:
        logging.warning(f"Validation error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"status": "error", "message": "Internal prediction error"}), 500


# ── Rank ──────────────────────────────────────────────────────────────────
@app.route('/rank', methods=['GET'])
def rank():
    try:
        if not os.path.exists(RANK_DATA_PATH):
            return jsonify({"status": "error", "message": "Ranking dataset not found"}), 404

        df = pd.read_csv(RANK_DATA_PATH)

        try:
            n = int(request.args.get("n", 10))
            n = max(1, min(n, 100))
        except (ValueError, TypeError):
            n = 10

        top_planets = df.head(n)
        logging.info(f"Rankings requested | n={n}")

        return jsonify({
            "status": "success",
            "count": len(top_planets),
            "top_planets": top_planets.to_dict(orient="records")
        })

    except Exception as e:
        logging.error(f"Ranking error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Local development only — Render uses gunicorn instead
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)

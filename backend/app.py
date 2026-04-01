"""
ExoHabitAI — Flask Backend + Frontend Server
Production-ready for Render deployment
"""

import os
import logging
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Silence werkzeug request logs in production
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "final_model.pkl")
DATA_PATH    = os.path.join(ROOT_DIR, "data", "processed", "preprocessed.csv")
RANKED_PATH  = os.path.join(ROOT_DIR, "data", "processed", "habitability_ranked.csv")

# ── Flask app ──────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ── Load model ─────────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded: {type(model).__name__}")
except Exception as e:
    model = None
    print(f"[ERROR] Could not load model: {e}")

# ── Feature config ──────────────────────────────────────────────
REQUIRED_FIELDS = [
    "radius_earth", "mass_earth", "orbital_period", "semimajor_axis",
    "eq_temp_k", "density", "star_temp_k", "star_luminosity",
    "star_metallicity", "habitability_score", "stellar_compatibility",
    "orbital_stability", "star_spectype",
]

NUMERIC_FEATURES = [
    "radius_earth", "pl_radj", "mass_earth", "pl_massj",
    "orbital_period", "semimajor_axis", "eq_temp_k", "density",
    "star_temp_k", "star_luminosity", "star_metallicity",
    "habitability_score", "stellar_compatibility", "orbital_stability",
    "radius_earth_scaled", "mass_earth_scaled", "orbital_period_scaled",
    "semimajor_axis_scaled", "eq_temp_k_scaled", "density_scaled",
    "star_temp_k_scaled", "star_luminosity_scaled", "star_metallicity_scaled",
]

STAR_FLAGS     = ["star_A","star_B","star_F","star_G","star_K","star_M","star_Unknown"]
HABITABLE_LABEL = {0: "Non-Habitable", 1: "Potentially Habitable"}

TOP_FEATURES = [
    {"feature": "Habitability Score",      "importance": 49.3},
    {"feature": "Equilibrium Temperature", "importance": 21.5},
    {"feature": "Eq. Temp (Scaled)",       "importance": 19.7},
    {"feature": "Planet Radius (Scaled)",  "importance": 2.7},
    {"feature": "Jupiter Radius",          "importance": 2.5},
    {"feature": "Star Temperature",        "importance": 1.7},
    {"feature": "Planet Radius",           "importance": 1.6},
]

# ── Helpers ────────────────────────────────────────────────────
def build_input_df(data: dict) -> pd.DataFrame:
    row = {}
    for col in NUMERIC_FEATURES:
        row[col] = float(data[col]) if col in data else np.nan
    row["star_spectype"] = data.get("star_spectype", "Unknown")
    spec = row["star_spectype"].strip().upper()
    provided_any = any(col in data for col in STAR_FLAGS)
    for col in STAR_FLAGS:
        if provided_any:
            row[col] = bool(data.get(col, False))
        else:
            letter = col.replace("star_", "")
            row[col] = spec.startswith(letter) if letter != "Unknown" else spec not in list("ABFGKM")
    return pd.DataFrame([row])


def validate_input(data: dict):
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    for f in [x for x in REQUIRED_FIELDS if x != "star_spectype"]:
        try:
            float(data[f])
        except (ValueError, TypeError):
            raise ValueError(f"Field '{f}' must be a number.")


def get_habitability_class(score: float) -> dict:
    if score >= 80:   return {"class": "High",     "color": "#22e87a", "icon": "🌍"}
    elif score >= 50: return {"class": "Moderate", "color": "#f5c842", "icon": "🌏"}
    elif score >= 20: return {"class": "Low",      "color": "#ff9a3c", "icon": "🌑"}
    else:             return {"class": "Unlikely", "color": "#ff5555", "icon": "💀"}


# ── Serve Frontend ─────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    full = os.path.join(FRONTEND_DIR, path)
    if os.path.exists(full):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


# ── API Routes ─────────────────────────────────────────────────
@app.route("/api", methods=["GET"])
def api_index():
    return jsonify({
        "project": "ExoHabitAI", "version": "1.0.0", "status": "running",
        "endpoints": {
            "POST /api/predict":       "Single planet prediction",
            "POST /api/predict/batch": "Batch prediction",
            "GET  /api/rank":          "Top habitable planets",
            "GET  /api/features":      "Feature importance",
            "GET  /api/stats":         "Dataset statistics",
            "GET  /api/health":        "Health check",
        }
    })


@app.route("/api/health", methods=["GET"])
def health():
    steps = list(model.named_steps.keys()) if model and hasattr(model, "named_steps") else []
    return jsonify({"status": "ok", "model_loaded": model is not None, "pipeline_steps": steps})


@app.route("/api/features", methods=["GET"])
def features():
    return jsonify({"required_fields": REQUIRED_FIELDS, "feature_importance": TOP_FEATURES})


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded."}), 503
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON body"}), 400
    try:
        validate_input(data)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 422
    try:
        X          = build_input_df(data)
        prediction = int(model.predict(X)[0])
        proba      = model.predict_proba(X)[0].tolist()
        hab_score  = round(proba[1] * 100, 2)
        hab_class  = get_habitability_class(hab_score)
        return jsonify({
            "status":             "success",
            "planet_name":        data.get("planet_name", "Unknown Planet"),
            "prediction":         prediction,
            "label":              HABITABLE_LABEL[prediction],
            "habitability_score": hab_score,
            "habitability_class": hab_class["class"],
            "class_color":        hab_class["color"],
            "class_icon":         hab_class["icon"],
            "probabilities": {
                "non_habitable": round(proba[0] * 100, 2),
                "habitable":     hab_score,
            },
            "input_summary": {
                "radius_earth":   data.get("radius_earth"),
                "eq_temp_k":      data.get("eq_temp_k"),
                "star_temp_k":    data.get("star_temp_k"),
                "orbital_period": data.get("orbital_period"),
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    data = request.get_json(silent=True)
    if not data or "planets" not in data:
        return jsonify({"status": "error", "message": "Body must contain a 'planets' list"}), 400
    planets, results, errors = data["planets"], [], []
    for i, planet in enumerate(planets):
        try:
            validate_input(planet)
            X          = build_input_df(planet)
            prediction = int(model.predict(X)[0])
            proba      = model.predict_proba(X)[0].tolist()
            hab_score  = round(proba[1] * 100, 2)
            results.append({"rank": None, "planet_name": planet.get("planet_name", f"Planet_{i+1}"),
                            "prediction": prediction, "label": HABITABLE_LABEL[prediction],
                            "habitability_score": hab_score})
        except Exception as e:
            errors.append({"index": i, "planet_name": planet.get("planet_name", f"Planet_{i+1}"), "error": str(e)})
    results.sort(key=lambda x: x["habitability_score"], reverse=True)
    for rank, p in enumerate(results, start=1):
        p["rank"] = rank
    return jsonify({"status": "success", "total": len(planets), "successful": len(results),
                    "failed": len(errors), "ranked_results": results, "errors": errors})


@app.route("/api/rank", methods=["GET"])
def rank():
    limit          = int(request.args.get("limit", 20))
    habitable_only = request.args.get("habitable_only", "false").lower() == "true"
    try:
        if os.path.exists(RANKED_PATH):
            df = pd.read_csv(RANKED_PATH)
        elif os.path.exists(DATA_PATH) and model is not None:
            df    = pd.read_csv(DATA_PATH)
            X     = df.drop(columns=["planet_name","host_star_name","habitable_binary"], errors="ignore")
            proba = model.predict_proba(X)[:, 1]
            df["habitability_probability"] = proba
            df["habitability_rank"]        = df["habitability_probability"].rank(ascending=False).astype(int)
            df = df.sort_values("habitability_probability", ascending=False)
        else:
            return jsonify({"status": "error", "message": "No data file found"}), 404

        prob_col = next((c for c in df.columns if "prob" in c.lower()), None)
        if prob_col is None:
            return jsonify({"status": "error", "message": "Probability column not found"}), 500
        if habitable_only:
            pred_col = next((c for c in df.columns if "predict" in c.lower() or "binary" in c.lower()), None)
            if pred_col:
                df = df[df[pred_col] == 1]
        df = df.head(limit)

        planets = []
        for _, row in df.iterrows():
            score = round(float(row[prob_col]) * 100, 2)
            hc    = get_habitability_class(score)
            planets.append({
                "rank":               int(row.get("habitability_rank", row.get("_habitability_rank", 0))),
                "planet_name":        str(row.get("planet_name", "Unknown")),
                "host_star_name":     str(row.get("host_star_name", "Unknown")),
                "habitability_score": score,
                "habitability_class": hc["class"],
                "class_color":        hc["color"],
                "radius_earth":       round(float(row["radius_earth"]), 3) if pd.notna(row.get("radius_earth")) else None,
                "eq_temp_k":          round(float(row["eq_temp_k"]), 1)    if pd.notna(row.get("eq_temp_k"))    else None,
                "orbital_period":     round(float(row["orbital_period"]), 2) if pd.notna(row.get("orbital_period")) else None,
                "star_temp_k":        round(float(row["star_temp_k"]), 1)  if pd.notna(row.get("star_temp_k"))  else None,
            })
        return jsonify({"status": "success", "count": len(planets), "planets": planets})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        df = pd.read_csv(DATA_PATH)
        habitable_count = int(df["habitable_binary"].sum())
        return jsonify({
            "status":             "success",
            "total_planets":      len(df),
            "habitable_planets":  habitable_count,
            "non_habitable":      len(df) - habitable_count,
            "habitability_rate":  round(habitable_count / len(df) * 100, 2),
            "avg_radius_earth":   round(float(df["radius_earth"].mean()), 3),
            "avg_eq_temp_k":      round(float(df["eq_temp_k"].mean()), 1),
            "avg_orbital_period": round(float(df["orbital_period"].mean()), 2),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

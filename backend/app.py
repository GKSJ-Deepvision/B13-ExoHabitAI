"""
app.py — ExoHabitAI Flask REST API
====================================
Serves the ML model and all data endpoints.

Run:
    python app.py

Endpoints:
    POST /api/predict        → Single planet habitability prediction
    POST /api/predict/batch  → Batch prediction (up to 500 planets)
    GET  /api/rank           → Top N habitable planets from dataset
    GET  /api/sample         → Curated sample planets for UI auto-fill
    GET  /api/sample/random  → One random planet row
    GET  /api/features       → Model feature metadata (for debugging)
    GET  /api/health         → Health check
    GET  /                   → Serve frontend index.html
"""

import os
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from utils import (
    get_model, get_feature_cols, get_spec_cols,
    predict, validate_input, validate_batch,
    get_sample_planets, get_random_sample, get_top_planets,
)

# ── App Initialisation ─────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Startup: warm model into memory ───────────────────────────────────────────
print("\n" + "="*60)
print("  🚀 ExoHabitAI API Starting Up")
print("="*60)
try:
    model = get_model()
    features = get_feature_cols()
    print(f"  ✅ XGBoost model loaded  ({len(features)} features)")
except Exception as e:
    print(f"  ⚠️  Model load warning: {e}")
print("="*60 + "\n")


# ── Frontend Serving ───────────────────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve the frontend SPA from /frontend directory."""
    if path and os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


# ── Health Check ───────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    """
    GET /api/health
    Returns model status and feature count.
    """
    try:
        feature_count = len(get_feature_cols())
        return jsonify({
            "status":        "ok",
            "model":         "XGBoost",
            "feature_count": feature_count,
            "version":       "1.0.0",
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Single Prediction ──────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Body (JSON):
    {
        "pl_rade":    -0.673,
        "pl_bmasse":  -0.187,
        "pl_orbper":   0.660,
        "pl_orbsmax":  0.875,
        "pl_eqt":     -2.198,
        "pl_dens":    -0.120,
        "st_teff":     0.201,
        "st_lum":      0.058,
        "st_met":     -1.519,
        "st_spectype": "G0 V"
    }
    Response:
    {
        "habitable": 1,
        "probability": 97.42,
        "label": "Potentially Habitable",
        "confidence_band": "High",
        "status": "success"
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({
                "status":  "error",
                "message": "Request body must be valid JSON."
            }), 400

        ok, err = validate_input(data)
        if not ok:
            return jsonify({"status": "error", "message": err}), 422

        result = predict(data)
        return jsonify(result), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Batch Prediction ───────────────────────────────────────────────────────────
@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    """
    POST /api/predict/batch
    Body (JSON): [ { ...planet fields... }, ... ]  (max 500 rows)

    Response:
    {
        "results": [ { "row": 1, "habitable": 1, "probability": 97.4, ... }, ... ],
        "summary": { "total": 10, "habitable_count": 3, "avg_probability": 45.2 },
        "status": "success"
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status":"error","message":"Body must be JSON array."}), 400

        ok, err = validate_batch(data)
        if not ok:
            return jsonify({"status":"error","message":err}), 422

        results = []
        for i, row in enumerate(data, 1):
            try:
                res = predict(row)
                res["row"] = i
                results.append(res)
            except Exception as e:
                results.append({"row":i,"status":"error","message":str(e)})

        habitable_count = sum(1 for r in results if r.get("habitable") == 1)
        probs = [r["probability"] for r in results if "probability" in r]
        avg_prob = round(sum(probs)/len(probs), 2) if probs else 0

        return jsonify({
            "results": results,
            "summary": {
                "total":            len(results),
                "habitable_count":  habitable_count,
                "non_habitable":    len(results) - habitable_count,
                "avg_probability":  avg_prob,
            },
            "status": "success",
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500


# ── Rankings ───────────────────────────────────────────────────────────────────
@app.route("/api/rank", methods=["GET"])
def api_rank():
    """
    GET /api/rank?n=10
    Returns the top N most habitable planets from the dataset.

    Query params:
        n (int): number of results, default 10, max 50
    """
    try:
        n = min(int(request.args.get("n", 10)), 50)
        planets = get_top_planets(n)
        return jsonify({"planets": planets, "count": len(planets), "status":"success"}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500


# ── Sample Data ────────────────────────────────────────────────────────────────
@app.route("/api/sample", methods=["GET"])
def api_sample():
    """
    GET /api/sample?n=12
    Returns N curated sample planets for the frontend auto-fill feature.
    Each planet has a human-readable habitability label and probability.
    """
    try:
        n = min(int(request.args.get("n", 12)), 30)
        samples = get_sample_planets(n)
        return jsonify({"samples": samples, "count": len(samples), "status":"success"}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500


@app.route("/api/sample/random", methods=["GET"])
def api_sample_random():
    """
    GET /api/sample/random
    Returns a single random planet from the dataset.
    """
    try:
        sample = get_random_sample()
        return jsonify({"sample": sample, "status":"success"}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)}), 500


# ── Feature Metadata ───────────────────────────────────────────────────────────
@app.route("/api/features", methods=["GET"])
def api_features():
    """
    GET /api/features
    Returns model feature metadata for debugging / documentation.
    """
    try:
        cols      = get_feature_cols()
        spec_cols = get_spec_cols()
        return jsonify({
            "total_features":      len(cols),
            "numeric_features":    9,
            "engineered_features": 4,
            "ohe_spectype_cols":   len(spec_cols),
            "all_features":        cols,
            "spectype_options":    [c.replace("st_spectype_","") for c in spec_cols],
            "status":              "success",
        }), 200
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500


# ── Error Handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"status":"error","message":"Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"status":"error","message":"Method not allowed."}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"status":"error","message":"Internal server error."}), 500


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)

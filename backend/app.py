"""
Flask REST API for Exoplanet Habitability Prediction
=====================================================
Endpoints:
    POST /predict  — Predict habitability for given exoplanet parameters
    GET  /rank     — Get ranked list of exoplanets by habitability score
    GET  /features — Get the list of expected input features
    GET  /health   — Health check
"""

from flask import Flask, request, jsonify
from utils import (
    load_model,
    get_feature_names,
    validate_input,
    predict_habitability,
    get_ranked_exoplanets,
)

# ==========================================
# FLASK APP INITIALIZATION
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD MODEL AT STARTUP
# ==========================================

with app.app_context():
    try:
        load_model()
        print("[INFO] Model pre-loaded successfully.")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[WARNING] Run the notebook save cells first, then restart the server.")


# ==========================================
# ROUTES
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "API is running"}), 200


@app.route("/features", methods=["GET"])
def features():
    """Return the list of feature names the model expects."""
    try:
        feature_list = get_feature_names()
        return jsonify({
            "status": "success",
            "feature_count": len(feature_list),
            "features": feature_list,
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict habitability for a single exoplanet.

    Expects JSON body with all required feature values.
    Example:
        {
            "feature_1": 0.5,
            "feature_2": 1.2,
            ...
        }
    """
    try:
        data = request.get_json(force=True, silent=True)

        # Validate input
        is_valid, result = validate_input(data)
        if not is_valid:
            return jsonify({"status": "error", "message": result}), 400

        # Run prediction
        prediction = predict_habitability(result)

        return jsonify({
            "status": "success",
            "data": prediction,
        }), 200

    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500


@app.route("/rank", methods=["GET", "POST"])
def rank():
    """
    Return ranked list of exoplanets by habitability score.

    Optional query parameter or JSON body:
        top_n (int): Number of top exoplanets to return (default: all)
    """
    try:
        # Accept top_n from query params or JSON body
        top_n = None

        if request.method == "POST":
            data = request.get_json(force=True, silent=True) or {}
            top_n = data.get("top_n")
        else:
            top_n = request.args.get("top_n")

        if top_n is not None:
            try:
                top_n = int(top_n)
                if top_n < 1:
                    return jsonify({
                        "status": "error",
                        "message": "top_n must be a positive integer.",
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    "status": "error",
                    "message": "top_n must be a valid integer.",
                }), 400

        ranked = get_ranked_exoplanets(top_n=top_n)

        return jsonify({
            "status": "success",
            "count": len(ranked),
            "data": ranked,
        }), 200

    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": f"Ranking failed: {str(e)}"}), 500


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    print("=" * 50)
    print("  Exoplanet Habitability Prediction API")
    print("=" * 50)
    print("  Endpoints:")
    print("    GET  /health   — Health check")
    print("    GET  /features — List expected features")
    print("    POST /predict  — Predict habitability")
    print("    GET  /rank     — Ranked exoplanet list")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)

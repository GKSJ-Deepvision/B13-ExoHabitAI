from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from utils import prepare_features

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")


@app.route("/")
def home():
    return "🚀 Exoplanet Habitability API is running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No input data provided"
            }), 400

        # Convert input → dataframe
        features = prepare_features(data)

        # Prediction
        prediction = int(model.predict(features)[0])

        # Probability (if available)
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])
        else:
            probability = None

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "habitability_probability": probability
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/rank", methods=["POST"])
def rank():
    try:
        data = request.get_json()

        if not isinstance(data, list):
            return jsonify({
                "status": "error",
                "message": "Input must be a list of exoplanets"
            }), 400

        df = prepare_features(data)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(df)[:, 1]
        else:
            scores = model.predict(df)

        df["habitability_score"] = scores
        df["rank"] = df["habitability_score"].rank(ascending=False, method="dense")

        result = df.sort_values("rank").to_dict(orient="records")

        return jsonify({
            "status": "success",
            "ranked_planets": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
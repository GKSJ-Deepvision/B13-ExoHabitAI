from flask import Flask, request, jsonify
import joblib
import numpy as np
from utils import validate_input, preprocess_input
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model at startup
model = joblib.load('../models/model.pkl')


@app.route('/')
def home():
    return "Exohabit AI Backend Running 🚀"


# ------------------ PREDICT API ------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check empty input
    if not data:
        return jsonify({
            "status": "error",
            "message": "No input data provided"
        }), 400

    # Validate input
    valid, message = validate_input(data)
    if not valid:
        return jsonify({
            "status": "error",
            "message": message
        }), 400

    try:
        # Preprocess input
        features = preprocess_input(data)

        # Prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            "status": "success",
            "message": "Prediction successful",
            "prediction": int(prediction),
            "habitability_score": float(probability)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ------------------ RANK API ------------------
@app.route('/rank', methods=['GET', 'POST'])
def rank():
    data = request.get_json()

    # Check empty input
    if not data:
        return jsonify({
            "status": "error",
            "message": "No input data provided"
        }), 400

    # Must be list
    if not isinstance(data, list):
        return jsonify({
            "status": "error",
            "message": "Input should be a list of planets"
        }), 400

    try:
        results = []

        for planet in data:
            valid, message = validate_input(planet)
            if not valid:
                return jsonify({
                    "status": "error",
                    "message": message
                }), 400

            features = preprocess_input(planet)
            score = model.predict_proba(features)[0][1]

            planet["habitability_score"] = float(score)
            results.append(planet)

        # Sort by score (descending)
        ranked = sorted(results, key=lambda x: x["habitability_score"], reverse=True)

        return jsonify({
            "status": "success",
            "message": "Ranking successful",
            "ranked_planets": ranked
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
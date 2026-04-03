from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import os


app = Flask(__name__)
CORS(app)

# Load model
# model = joblib.load('model.pkl')
scaler = joblib.load("C:\\Users\\hp\\Desktop\\project_intern\\New folder\\B13-ExoHabitAI\\models\\scaler.pkl") 
model = joblib.load('C:\\Users\\hp\\Desktop\\project_intern\\New folder\\B13-ExoHabitAI\\models\\model.pkl')


# ------------------ HOME ------------------
@app.route('/')
def home():
    return "ExoHabitAI Backend Running 🚀"


# ------------------ PREDICT API ------------------

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate input
    required_fields = ['pl_bmasse', 'pl_dens', 'st_teff', 'st_met']

    for field in required_fields:
        if field not in data:
            return jsonify({
                "status": "error",
                "message": f"Missing field: {field}"
            }), 400

    try:
        # Convert to numpy array
        features = np.array([[
            float(data['pl_bmasse']),
            float(data['pl_dens']),
            float(data['st_teff']),
            float(data['st_met'])
        ]])
        features = scaler.transform(features)

        mean = [1.6, 2.2, 5.4, -1]
        std = [1, 1, 1, 1]
        features = [[(features[0][i] - mean[i]) / std[i] for i in range(4)]]

        # Prediction
        # prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # 🔥 custom threshold
        threshold = 0.55

        prediction = 1 if probability > threshold else 0
        
        # probability = model.predict_proba(features)[0][1]

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "habitability_score": float(probability),
            "confidence": float(probability)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ------------------ RANK API ------------------
@app.route('/rank', methods=['POST'])
def rank():
    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({
            "status": "error",
            "message": "Input should be a list of planets"
        }), 400

    try:
        results = []

        for planet in data:
            features = np.array([[ 
                float(planet['pl_bmasse']),
                float(planet['pl_dens']),
                float(planet['st_teff']),
                float(planet['st_met'])
            ]])

            score = model.predict_proba(features)[0][1]

            planet['habitability_score'] = float(score)
            results.append(planet)

        # Sort by score
        ranked = sorted(results, key=lambda x: x['habitability_score'], reverse=True)

        return jsonify({
            "status": "success",
            "ranked_planets": ranked
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ------------------ RUN ------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
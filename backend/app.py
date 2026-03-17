from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from utils import validate_input, calculate_habitability, get_confidence_label

app = Flask(__name__)
CORS(app)

# Model load 
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'exohabit_model.pkl'
)
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model successfully loaded!")
    MODEL_LOADED = True
except Exception:
    model = None
    MODEL_LOADED = False
    print("⚠️  Model not loaded — using rule-based prediction")


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to ExoHabitAI API!",
        "endpoints": {
            "predict": "POST /predict",
            "rank":    "POST /rank",
            "health":  "GET  /health"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":       "healthy",
        "model_loaded": MODEL_LOADED,
        "api_version":  "1.0.0"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        # Rule-based calculation
        score, prediction = calculate_habitability(data)
        confidence = get_confidence_label(score)

        return jsonify({
            "status":              "success",
            "planet_name":         data.get("planet_name", "Unknown"),
            "prediction":          prediction,
            "habitability_status": "Potentially Habitable"
                                   if prediction == 1 else "Not Habitable",
            "habitability_score":  score,
            "confidence":          confidence,
            "input_parameters":    data,
            "message":             "Analysis complete"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/rank', methods=['POST'])
def rank():
    try:
        data    = request.get_json()
        planets = data.get('planets', [])

        if not planets:
            return jsonify({"error": "No planets provided"}), 400

        results = []
        for planet in planets:
            is_valid, _ = validate_input(planet)
            if not is_valid:
                continue
            score, prediction = calculate_habitability(planet)
            results.append({
                "planet_name":         planet.get("planet_name", "Unknown"),
                "habitability_score":  score,
                "habitability_status": "Potentially Habitable"
                                       if prediction == 1 else "Not Habitable"
            })

        results = sorted(
            results, key=lambda x: x['habitability_score'], reverse=True
        )
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return jsonify({
            "status":         "success",
            "total_planets":  len(results),
            "ranked_planets": results
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
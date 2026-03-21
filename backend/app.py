"""
Flask REST API for Exoplanet Habitability Prediction.
predict : POST — predicts habitability from exoplanet parameters
rank    : GET/POST — returns ranked list of habitable exoplanets
"""

import os
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path for model access
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    validate_input,
    prepare_features,
    format_prediction_response,
    load_ranked_data,
    FEATURE_NAMES
)

# ──────────────────────────────────────────────
# App Configuration
# ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, 'models', 'xgboost.pkl')
RANKED_CSV_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'habitability_ranked.csv')

# ──────────────────────────────────────────────
# Load Model on Startup
# ──────────────────────────────────────────────
model = None

def load_model():
    """Load the trained XGBoost model from disk."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from: {MODEL_PATH}")
        print(f"   Model type: {type(model).__name__}")
    except FileNotFoundError:
        print(f"❌ Model file not found at: {MODEL_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

load_model()


# ──────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    """API home / health check endpoint."""
    return jsonify({
        'status': 'online',
        'api': 'Exoplanet Habitability Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': {
                'method': 'POST',
                'description': 'Predict habitability of an exoplanet',
                'input': 'JSON with 14 exoplanet parameters'
            },
            '/rank': {
                'method': 'GET or POST',
                'description': 'Get ranked list of habitable exoplanets',
                'input': 'Optional: {"top_n": 10}'
            }
        },
        'required_features': FEATURE_NAMES
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict habitability of an exoplanet.

    Expects JSON body with 14 parameters:
    - Planet_Radius, Planet_Mass, Orbital_Period, Semi_Major_Axis,
      Equilibrium_Temp, Planet_Density, Stellar_Temp, Stellar_Luminosity,
      Stellar_Metallicity, StarType_A, StarType_F, StarType_G, StarType_K, StarType_M

    Returns:
        JSON with prediction result, habitability score, and confidence.
    """
    
    if not request.is_json:
        return jsonify({
            'status': 'error',
            'message': 'Content-Type must be application/json. '
                       'Send your request with header: Content-Type: application/json'
        }), 400

    data = request.get_json(silent=True)


    is_valid, error_msg, cleaned_data = validate_input(data)
    if not is_valid:
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'required_features': FEATURE_NAMES
        }), 400

    try:

        features_df = prepare_features(cleaned_data)


        prediction = model.predict(features_df)[0]

    
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)[0]
            probability = probabilities[1]  
        else:
            probability = float(prediction)

        
        response = format_prediction_response(prediction, probability, cleaned_data)
        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/rank', methods=['GET', 'POST'])
def rank():
    """
    Get ranked list of exoplanets based on habitability score.

    Optional parameters (via POST JSON or GET query params):
    - top_n: Number of top results to return (default: all)

    Returns:
        JSON with ranked list of habitable exoplanets.
    """
    top_n = None

    if request.method == 'POST' and request.is_json:
        data = request.get_json(silent=True)
        if data and 'top_n' in data:
            try:
                top_n = int(data['top_n'])
                if top_n <= 0:
                    return jsonify({
                        'status': 'error',
                        'message': 'top_n must be a positive integer.'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'status': 'error',
                    'message': 'top_n must be a valid integer.'
                }), 400
    elif request.method == 'GET':
        top_n_param = request.args.get('top_n')
        if top_n_param:
            try:
                top_n = int(top_n_param)
                if top_n <= 0:
                    return jsonify({
                        'status': 'error',
                        'message': 'top_n must be a positive integer.'
                    }), 400
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': 'top_n must be a valid integer.'
                }), 400

    # Load ranked data
    ranked_data = load_ranked_data(RANKED_CSV_PATH, top_n=top_n)

    if ranked_data is None:
        return jsonify({
            'status': 'error',
            'message': f'Ranked data file not found. Expected at: {RANKED_CSV_PATH}'
        }), 404

    return jsonify({
        'status': 'success',
        'total_results': len(ranked_data),
        'top_n': top_n if top_n else 'all',
        'rankings': ranked_data,
        'message': f'Returned {len(ranked_data)} ranked exoplanets.'
    }), 200


# ──────────────────────────────────────────────
# Error Handlers
# ──────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found. Visit / for available endpoints.'
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        'status': 'error',
        'message': 'HTTP method not allowed for this endpoint.'
    }), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error. Please try again later.'
    }), 500


# ──────────────────────────────────────────────
# Run Server
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("\n🚀 Starting Exoplanet Habitability Prediction API...")
    print(f"📂 Model: {MODEL_PATH}")
    print(f"📊 Ranked Data: {RANKED_CSV_PATH}")
    print(f"🌐 Server: http://127.0.0.1:5000\n")
    app.run(debug=True, host='127.0.0.1', port=5000)

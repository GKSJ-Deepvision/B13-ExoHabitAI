"""
ExoHabitAI Backend API
Flask REST API for exoplanet habitability prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from utils import validate_prediction_input, format_prediction_response, rank_exoplanets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load models
try:
    # Binary classification model (legacy/optional)
    binary_model = joblib.load(MODELS_DIR / "best_model.pkl")
    logger.info("Binary model loaded successfully")
except Exception as e:
    logger.warning(f"Binary model not found: {e}")
    binary_model = None

try:
    # Multi-class classification model v2.1
    multiclass_model = joblib.load(MODELS_DIR / "multiclass_v2.1_pipeline.pkl")
    multiclass_features = joblib.load(MODELS_DIR / "multiclass_v2.1_features.pkl")
    multiclass_classes = joblib.load(MODELS_DIR / "multiclass_v2.1_class_names.pkl")
    multiclass_mapping = joblib.load(MODELS_DIR / "multiclass_v2.1_class_mapping.pkl")
    multiclass_reverse = joblib.load(MODELS_DIR / "multiclass_v2.1_reverse_mapping.pkl")
    logger.info("Multi-class model v2.1 loaded successfully")
except Exception as e:
    logger.warning(f"Multi-class model v2.1 not found: {e}")
    multiclass_model = None
    multiclass_features = None
    multiclass_classes = None
    multiclass_mapping = None
    multiclass_reverse = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'multiclass_model': multiclass_model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Habitability prediction endpoint (Multi-class)
    """
    try:
        if multiclass_model is None:
            return jsonify({'error': 'Multi-class model not loaded'}), 500
        
        # Get input data
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate input
        is_valid, error_msg = validate_prediction_input(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Prepare base features
        base_features = [
            "orbital_period", "semimajor_axis", "star_temp_k", 
            "star_luminosity", "star_metallicity",
            "log_surface_gravity", "bulk_density_gcc"
        ]
        
        feature_dict = {f: data[f] for f in base_features}
        
        # Add star class one-hot encoding
        star_class = data.get('star_class', 'G')
        star_classes = ['A', 'B', 'F', 'G', 'K', 'M', 'Unknown']
        
        for sc in star_classes:
            feature_dict[f'star_class_{sc}'] = (star_class == sc)
        
        # Prepare features DataFrame
        features = pd.DataFrame([feature_dict])
        
        # Make prediction
        prediction_mapped = multiclass_model.predict(features)[0]
        probabilities = multiclass_model.predict_proba(features)[0]
        
        # Remap prediction
        prediction = multiclass_reverse[prediction_mapped]
        
        # Prepare response probabilities
        probs_dict = {
            multiclass_classes[multiclass_reverse[i]]: float(probabilities[i]) 
            for i in range(len(probabilities))
        }
        
        response = format_prediction_response(
            multiclass_classes[prediction],
            max(probabilities),
            probs_dict
        )
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rank', methods=['POST'])
def rank():
    """
    Ranked list of exoplanets based on habitability score.
    """
    try:
        if multiclass_model is None:
            return jsonify({'error': 'Multi-class model not loaded'}), 500
            
        data = request.json
        planets_list = data.get('planets', [])
        
        if not planets_list:
            return jsonify({'error': 'No planets list provided'}), 400
            
        ranked_list = rank_exoplanets(
            planets_list, 
            multiclass_model, 
            multiclass_reverse, 
            multiclass_classes
        )
        
        return jsonify({
            "status": "success",
            "ranked_planets": ranked_list
        })
        
    except Exception as e:
        logger.error(f"Ranking error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/multiclass', methods=['POST'])
def predict_multiclass_legacy():
    return predict()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

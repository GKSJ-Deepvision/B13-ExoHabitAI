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
    # Binary classification model
    binary_model = joblib.load(MODELS_DIR / "best_model.pkl")
    logger.info("Binary model loaded successfully")
except Exception as e:
    logger.warning(f"Binary model not found: {e}")
    binary_model = None

try:
    # Multi-class classification model
    multiclass_model = joblib.load(MODELS_DIR / "multiclass_xgboost.pkl")
    multiclass_features = joblib.load(MODELS_DIR / "multiclass_features.pkl")
    multiclass_classes = joblib.load(MODELS_DIR / "multiclass_class_names.pkl")
    logger.info("Multi-class model loaded successfully")
except Exception as e:
    logger.warning(f"Multi-class model not found: {e}")
    multiclass_model = None
    multiclass_features = None
    multiclass_classes = None

# Feature definitions
REQUIRED_FEATURES = [
    "mass_earth",
    "semimajor_axis",
    "star_temp_k",
    "star_luminosity",
    "star_metallicity",
    "log_stellar_flux",
    "log_surface_gravity",
    "bulk_density_gcc"
]


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'binary_model': binary_model is not None,
        'multiclass_model': multiclass_model is not None
    })


@app.route('/api/predict/binary', methods=['POST'])
def predict_binary():
    """
    Binary classification prediction endpoint
    
    Expected input:
    {
        "mass_earth": float,
        "semimajor_axis": float,
        "star_temp_k": float,
        "star_luminosity": float,
        "star_metallicity": float,
        "log_stellar_flux": float,
        "log_surface_gravity": float,
        "bulk_density_gcc": float
    }
    """
    try:
        if binary_model is None:
            return jsonify({'error': 'Binary model not loaded'}), 500
        
        # Get input data
        data = request.json
        
        # Validate input
        missing_features = [f for f in REQUIRED_FEATURES if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {missing_features}'
            }), 400
        
        # Prepare features
        features = pd.DataFrame([{f: data[f] for f in REQUIRED_FEATURES}])
        
        # Make prediction
        prediction = binary_model.predict(features)[0]
        probability = binary_model.predict_proba(features)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'prediction_label': 'Habitable' if prediction == 1 else 'Not Habitable',
            'probability': {
                'not_habitable': float(probability[0]),
                'habitable': float(probability[1])
            },
            'confidence': float(max(probability))
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Binary prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/multiclass', methods=['POST'])
def predict_multiclass():
    """
    Multi-class classification prediction endpoint
    
    Expected input: Same as binary + star_class (optional)
    
    Returns:
    {
        "prediction": int (0-3),
        "prediction_label": str,
        "probabilities": dict,
        "confidence": float
    }
    """
    try:
        if multiclass_model is None:
            return jsonify({'error': 'Multi-class model not loaded'}), 500
        
        # Get input data
        data = request.json
        
        # Prepare base features
        base_features = [
            "mass_earth", "semimajor_axis", "star_temp_k", 
            "star_luminosity", "star_metallicity", "log_stellar_flux",
            "log_surface_gravity", "bulk_density_gcc"
        ]
        
        # Check for missing base features
        missing_base = [f for f in base_features if f not in data]
        if missing_base:
            return jsonify({
                'error': f'Missing required features: {missing_base}'
            }), 400
        
        # Create feature dictionary with base features
        feature_dict = {f: data[f] for f in base_features}
        
        # Add star class one-hot encoding
        # Default to G-class (Sun-like) if not specified
        star_class = data.get('star_class', 'G')
        star_classes = ['A', 'B', 'F', 'G', 'K', 'M', 'Unknown']
        
        for sc in star_classes:
            feature_dict[f'star_class_{sc}'] = (star_class == sc)
        
        # Prepare features DataFrame
        features = pd.DataFrame([feature_dict])
        
        # Make prediction
        prediction = multiclass_model.predict(features)[0]
        probabilities = multiclass_model.predict_proba(features)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'prediction_label': multiclass_classes[prediction],
            'probabilities': {
                multiclass_classes[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            'confidence': float(max(probabilities)),
            'all_classes': multiclass_classes
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Multi-class prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/both', methods=['POST'])
def predict_both():
    """
    Get both binary and multi-class predictions
    """
    try:
        data = request.json
        
        # Get binary prediction
        binary_result = None
        if binary_model is not None:
            features = pd.DataFrame([{f: data[f] for f in REQUIRED_FEATURES}])
            prediction = binary_model.predict(features)[0]
            probability = binary_model.predict_proba(features)[0]
            binary_result = {
                'prediction': int(prediction),
                'prediction_label': 'Habitable' if prediction == 1 else 'Not Habitable',
                'probability': {
                    'not_habitable': float(probability[0]),
                    'habitable': float(probability[1])
                },
                'confidence': float(max(probability))
            }
        
        # Get multi-class prediction
        multiclass_result = None
        if multiclass_model is not None:
            # Prepare base features
            base_features = [
                "mass_earth", "semimajor_axis", "star_temp_k", 
                "star_luminosity", "star_metallicity", "log_stellar_flux",
                "log_surface_gravity", "bulk_density_gcc"
            ]
            
            feature_dict = {f: data[f] for f in base_features}
            
            # Add star class one-hot encoding
            star_class = data.get('star_class', 'G')
            star_classes = ['A', 'B', 'F', 'G', 'K', 'M', 'Unknown']
            
            for sc in star_classes:
                feature_dict[f'star_class_{sc}'] = (star_class == sc)
            
            features = pd.DataFrame([feature_dict])
            prediction = multiclass_model.predict(features)[0]
            probabilities = multiclass_model.predict_proba(features)[0]
            multiclass_result = {
                'prediction': int(prediction),
                'prediction_label': multiclass_classes[prediction],
                'probabilities': {
                    multiclass_classes[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                },
                'confidence': float(max(probabilities))
            }
        
        return jsonify({
            'binary': binary_result,
            'multiclass': multiclass_result
        })
    
    except Exception as e:
        logger.error(f"Combined prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/features', methods=['GET'])
def get_features():
    """Get required features for prediction"""
    return jsonify({
        'binary_features': REQUIRED_FEATURES,
        'multiclass_features': multiclass_features if multiclass_features else [],
        'multiclass_classes': multiclass_classes if multiclass_classes else []
    })


@app.route('/api/batch/predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Expected input:
    {
        "model": "binary" or "multiclass",
        "data": [
            {...features...},
            {...features...}
        ]
    }
    """
    try:
        request_data = request.json
        model_type = request_data.get('model', 'binary')
        data_list = request_data.get('data', [])
        
        if not data_list:
            return jsonify({'error': 'No data provided'}), 400
        
        results = []
        
        if model_type == 'binary' and binary_model is not None:
            for data in data_list:
                features = pd.DataFrame([{f: data[f] for f in REQUIRED_FEATURES}])
                prediction = binary_model.predict(features)[0]
                probability = binary_model.predict_proba(features)[0]
                results.append({
                    'prediction': int(prediction),
                    'prediction_label': 'Habitable' if prediction == 1 else 'Not Habitable',
                    'confidence': float(max(probability))
                })
        
        elif model_type == 'multiclass' and multiclass_model is not None:
            for data in data_list:
                features = pd.DataFrame([{f: data[f] for f in multiclass_features}])
                prediction = multiclass_model.predict(features)[0]
                probabilities = multiclass_model.predict_proba(features)[0]
                results.append({
                    'prediction': int(prediction),
                    'prediction_label': multiclass_classes[prediction],
                    'confidence': float(max(probabilities))
                })
        
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

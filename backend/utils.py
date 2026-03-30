"""
ExoHabitAI Backend Utilities
Validation and formatting functions for API endpoints.
"""

import logging

logger = logging.getLogger(__name__)

def validate_prediction_input(data):
    """
    Validate incoming JSON data for prediction.
    """
    required_features = [
        "orbital_period",
        "semimajor_axis",
        "star_temp_k",
        "star_luminosity",
        "star_metallicity",
        "log_surface_gravity",
        "bulk_density_gcc"
    ]
    
    missing = [f for f in required_features if f not in data]
    if missing:
        return False, f"Missing required features: {missing}"
    
    # Simple type check
    for f in required_features:
        if not isinstance(data[f], (int, float)):
            try:
                data[f] = float(data[f])
            except (ValueError, TypeError):
                return False, f"Feature '{f}' must be a number."
                
    return True, None

def format_prediction_response(prediction_label, confidence, probabilities):
    """
    Standardize the structure of prediction responses.
    """
    return {
        "status": "success",
        "prediction_label": prediction_label,
        "confidence_score": float(confidence),
        "probabilities": probabilities
    }

def rank_exoplanets(planets_data, multiclass_model, multiclass_reverse, multiclass_classes):
    """
    Rank a list of planets based on their habitability score.
    Habitability score is based on the confidence of the 'Rocky-Habitable' class.
    """
    import pandas as pd
    
    star_classes = ['A', 'B', 'F', 'G', 'K', 'M', 'Unknown']
    base_features = [
        "orbital_period", "semimajor_axis", "star_temp_k", 
        "star_luminosity", "star_metallicity",
        "log_surface_gravity", "bulk_density_gcc"
    ]
    
    scored_planets = []
    
    for i, data in enumerate(planets_data):
        try:
            # Prepare feature dict
            feature_dict = {f: data[f] for f in base_features}
            star_class = data.get('star_class', 'G')
            for sc in star_classes:
                feature_dict[f'star_class_{sc}'] = (star_class == sc)
            
            features_df = pd.DataFrame([feature_dict])
            
            # Get probabilities
            probs = multiclass_model.predict_proba(features_df)[0]
            
            # Map probabilities to class names
            mapped_probs = {
                multiclass_classes[multiclass_reverse[j]]: float(probs[j]) 
                for j in range(len(probs))
            }
            
            # Rank score = probability of being 'Rocky-Habitable'
            habitability_score = mapped_probs.get('Rocky-Habitable', 0.0)
            
            scored_planets.append({
                "index": i,
                "name": data.get("name", f"Planet {i+1}"),
                "habitability_score": habitability_score,
                "prediction": multiclass_classes[multiclass_reverse[multiclass_model.predict(features_df)[0]]]
            })
        except Exception as e:
            logger.error(f"Error ranking planet at index {i}: {e}")
    
    # Sort by habitability_score descending
    ranked_list = sorted(scored_planets, key=lambda x: x['habitability_score'], reverse=True)
    return ranked_list

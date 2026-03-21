"""
Utility functions for exoplanet habitability prediction API.
"""

import pandas as pd
import numpy as np
import os

# Feature names in the exact order expected by the model
FEATURE_NAMES = [
    'Planet_Radius',
    'Planet_Mass',
    'Orbital_Period',
    'Semi_Major_Axis',
    'Equilibrium_Temp',
    'Planet_Density',
    'Stellar_Temp',
    'Stellar_Luminosity',
    'Stellar_Metallicity',
    'StarType_A',
    'StarType_F',
    'StarType_G',
    'StarType_K',
    'StarType_M'
]

# Numeric feature ranges
FEATURE_RANGES = {
    'Planet_Radius': (0.1, 30.0),
    'Planet_Mass': (0.01, 10000.0),
    'Orbital_Period': (0.1, 1000.0),
    'Semi_Major_Axis': (0.001, 10.0),
    'Equilibrium_Temp': (50.0, 5000.0),
    'Planet_Density': (0.01, 50.0),
    'Stellar_Temp': (2000.0, 10000.0),
    'Stellar_Luminosity': (0.001, 100.0),
    'Stellar_Metallicity': (-1.0, 1.0),
}

STAR_TYPES = ['StarType_A', 'StarType_F', 'StarType_G', 'StarType_K', 'StarType_M']


def validate_input(data):
    
    if not data:
        return False, "Request body is empty. Please provide exoplanet parameters in JSON format.", None

    # Check for missing features
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return False, f"Missing required parameters: {', '.join(missing)}", None

    cleaned = {}

    # Validate numeric features
    for feature in FEATURE_NAMES:
        if feature in STAR_TYPES:
            
            val = data[feature]
            if isinstance(val, bool):
                cleaned[feature] = val
            elif isinstance(val, str):
                cleaned[feature] = val.lower() in ('true', '1', 'yes')
            elif isinstance(val, (int, float)):
                cleaned[feature] = bool(val)
            else:
                return False, f"Invalid value for '{feature}': expected boolean (true/false).", None
        else:
        
            try:
                val = float(data[feature])
            except (ValueError, TypeError):
                return False, f"Invalid value for '{feature}': expected a numeric value, got '{data[feature]}'.", None

            
            if feature in FEATURE_RANGES:
                low, high = FEATURE_RANGES[feature]
                if val < low or val > high:
                    pass

            cleaned[feature] = val


    star_type_count = sum(1 for st in STAR_TYPES if cleaned.get(st, False))
    if star_type_count == 0:
        return False, "At least one StarType must be set to true (StarType_A, StarType_F, StarType_G, StarType_K, or StarType_M).", None
    if star_type_count > 1:
        return False, "Only one StarType should be set to true at a time.", None

    return True, None, cleaned


def prepare_features(cleaned_data):
    
    features = {}
    for name in FEATURE_NAMES:
        if name in STAR_TYPES:
            features[name] = [1 if cleaned_data[name] else 0]
        else:
            features[name] = [cleaned_data[name]]

    return pd.DataFrame(features, columns=FEATURE_NAMES)


def format_prediction_response(prediction, probability, input_data):
    
    is_habitable = int(prediction) == 1
    confidence = round(float(probability), 6)

    return {
        'status': 'success',
        'prediction': {
            'habitable': is_habitable,
            'class': int(prediction),
            'label': 'Potentially Habitable' if is_habitable else 'Not Habitable',
            'habitability_score': confidence,
            'confidence': f"{confidence * 100:.2f}%"
        },
        'message': (
            f"The exoplanet is predicted to be {'potentially habitable' if is_habitable else 'not habitable'} "
            f"with a confidence score of {confidence * 100:.2f}%."
        )
    }


def load_ranked_data(csv_path, top_n=None):
    
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    
    habitable_df = df[df['Predicted_Class'] == 1].copy()

    if top_n is not None and top_n > 0:
        habitable_df = habitable_df.head(top_n)

    records = []
    for _, row in habitable_df.iterrows():
        records.append({
            'rank': int(row['Rank']),
            'habitability_score': round(float(row['Habitability_Score']), 8),
            'predicted_class': int(row['Predicted_Class']),
            'planet_radius': round(float(row['Planet_Radius']), 4),
            'planet_mass': round(float(row['Planet_Mass']), 4),
            'orbital_period': round(float(row['Orbital_Period']), 4),
            'semi_major_axis': round(float(row['Semi_Major_Axis']), 6),
            'equilibrium_temp': round(float(row['Equilibrium_Temp']), 1),
            'planet_density': round(float(row['Planet_Density']), 4),
            'stellar_temp': round(float(row['Stellar_Temp']), 1),
            'stellar_luminosity': round(float(row['Stellar_Luminosity']), 6),
            'stellar_metallicity': round(float(row['Stellar_Metallicity']), 4),
        })

    return records

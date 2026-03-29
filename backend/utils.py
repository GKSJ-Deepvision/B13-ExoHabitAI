# Import Libraries

from pathlib import Path
import joblib
import logging

# Step 1: Required Features

# These are all inputs needed by the ML model

FEATURES = [
    'pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbsmax',
    'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met',
    'spec_F', 'spec_G', 'spec_K', 'spec_M', 'spec_m',
    'st_lum', 'temp_score', 'dist_score', 'lum_score',
    'stellar_temp_score', 'stellar_mass_score',
    'stellar_compatibility_index', 'orbital_stability'
]

# Step 2: Load Trained Model

def load_model():
    try:
        # Path(__file__) matlab 'utils.py'
        base_dir = Path(__file__).resolve().parent
       
        model_path = base_dir / "models" / "random_forest.pkl"
        
        logging.info(f"Trying to load model from: {model_path}")
        
        model = joblib.load(model_path)
        logging.info("Model loaded successfully")
        return model

    except Exception as e:
        logging.error(f"Model error: {e}")
        return None
# Step 3: Validate Input Data

def validate_input(data):

    # Check if JSON data is received
    if data is None:
        return False, "No JSON data"

    # Check if all required features are present
    for feature in FEATURES:
        if feature not in data:
            return False, f"Missing: {feature}"

    # Step 4: Define Feature Types
    # Numeric features (must be numbers)
    numeric_features = [
        'pl_rade','pl_bmasse','pl_orbper','pl_orbsmax','pl_eqt',
        'st_teff','st_rad','st_mass','st_met','st_lum',
        'temp_score','dist_score','lum_score','stellar_temp_score',
        'stellar_mass_score','stellar_compatibility_index','orbital_stability'
    ]

    # Binary features (must be 0 or 1)
    binary_features = ['spec_F','spec_G','spec_K','spec_M','spec_m']

    # Step 5: Validate Numeric Values
    for f in numeric_features:
        try:
            # Check null values
            if data[f] is None:
                return False, f"{f} is null"

            # Convert to float (check valid number)
            float(data[f])

        except:
            return False, f"Invalid value: {f}"

    # Step 6: Validate Binary Values
    for f in binary_features:
        # Allow only 0 or 1 values
        if data[f] not in [0, 1, '0', '1', True, False]:
            return False, f"{f} must be 0 or 1"

    # All checks passed 
    return True, "Valid input"

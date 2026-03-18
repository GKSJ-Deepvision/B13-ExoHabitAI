import numpy as np

# Correct features (from your model)
REQUIRED_FIELDS = [
    "pl_rade",
    "st_lum",
    "pl_orbper",
    "st_teff",
    "pl_eqt",
    "pl_orbsmax"
]


# ------------------ VALIDATION ------------------
def validate_input(data):
    if not data:
        return False, "No input data provided"

    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing field: {field}"

        if not isinstance(data[field], (int, float)):
            return False, f"Invalid type for {field}. Must be int or float"

    return True, "Valid input"


# ------------------ PREPROCESSING ------------------
def preprocess_input(data):
    try:
        features = [
            data["pl_rade"],
            data["st_lum"],
            data["pl_orbper"],
            data["st_teff"],
            data["pl_eqt"],
            data["pl_orbsmax"]
        ]

        return np.array(features).reshape(1, -1)

    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")
"""
Utility module for loading the trained ML model and performing predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd

# ==========================================
# MODEL LOADING
# ==========================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost.pkl")
RANKED_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "habitability_ranked.csv")

_model = None


def load_model():
    """Load the trained XGBoost model from disk."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please run the training notebook first to generate the model."
            )
        _model = joblib.load(MODEL_PATH)
        print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
    return _model


def get_feature_names():
    """Return the list of feature names the model expects."""
    model = load_model()
    # XGBoost / sklearn models store feature names after .fit() on a DataFrame
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        return model.get_booster().feature_names
    else:
        raise RuntimeError("Cannot determine feature names from the model.")


# ==========================================
# INPUT VALIDATION
# ==========================================

def validate_input(data):
    """
    Validate incoming JSON data against the expected feature names.

    Returns:
        (True, df)  on success — df is a single-row DataFrame ready for prediction
        (False, error_message) on failure
    """
    if data is None:
        return False, "Request body is empty. Please send JSON data."

    expected_features = get_feature_names()

    # Check for missing features
    missing = [f for f in expected_features if f not in data]
    if missing:
        return False, {
            "error": "Missing required parameters",
            "missing_features": missing,
            "expected_features": expected_features,
        }

    # Check for non-numeric values
    row = {}
    for feature in expected_features:
        val = data[feature]
        try:
            row[feature] = float(val)
        except (ValueError, TypeError):
            return False, {
                "error": f"Invalid value for '{feature}': '{val}'. Must be numeric.",
            }

    df = pd.DataFrame([row], columns=expected_features)
    return True, df


# ==========================================
# PREDICTION
# ==========================================

def predict_habitability(input_df):
    """
    Run the model on a validated input DataFrame.

    Returns a dict with prediction and probability score.
    """
    model = load_model()
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    confidence = float(probabilities[prediction])
    habitability_score = float(probabilities[1])  # probability of class 1 (habitable)

    label = "Habitable" if prediction == 1 else "Not Habitable"

    return {
        "prediction": prediction,
        "label": label,
        "habitability_score": round(habitability_score, 4),
        "confidence": round(confidence, 4),
    }


# ==========================================
# RANKING
# ==========================================

def get_ranked_exoplanets(top_n=None):
    """
    Load the pre-computed ranked exoplanet list from CSV.

    Args:
        top_n: If provided, return only the top N exoplanets.

    Returns:
        List of dicts with exoplanet data sorted by habitability probability.
    """
    if not os.path.exists(RANKED_CSV_PATH):
        raise FileNotFoundError(
            f"Ranked CSV not found at {RANKED_CSV_PATH}. "
            "Please run the training notebook first to generate rankings."
        )

    df = pd.read_csv(RANKED_CSV_PATH)

    # Ensure it's sorted by habitability_probability descending
    if "habitability_probability" in df.columns:
        df = df.sort_values(by="habitability_probability", ascending=False)

    # Add rank column
    df.insert(0, "rank", range(1, len(df) + 1))

    if top_n is not None:
        df = df.head(top_n)

    return df.to_dict(orient="records")

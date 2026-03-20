from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.pkl"
RANKED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "habitability_ranked.csv"


def get_expected_features(model):
    """
    Use the exact feature names the trained model expects.
    If the model was trained with column names, feature_names_in_ will exist.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # Fallback: edit this list if your model expects different columns
    return [
        "radius_earth",
        "mass_earth",
        "orbital_period",
        "semimajor_axis",
        "eq_temp_k",
        "density",
        "star_temp_k",
        "star_luminosity",
        "star_metallicity",
        "star_spectype",
    ]


def validate_predict_payload(payload, expected_features):
    """
    Validate JSON input and convert it into a single-row DataFrame.
    """
    if not isinstance(payload, dict):
        return False, "Request body must be a JSON object.", None

    missing = [f for f in expected_features if f not in payload]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}", None

    row = {f: payload[f] for f in expected_features}
    return True, "OK", pd.DataFrame([row])


def load_ranked_csv():
    """
    Load the habitability ranking CSV if it exists.
    """
    if RANKED_CSV_PATH.exists():
        return pd.read_csv(RANKED_CSV_PATH)
    return pd.DataFrame()
"""
ExoHabitAI — Flask Backend API
utils.py: Input validation, feature engineering, and data loading helpers.

The scaler is now fitted on exactly 16 features (no pl_rade, no habitability).
build_feature_vector assembles the same 16 features and applies the scaler
before prediction — matching the training pipeline exactly.
"""

import os
import numpy as np
import pandas as pd
import joblib

# -----------------------------------
# Required input fields and their valid ranges
# -----------------------------------
REQUIRED_FIELDS = {
    "pl_orbper" : {"type": float, "min": 0.01,  "max": 1e9,   "desc": "Orbital period (days)"},
    "pl_orbsmax": {"type": float, "min": 0.001, "max": 100.0, "desc": "Semi-major axis (AU)"},
    "st_teff"   : {"type": float, "min": 300.0, "max": 60000, "desc": "Star temperature (K)"},
    "st_met"    : {"type": float, "min": -5.0,  "max": 10.0,  "desc": "Star metallicity (dex)"},
    "st_spectype": {"type": str,  "allowed": ["G","K","M","F","B","D","L","T","W","m","O","A"],
                   "desc": "Star spectral type"},
}

KNOWN_SPECTYPES = ["B", "D", "F", "G", "K", "L", "M", "T", "W", "m"]

# -----------------------------------
# Load scaler and feature columns at module import
# -----------------------------------
_DIR          = os.path.dirname(__file__)
_SCALER_PATH  = os.path.join(_DIR, "..", "models", "scaler.pkl")
_FCOLS_PATH   = os.path.join(_DIR, "..", "models", "feature_cols.pkl")

try:
    _scaler       = joblib.load(_SCALER_PATH)
    _feature_cols = joblib.load(_FCOLS_PATH)
    print(f"[OK] Scaler loaded from {_SCALER_PATH}")
    print(f"[OK] Feature cols loaded: {len(_feature_cols)} features")
    print(f"[OK] Columns: {_feature_cols}")
except FileNotFoundError as e:
    _scaler       = None
    _feature_cols = None
    print(f"[WARN] {e}")
    print("[WARN] Run Exploratory-Data-Analysis.py first.")


# ============================================================
# validate_input
# ============================================================
def validate_input(data: dict) -> list:
    errors = []
    for field, rules in REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}' ({rules['desc']})")
            continue
        value = data[field]
        if rules["type"] == float:
            try:
                value = float(value)
            except (TypeError, ValueError):
                errors.append(f"'{field}' must be a number. Got: {value}")
                continue
            if "min" in rules and value < rules["min"]:
                errors.append(f"'{field}' must be >= {rules['min']}. Got: {value}")
            if "max" in rules and value > rules["max"]:
                errors.append(f"'{field}' must be <= {rules['max']}. Got: {value}")
        elif rules["type"] == str:
            if not isinstance(value, str):
                errors.append(f"'{field}' must be a string. Got: {type(value).__name__}")
                continue
            spectype = value.strip()[0].upper() if value.strip() else ""
            if "allowed" in rules:
                allowed = [s.upper() for s in rules["allowed"]]
                if spectype not in allowed:
                    errors.append(f"'{field}' must be one of {rules['allowed']}. Got: '{value}'")
    return errors


# ============================================================
# build_feature_vector
# ============================================================
def build_feature_vector(data: dict, feature_cols: list) -> pd.DataFrame:
    """
    Build a scaled feature vector from raw user input.

    Steps:
    1. Extract raw values from JSON
    2. Compute engineered features (same formulas as EDA)
    3. One-hot encode spectral type
    4. Align to saved feature_cols (16 features, no pl_rade)
    5. Apply saved StandardScaler
    6. Return scaled DataFrame ready for model.predict()
    """
    pl_orbper   = float(data["pl_orbper"])
    pl_orbsmax  = float(data["pl_orbsmax"])
    st_teff     = float(data["st_teff"])
    st_met      = float(data["st_met"])
    st_spectype = str(data["st_spectype"]).strip()[0].upper()

    # st_lum dropped as zero-variance → use proxy 0.0
    st_lum_proxy = 0.0

    # Engineered features — same formulas as EDA
    stellar_compatibility = (
        (1 / (1 + abs(st_teff - 5778))) +
        (1 / (1 + abs(st_lum_proxy)))
    )
    orbital_stability = (
        (1 / (1 + abs(pl_orbper))) +
        (1 / (1 + abs(pl_orbsmax)))
    )

    # One-hot encode spectral type
    spectype_cols = {
        "st_spectype_B": 0, "st_spectype_D": 0,
        "st_spectype_F": 0, "st_spectype_G": 0,
        "st_spectype_K": 0, "st_spectype_L": 0,
        "st_spectype_M": 0, "st_spectype_T": 0,
        "st_spectype_W": 0, "st_spectype_m": 0,
    }
    col_key = f"st_spectype_{st_spectype}"
    if col_key in spectype_cols:
        spectype_cols[col_key] = 1

    # Assemble row — no pl_rade
    row = {
        "pl_orbper"            : pl_orbper,
        "pl_orbsmax"           : pl_orbsmax,
        "st_teff"              : st_teff,
        "st_met"               : st_met,
        "stellar_compatibility": stellar_compatibility,
        "orbital_stability"    : orbital_stability,
        **spectype_cols
    }

    # Use saved feature cols (16 features, no pl_rade)
    cols = _feature_cols if _feature_cols is not None else feature_cols
    df   = pd.DataFrame([row])
    df   = df.reindex(columns=cols, fill_value=0)

    print(f"[DEBUG] Feature vector shape: {df.shape}")
    print(f"[DEBUG] Columns: {list(df.columns)}")

    # Apply StandardScaler
    if _scaler is not None:
        scaled_values = _scaler.transform(df)
        df = pd.DataFrame(scaled_values, columns=df.columns)
    else:
        print("[WARN] Scaler not available — using raw values.")

    return df


# ============================================================
# load_ranked_data
# ============================================================
def load_ranked_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ranked data not found: {path}")
    df = pd.read_csv(path)
    df = df.sort_values("habitability_probability", ascending=False).reset_index(drop=True)
    return df
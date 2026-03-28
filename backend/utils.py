"""
utils.py — ExoHabitAI Backend Utilities
=========================================
Handles all preprocessing, feature alignment, and helper functions.

Key insight: The model was trained on 304 features (9 numeric + 4 engineered
+ 291 OHE spectral type columns). This module reconstructs that exact feature
vector from simple user inputs automatically.
"""

import os
import json
import random
import pickle
import joblib
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "xgboost.pkl")
CSV_PATH     = os.path.join(BASE_DIR, "data", "habitability_ranked.csv")
SAMPLES_PATH = os.path.join(BASE_DIR, "data", "sample_planets.json")
FEATURES_PATH= os.path.join(BASE_DIR, "data", "feature_cols.json")

# ── Singleton model loader ─────────────────────────────────────────────────────
_model        = None
_feature_cols = None
_spec_cols    = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
    return _model


def get_feature_cols():
    """Return the ordered list of 304 feature columns the model expects."""
    global _feature_cols
    if _feature_cols is None:
        if os.path.exists(FEATURES_PATH):
            with open(FEATURES_PATH) as f:
                _feature_cols = json.load(f)
        else:
            # Derive from model booster
            model = get_model()
            _feature_cols = model.get_booster().feature_names
    return _feature_cols


def get_spec_cols():
    """Return only the OHE spectral type column names."""
    global _spec_cols
    if _spec_cols is None:
        _spec_cols = [c for c in get_feature_cols() if c.startswith("st_spectype_")]
    return _spec_cols


# ── Core preprocessing ─────────────────────────────────────────────────────────

def build_feature_vector(
    pl_rade:    float,
    pl_bmasse:  float,
    pl_orbper:  float,
    pl_orbsmax: float,
    pl_eqt:     float,
    pl_dens:    float,
    st_teff:    float,
    st_lum:     float,
    st_met:     float,
    st_spectype: str = "G",
    # Advanced optional fields (already scaled values from sample data)
    stellar_temp_score: float = None,
    stellar_lum_score: float = None,
    stellar_compatibility_index: float = None,
    orbital_stability_factor: float = None,
) -> pd.DataFrame:
    """
    Transform raw inputs → 1-row DataFrame aligned to the model's 304-feature schema.

    The data stored in habitability_ranked.csv is already StandardScaler-normalised
    (from preprocess.ipynb cell 27). When a user provides sample data from the CSV,
    the values are passed through directly. When users type custom values, the values
    are treated as already-normalized (z-score) values since we don't have the
    original scaler saved.

    NOTE for production: Save the StandardScaler during training:
        joblib.dump(scaler, 'backend/models/scaler.pkl')
    Then load and apply it here for raw (un-normalised) user inputs.
    """

    # ── Engineered features (mirrors preprocess.ipynb cells 22-24) ────────────
    # These are computed BEFORE scaling in the original pipeline, then scaled.
    # Since all input values here are already scaled, we compute scaled versions
    # of the derived features using the scaled inputs.

    if stellar_temp_score is None:
        # Approximation: scaled teff ≈ (teff - 5500) / 800
        # stellar_temp_score = 1 - |teff - 5778| / 5778  (pre-scaling formula)
        # For scaled input we use a linear approximation
        stellar_temp_score = -abs(st_teff) * 0.15

    if stellar_lum_score is None:
        stellar_lum_score = -abs(st_lum) * 0.3

    if stellar_compatibility_index is None:
        stellar_compatibility_index = (
            0.6 * stellar_temp_score + 0.4 * stellar_lum_score
        )

    if orbital_stability_factor is None:
        # orbital_stability_factor = orbsmax / orbper (pre-scaling)
        # Approximation with scaled values
        orbital_stability_factor = (pl_orbsmax - pl_orbper) * 0.1

    # ── Build base dict ────────────────────────────────────────────────────────
    row = {
        "pl_rade":                       float(pl_rade),
        "pl_bmasse":                     float(pl_bmasse),
        "pl_orbper":                     float(pl_orbper),
        "pl_orbsmax":                    float(pl_orbsmax),
        "pl_eqt":                        float(pl_eqt),
        "pl_dens":                       float(pl_dens),
        "st_teff":                       float(st_teff),
        "st_lum":                        float(st_lum),
        "st_met":                        float(st_met),
        "stellar_temp_score":            float(stellar_temp_score),
        "stellar_lum_score":             float(stellar_lum_score),
        "stellar_compatibility_index":   float(stellar_compatibility_index),
        "orbital_stability_factor":      float(orbital_stability_factor),
    }

    # ── OHE spectral type (drop_first=True was used in training) ─────────────
    spec_cols = get_spec_cols()
    ohe_dict  = {col: 0 for col in spec_cols}

    spec_key = f"st_spectype_{st_spectype.strip()}"
    if spec_key in ohe_dict:
        ohe_dict[spec_key] = 1
    else:
        # Try prefix match: "G2" matches "G2 V", "G2 IV", etc.
        prefix = f"st_spectype_{st_spectype.strip()}"
        for col in spec_cols:
            if col.startswith(prefix):
                ohe_dict[col] = 1
                break
        # Unknown spectype → all zeros (treated as baseline/reference category)

    row.update(ohe_dict)

    # ── Align to exact model feature order ────────────────────────────────────
    feature_cols = get_feature_cols()
    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    return df


def predict(input_dict: dict) -> dict:
    """
    Full prediction pipeline.

    Parameters
    ----------
    input_dict : dict with keys matching the 9 numeric fields + st_spectype.
                 Optionally includes pre-computed engineered features.

    Returns
    -------
    dict: { habitable, probability, label, confidence_band, status }
    """
    # Extract values with defaults
    def _f(k, default=0.0):
        v = input_dict.get(k, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    df = build_feature_vector(
        pl_rade    = _f("pl_rade",    0.0),
        pl_bmasse  = _f("pl_bmasse",  0.0),
        pl_orbper  = _f("pl_orbper",  0.0),
        pl_orbsmax = _f("pl_orbsmax", 0.0),
        pl_eqt     = _f("pl_eqt",     0.0),
        pl_dens    = _f("pl_dens",    0.0),
        st_teff    = _f("st_teff",    0.0),
        st_lum     = _f("st_lum",     0.0),
        st_met     = _f("st_met",     0.0),
        st_spectype= str(input_dict.get("st_spectype", "G")).strip(),
        # Pass through pre-computed engineered features if present (from sample data)
        stellar_temp_score         = _f("stellar_temp_score", None) if "stellar_temp_score" in input_dict else None,
        stellar_lum_score          = _f("stellar_lum_score", None) if "stellar_lum_score" in input_dict else None,
        stellar_compatibility_index= _f("stellar_compatibility_index", None) if "stellar_compatibility_index" in input_dict else None,
        orbital_stability_factor   = _f("orbital_stability_factor", None) if "orbital_stability_factor" in input_dict else None,
    )

    model       = get_model()
    prediction  = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1]) * 100

    # Confidence band
    if probability >= 75:
        confidence = "High"
    elif probability >= 45:
        confidence = "Moderate"
    else:
        confidence = "Low"

    label = "Potentially Habitable" if prediction == 1 else "Not Habitable"

    return {
        "habitable":       prediction,
        "probability":     round(probability, 2),
        "label":           label,
        "confidence_band": confidence,
        "status":          "success",
    }


# ── Sample data helpers ────────────────────────────────────────────────────────

def get_sample_planets(n: int = 12) -> list:
    """Return curated sample planets for the frontend sample picker."""
    if os.path.exists(SAMPLES_PATH):
        with open(SAMPLES_PATH) as f:
            samples = json.load(f)
        return samples[:n]

    # Fallback: pull random rows from CSV
    return _sample_from_csv(n)


def get_random_sample() -> dict:
    """Return a single random planet row from the dataset."""
    samples = get_sample_planets(12)
    if samples:
        return random.choice(samples)
    return _sample_from_csv(1)[0]


def _sample_from_csv(n: int) -> list:
    """Internal: sample n rows from the ranked CSV."""
    if not os.path.exists(CSV_PATH):
        return []

    df       = pd.read_csv(CSV_PATH)
    spec_cols = [c for c in df.columns if c.startswith("st_spectype_")]
    leakage   = {"habitable","habitable_multi","habitability_binary","habitability_probability",
                 "habitability_index","radius_score","temp_score","distance_score"}
    base_cols = ["pl_rade","pl_bmasse","pl_orbper","pl_orbsmax","pl_eqt","pl_dens",
                 "st_teff","st_lum","st_met"]

    results = []
    for _, row in df.sample(min(n, len(df)), random_state=42).iterrows():
        active = [c.replace("st_spectype_", "") for c in spec_cols if row.get(c, False)]
        spec   = active[0] if active else "G"
        rec    = {k: round(float(row[k]), 4) for k in base_cols}
        rec["st_spectype"]            = spec
        rec["habitability_probability"] = round(float(row.get("habitability_probability", 0)) * 100, 1)
        rec["habitability_binary"]    = int(row.get("habitability_binary", 0))
        results.append(rec)
    return results


# ── Ranking helper ─────────────────────────────────────────────────────────────

def get_top_planets(n: int = 10) -> list:
    """Return the top N most habitable planets from the ranked CSV."""
    if not os.path.exists(CSV_PATH):
        return []

    df        = pd.read_csv(CSV_PATH)
    spec_cols = [c for c in df.columns if c.startswith("st_spectype_")]
    top       = df.nlargest(n, "habitability_probability")

    results = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        active = [c.replace("st_spectype_", "") for c in spec_cols if row.get(c, False)]
        spec   = active[0] if active else "Unknown"
        results.append({
            "rank":                    i,
            "planet_id":               f"EXO-{i:04d}",
            "st_spectype":             spec,
            "pl_rade":                 round(float(row["pl_rade"]), 3),
            "pl_bmasse":               round(float(row["pl_bmasse"]), 3),
            "pl_orbper":               round(float(row["pl_orbper"]), 3),
            "pl_orbsmax":              round(float(row["pl_orbsmax"]), 3),
            "pl_eqt":                  round(float(row["pl_eqt"]), 3),
            "st_teff":                 round(float(row["st_teff"]), 3),
            "habitability_probability":round(float(row["habitability_probability"]) * 100, 2),
            "habitable":               int(row["habitability_binary"]),
        })
    return results


# ── Validation ────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = ["pl_rade","pl_bmasse","pl_orbper","pl_orbsmax","pl_eqt",
                   "pl_dens","st_teff","st_lum","st_met","st_spectype"]

def validate_input(data: dict) -> tuple[bool, str]:
    """
    Validate incoming prediction request.
    Returns (is_valid: bool, error_message: str).
    """
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: '{field}'"

    for field in REQUIRED_FIELDS[:-1]:  # All except st_spectype
        try:
            float(data[field])
        except (TypeError, ValueError):
            return False, f"Field '{field}' must be a valid number."

    if not isinstance(data.get("st_spectype"), str) or not data["st_spectype"].strip():
        return False, "Field 'st_spectype' must be a non-empty string (e.g. 'G2 V')."

    return True, ""


def validate_batch(rows: list) -> tuple[bool, str]:
    """Validate a batch of prediction rows."""
    if not isinstance(rows, list) or len(rows) == 0:
        return False, "Batch must be a non-empty list."
    if len(rows) > 500:
        return False, "Batch size cannot exceed 500 rows."
    for i, row in enumerate(rows):
        ok, msg = validate_input(row)
        if not ok:
            return False, f"Row {i+1}: {msg}"
    return True, ""

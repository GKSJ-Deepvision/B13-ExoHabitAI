"""
ExoHabitAI — Flask Backend API  (v2.1)
App.py: Main Flask application

Endpoints:
  POST /train-model   — Accept CSV/JSON dataset, train model, save artifacts
  POST /predict       — Single-planet habitability prediction
  POST /predict-batch — Batch prediction from CSV/JSON
  GET  /health        — Health check

Field alias support
-------------------
The frontend uses NASA Exoplanet Archive column names.
This backend accepts BOTH sets of names and maps them internally:

  Frontend name   →  Canonical name
  pl_orbper       →  orbital_period
  pl_orbsmax      →  semi_major_axis
  pl_rade         →  planet_radius
  pl_bmasse       →  planet_mass
  pl_eqt          →  surface_temp
  st_teff         →  star_temperature
  st_met          →  star_metallicity
  st_lum          →  luminosity
  st_rad          →  star_radius
  st_spectype     →  spectral_type
"""

import io
import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# ──────────────────────────────────────────────────────────────────────────────
# App initialisation
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:3001", "http://127.0.0.1:3001",
    "http://localhost:5173", "http://127.0.0.1:5173",
])

# ──────────────────────────────────────────────────────────────────────────────
# Constants & field aliases
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH   = os.path.join(BASE_DIR, "trained_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

# Maps OLD frontend/NASA names → canonical internal names
FIELD_ALIASES = {
    # planetary
    "pl_orbper"  : "orbital_period",
    "pl_orbsmax" : "semi_major_axis",
    "pl_rade"    : "planet_radius",
    "pl_bmasse"  : "planet_mass",
    "pl_eqt"     : "surface_temp",
    # stellar
    "st_teff"    : "star_temperature",
    "st_met"     : "star_metallicity",
    "st_lum"     : "luminosity",
    "st_rad"     : "star_radius",
    "st_spectype": "spectral_type",
    # label alias
    "habitable"  : "habitability_label",
    "label"      : "habitability_label",
}

# Canonical feature order used during training — never change this order
NUMERIC_FEATURES = [
    "planet_radius",
    "planet_mass",
    "orbital_period",
    "semi_major_axis",
    "surface_temp",
    "star_temperature",
    "star_metallicity",
    "luminosity",
    "star_radius",
]
CATEGORICAL_FEATURE  = "spectral_type"
TARGET_COL           = "habitability_label"

ALL_REQUIRED_COLUMNS = NUMERIC_FEATURES + [CATEGORICAL_FEATURE, TARGET_COL]
ALL_FEATURE_COLUMNS  = NUMERIC_FEATURES + [CATEGORICAL_FEATURE]

# Default values used when optional fields are missing in frontend calls
FEATURE_DEFAULTS = {
    "planet_radius"   : 1.0,
    "planet_mass"     : 1.0,
    "orbital_period"  : 365.0,
    "semi_major_axis" : 1.0,
    "surface_temp"    : 288.0,
    "star_temperature": 5778.0,
    "star_metallicity": 0.0,
    "luminosity"      : 1.0,
    "star_radius"     : 1.0,
    "spectral_type"   : "G",
}

THRESHOLD = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Load persisted artifacts at startup
# ──────────────────────────────────────────────────────────────────────────────
def _load_artifacts():
    m = s = e = None
    try:
        m = joblib.load(MODEL_PATH)
        s = joblib.load(SCALER_PATH)
        e = joblib.load(ENCODER_PATH)
        print("[OK] All ML artifacts loaded successfully.")
    except FileNotFoundError:
        print("[INFO] Artifacts not found — call POST /train-model first.")
    return m, s, e

model, scaler, encoder = _load_artifacts()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: apply field aliases to a dict or DataFrame
# ──────────────────────────────────────────────────────────────────────────────
def _apply_aliases_dict(d: dict) -> dict:
    """Rename keys in a dict using FIELD_ALIASES (old name → canonical name)."""
    out = {}
    for k, v in d.items():
        canonical = FIELD_ALIASES.get(k, k)   # use alias if known, else keep as-is
        out[canonical] = v
    return out


def _apply_aliases_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns in a DataFrame using FIELD_ALIASES."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    df = df.rename(columns=FIELD_ALIASES)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Helper: fill missing features with defaults
# ──────────────────────────────────────────────────────────────────────────────
def _fill_defaults(d: dict) -> dict:
    """Fill any missing canonical feature fields with sensible defaults."""
    out = dict(d)
    for feat, default in FEATURE_DEFAULTS.items():
        if feat not in out or out[feat] is None or str(out[feat]).strip() == "":
            out[feat] = default
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Helper: preprocess a canonical feature DataFrame → scaled numpy array
# ──────────────────────────────────────────────────────────────────────────────
def _preprocess(df: pd.DataFrame, sc: StandardScaler, le: LabelEncoder) -> np.ndarray:
    df = df.copy()
    known = list(le.classes_)
    df[CATEGORICAL_FEATURE] = (
        df[CATEGORICAL_FEATURE]
        .astype(str).str.strip()
        .apply(lambda x: x if x in known else known[0])
    )
    df["spectral_type_encoded"] = le.transform(df[CATEGORICAL_FEATURE])
    feature_order = NUMERIC_FEATURES + ["spectral_type_encoded"]
    X = df[feature_order].astype(float).values
    return sc.transform(X)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: parse request body → normalised DataFrame
# ──────────────────────────────────────────────────────────────────────────────
def _parse_body(req) -> pd.DataFrame:
    ct = req.content_type or ""
    if "application/json" in ct:
        data = req.get_json(force=True, silent=True)
        if data is None:
            raise ValueError("Invalid JSON body.")
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
    elif "multipart/form-data" in ct:
        if "file" not in req.files:
            raise ValueError("Expected a file under the key 'file'.")
        df = pd.read_csv(req.files["file"])
    elif "text/csv" in ct:
        df = pd.read_csv(io.StringIO(req.get_data(as_text=True)))
    else:
        # Try JSON first, then CSV
        try:
            data = req.get_json(force=True, silent=False)
            if isinstance(data, dict):
                data = [data]
            df = pd.DataFrame(data)
        except Exception:
            try:
                df = pd.read_csv(io.StringIO(req.get_data(as_text=True)))
            except Exception:
                raise ValueError(
                    "Cannot parse request body. "
                    "Send JSON (application/json) or CSV (text/csv / multipart)."
                )
    return _apply_aliases_df(df)


# ──────────────────────────────────────────────────────────────────────────────
# ROUTE: Health check
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"      : "ok",
        "api"         : "ExoHabitAI Backend",
        "version"     : "2.1",
        "model_loaded": model is not None,
        "endpoints"   : {
            "POST /train-model"  : "Upload CSV/JSON dataset to train the model",
            "POST /predict"      : "Predict habitability for a single planet",
            "POST /predict-batch": "Batch predict from CSV or JSON array",
            "GET  /health"       : "Health check",
        }
    }), 200


# ──────────────────────────────────────────────────────────────────────────────
# ROUTE: POST /train-model
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/train-model", methods=["POST"])
def train_model():
    global model, scaler, encoder

    try:
        df = _parse_body(request)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    # Validate required columns (after alias mapping)
    missing = [c for c in ALL_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return jsonify({
            "status" : "error",
            "message": f"Missing required columns: {missing}",
            "tip"    : "Accepted aliases: pl_orbper→orbital_period, st_teff→star_temperature, etc.",
        }), 400

    # Coerce types & drop NaN
    df[NUMERIC_FEATURES]     = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    df[CATEGORICAL_FEATURE]  = df[CATEGORICAL_FEATURE].astype(str).str.strip()
    n_before = len(df)
    df = df.dropna(subset=ALL_REQUIRED_COLUMNS)
    n_dropped = n_before - len(df)

    if len(df) < 10:
        return jsonify({
            "status" : "error",
            "message": f"Only {len(df)} valid rows after cleaning — need at least 10."
        }), 400

    y = df[TARGET_COL].astype(int)
    if not set(y.unique()).issubset({0, 1}):
        return jsonify({
            "status" : "error",
            "message": f"habitability_label must be 0 or 1. Found: {sorted(y.unique().tolist())}"
        }), 400

    # Encode + scale
    le = LabelEncoder()
    df["spectral_type_encoded"] = le.fit_transform(df[CATEGORICAL_FEATURE])
    feature_order = NUMERIC_FEATURES + ["spectral_type_encoded"]
    X = df[feature_order].astype(float)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))

    # Save
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(sc,  SCALER_PATH)
    joblib.dump(le,  ENCODER_PATH)

    model, scaler, encoder = clf, sc, le

    return jsonify({
        "status"          : "success",
        "message"         : "Model trained and saved successfully.",
        "samples_used"    : len(df),
        "samples_dropped" : n_dropped,
        "train_size"      : len(X_train),
        "test_size"       : len(X_test),
        "accuracy"        : round(float(acc), 4),
        "accuracy_pct"    : f"{acc * 100:.2f}%",
        "spectral_classes": list(le.classes_),
        "artifacts_saved" : {
            "model"  : MODEL_PATH,
            "scaler" : SCALER_PATH,
            "encoder": ENCODER_PATH,
        }
    }), 200


# ──────────────────────────────────────────────────────────────────────────────
# ROUTE: POST /predict
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "status" : "error",
            "message": "Model not loaded. Run POST /train-model first."
        }), 503

    # Parse JSON
    raw = request.get_json(force=True, silent=True)
    if raw is None:
        return jsonify({"status": "error", "message": "Invalid JSON body."}), 400

    # Normalise keys: lowercase + apply aliases
    data = {k.strip().lower().replace(" ", "_").replace("-", "_"): v for k, v in raw.items()}
    data = _apply_aliases_dict(data)

    # Fill optional/missing fields with defaults so frontend doesn't have to send all 10
    data = _fill_defaults(data)

    # Validate numeric fields
    errors = []
    for feat in NUMERIC_FEATURES:
        try:
            float(data[feat])
        except (TypeError, ValueError):
            errors.append(f"'{feat}' must be a number. Got: {data[feat]}")
    if errors:
        return jsonify({"status": "error", "message": "Validation errors.", "errors": errors}), 400

    # Validate spectral type
    known_classes = list(encoder.classes_)
    spectral = str(data.get(CATEGORICAL_FEATURE, "")).strip()
    if spectral not in known_classes:
        # Graceful fallback instead of hard error — frontend may send limited types
        spectral     = known_classes[0]
        data[CATEGORICAL_FEATURE] = spectral

    # Preprocess & predict
    try:
        row_df   = pd.DataFrame([data])
        X_scaled = _preprocess(row_df, scaler, encoder)
        proba    = float(model.predict_proba(X_scaled)[0][1])
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500

    is_hab     = proba >= THRESHOLD
    label      = "Potentially Habitable" if is_hab else "Non-Habitable"
    confidence = f"{proba * 100:.2f}%"

    return jsonify({
        # ── New canonical field names (for new consumers) ──
        "habitability_score" : round(proba, 6),
        "status"             : "Habitable" if is_hab else "Not Habitable",

        # ── Legacy field names (for existing ResultPanel.jsx) ──
        "prediction"              : 1 if is_hab else 0,
        "habitability_probability": round(proba, 6),
        "label"                   : label,
        "confidence"              : confidence,

        "threshold_used": THRESHOLD,
    }), 200


# ──────────────────────────────────────────────────────────────────────────────
# ROUTE: POST /predict-batch
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({
            "status" : "error",
            "message": "Model not loaded. Run POST /train-model first."
        }), 503

    try:
        df = _parse_body(request)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    # Validate feature columns
    missing = [f for f in ALL_FEATURE_COLUMNS if f not in df.columns]
    if missing:
        return jsonify({
            "status" : "error",
            "message": f"Missing required feature columns: {missing}",
        }), 400

    # Clean
    df[NUMERIC_FEATURES]    = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    df[CATEGORICAL_FEATURE] = df[CATEGORICAL_FEATURE].astype(str).str.strip()

    # Fill defaults for missing numeric values row-by-row
    for feat in NUMERIC_FEATURES:
        df[feat] = df[feat].fillna(FEATURE_DEFAULTS[feat])
    df[CATEGORICAL_FEATURE] = df[CATEGORICAL_FEATURE].replace("", FEATURE_DEFAULTS["spectral_type"])
    df[CATEGORICAL_FEATURE] = df[CATEGORICAL_FEATURE].replace("nan", FEATURE_DEFAULTS["spectral_type"])

    n_rows = len(df)
    if n_rows == 0:
        return jsonify({"status": "error", "message": "No valid rows found."}), 400

    # Preprocess & predict
    try:
        X_scaled = _preprocess(df, scaler, encoder)
        probas   = model.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500

    results = []
    for i, prob in enumerate(probas):
        is_hab = float(prob) >= THRESHOLD
        results.append({
            "row_index"               : i,
            "habitability_score"      : round(float(prob), 6),
            "habitability_probability": round(float(prob), 6),
            "prediction"              : 1 if is_hab else 0,
            "status"                  : "Habitable" if is_hab else "Not Habitable",
            "label"                   : "Potentially Habitable" if is_hab else "Non-Habitable",
            "confidence"              : f"{prob * 100:.2f}%",
        })

    return jsonify({
        "status"        : "success",
        "total_rows"    : len(results),
        "threshold_used": THRESHOLD,
        "predictions"   : results,
    }), 200


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ExoHabitAI Flask API  v2.1")
    print("=" * 60)
    print("  POST /train-model   — Train and save model from dataset")
    print("  POST /predict       — Single planet prediction")
    print("  POST /predict-batch — Batch prediction")
    print("  GET  /health        — Health check")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
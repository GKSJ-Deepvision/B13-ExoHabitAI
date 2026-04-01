"""
ExoHabitAI — Standalone Training Script
train.py: Trains a Random Forest classifier using the new feature set and
          saves trained_model.pkl, scaler.pkl, encoder.pkl to the backend folder.

Feature set (strict):
  planet_radius, planet_mass, orbital_period, semi_major_axis,
  surface_temp, star_temperature, star_metallicity, luminosity,
  star_radius, spectral_type

Target:
  habitability_label  (1 = Habitable, 0 = Not Habitable)

Usage:
    cd backend
    python train.py                         # uses built-in synthetic data
    python train.py --csv path/to/data.csv  # uses a real CSV dataset
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH   = os.path.join(BASE_DIR, "trained_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

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
CATEGORICAL_FEATURE = "spectral_type"
TARGET_COL          = "habitability_label"

SPECTRAL_TYPES = ["B", "F", "G", "K", "M", "A", "O", "L", "T"]

RANDOM_SEED = 42
N_SAMPLES   = 6000


# ──────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Dataset Generator
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    """
    Generate a realistic synthetic exoplanet dataset with:
      - planet_radius    : Earth radii      (0.3 – 15)
      - planet_mass      : Earth masses     (0.1 – 300)
      - orbital_period   : days             (1 – 50 000)
      - semi_major_axis  : AU               (0.01 – 50)
      - surface_temp     : Kelvin           (100 – 800)
      - star_temperature : Kelvin           (2 500 – 40 000)
      - star_metallicity : dex              (-3 – 1.5)
      - luminosity       : solar luminosity (0.001 – 1 000)
      - star_radius      : solar radii      (0.1 – 20)
      - spectral_type    : categorical

    Habitability heuristic (at least 3 of 5 conditions):
      1. star_temperature in [4 000, 7 000] K   (F/G/K-ish)
      2. orbital_period   in [50,    700]  days
      3. semi_major_axis  in [0.5,   2.0]  AU
      4. star_metallicity in [-1.0,  0.5]  dex
      5. spectral_type in {G, K, F}
    """
    np.random.seed(RANDOM_SEED)

    planet_radius   = np.random.lognormal(mean=0.5, sigma=0.8, size=n).clip(0.3, 15.0)
    planet_mass     = np.random.lognormal(mean=1.0, sigma=1.5, size=n).clip(0.1, 300.0)
    orbital_period  = np.random.lognormal(mean=5.0, sigma=1.2, size=n).clip(1.0, 50000.0)
    semi_major_axis = np.random.lognormal(mean=0.0, sigma=1.0, size=n).clip(0.01, 50.0)
    surface_temp    = np.random.normal(loc=300, scale=150, size=n).clip(100.0, 800.0)
    star_temperature= np.random.normal(loc=5500, scale=2000, size=n).clip(2500.0, 40000.0)
    star_metallicity= np.random.normal(loc=0.0, scale=0.3, size=n).clip(-3.0, 1.5)
    luminosity      = np.random.lognormal(mean=0.0, sigma=1.5, size=n).clip(0.001, 1000.0)
    star_radius     = np.random.lognormal(mean=0.0, sigma=0.5, size=n).clip(0.1, 20.0)

    spec_weights = [0.03, 0.08, 0.25, 0.20, 0.35, 0.04, 0.01, 0.02, 0.02]
    spectral_type = np.random.choice(SPECTRAL_TYPES, size=n, p=spec_weights)

    # ── 7-condition habitability heuristic ──────────────────────────
    # Condition 1: Star temperature in habitable zone (F/G/K range)
    cond1 = (star_temperature >= 4000) & (star_temperature <= 7000)

    # Condition 2: Orbital period in Earth-like range (not too short/long)
    cond2 = (orbital_period >= 50) & (orbital_period <= 700)

    # Condition 3: Semi-major axis in habitable zone
    cond3 = (semi_major_axis >= 0.5) & (semi_major_axis <= 2.0)

    # Condition 4: Star metallicity not extreme
    cond4 = (star_metallicity >= -1.0) & (star_metallicity <= 0.5)

    # Condition 5: Spectral type is F, G, or K (most habitable)
    cond5 = np.isin(spectral_type, ["G", "K", "F"])

    # Condition 6: Planet is rocky/super-Earth, NOT a gas giant
    #   Earth = 1 M⊕, Neptune ≈ 17 M⊕, Jupiter ≈ 318 M⊕
    #   Habitable mass window: 0.1 – 10 Earth masses
    cond6 = (planet_mass >= 0.1) & (planet_mass <= 10.0)

    # Condition 7: Planet radius in rocky/super-Earth range (< 2.5 R⊕)
    #   Above ~1.6 R⊕ radius cliff → likely has H/He envelope (mini-Neptune+)
    #   We allow up to 2.5 to catch some super-Earths
    cond7 = (planet_radius >= 0.5) & (planet_radius <= 2.5)

    score = (cond1.astype(int) + cond2.astype(int) + cond3.astype(int) +
             cond4.astype(int) + cond5.astype(int) +
             cond6.astype(int) + cond7.astype(int))

    # Need 5 of 7 conditions for habitability (more strict than before)
    # 4% noise
    noise = np.random.random(n) < 0.04
    habitability_label = ((score >= 5) ^ noise).astype(int)

    df = pd.DataFrame({
        "planet_radius"   : planet_radius,
        "planet_mass"     : planet_mass,
        "orbital_period"  : orbital_period,
        "semi_major_axis" : semi_major_axis,
        "surface_temp"    : surface_temp,
        "star_temperature": star_temperature,
        "star_metallicity": star_metallicity,
        "luminosity"      : luminosity,
        "star_radius"     : star_radius,
        "spectral_type"   : spectral_type,
        "habitability_label": habitability_label,
    })
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_and_validate_csv(path: str) -> pd.DataFrame:
    """Load a user-provided CSV; normalize column names; validate required cols."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    required = NUMERIC_FEATURES + [CATEGORICAL_FEATURE, TARGET_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] CSV is missing required columns: {missing}")
        sys.exit(1)

    return df


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
      1. Drop rows with any NaN in required columns
      2. Encode spectral_type with LabelEncoder
      3. Build X (numeric + encoded cat) in strict order
      4. Scale with StandardScaler
    Returns X_scaled, y, fitted_scaler, fitted_encoder
    """
    required = NUMERIC_FEATURES + [CATEGORICAL_FEATURE, TARGET_COL]
    before   = len(df)
    df       = df.dropna(subset=required).copy()
    dropped  = before - len(df)
    if dropped:
        print(f"  [INFO] Dropped {dropped} rows with missing values.")

    df[NUMERIC_FEATURES]  = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    df[CATEGORICAL_FEATURE] = df[CATEGORICAL_FEATURE].astype(str).str.strip()
    df = df.dropna(subset=NUMERIC_FEATURES)

    # Encode spectral_type
    le = LabelEncoder()
    df["spectral_type_encoded"] = le.fit_transform(df[CATEGORICAL_FEATURE])
    print(f"  spectral_type classes : {list(le.classes_)}")

    # Build feature matrix in strict order
    feature_order = NUMERIC_FEATURES + ["spectral_type_encoded"]
    X = df[feature_order].astype(float)
    y = df[TARGET_COL].astype(int)

    if not set(y.unique()).issubset({0, 1}):
        print(f"[ERROR] habitability_label contains non-binary values: {sorted(y.unique().tolist())}")
        sys.exit(1)

    # Scale
    sc      = StandardScaler()
    X_scaled = sc.fit_transform(X)

    return X_scaled, y, sc, le


# ──────────────────────────────────────────────────────────────────────────────
# 3. Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ExoHabitAI Model Trainer")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a real CSV dataset. If not provided, uses synthetic data.")
    args = parser.parse_args()

    print("=" * 62)
    print("  ExoHabitAI — Model Training Script  (v2.0)")
    print("=" * 62)

    # ── Step 1: Load data ──
    if args.csv:
        print(f"\n[1/5] Loading dataset from {args.csv} ...")
        df = load_and_validate_csv(args.csv)
        print(f"      Loaded {len(df)} rows from CSV.")
    else:
        print(f"\n[1/5] Generating {N_SAMPLES} synthetic exoplanet records ...")
        df = generate_synthetic_dataset(N_SAMPLES)
        print(f"      Generated {len(df)} rows.")

    n_hab   = int(df[TARGET_COL].sum())
    n_total = len(df)
    print(f"      Habitable   : {n_hab} ({n_hab/n_total*100:.1f}%)")
    print(f"      Non-habitable: {n_total-n_hab} ({(n_total-n_hab)/n_total*100:.1f}%)")

    # ── Step 2: Preprocess ──
    print("\n[2/5] Preprocessing (encode + scale) ...")
    X_scaled, y, sc, le = preprocess(df)
    print(f"      Feature matrix shape: {X_scaled.shape}")

    # ── Step 3: Train / Test Split (80 / 20) ──
    print("\n[3/5] Splitting dataset (80/20 stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"      Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # ── Step 4: Train RandomForestClassifier ──
    print("\n[4/5] Training RandomForestClassifier ...")
    clf = RandomForestClassifier(
        n_estimators     = 200,
        max_depth        = 12,
        min_samples_leaf = 4,
        class_weight     = "balanced",
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
    )
    clf.fit(X_train, y_train)
    print("      Training complete.")

    # ── Step 5: Evaluate ──
    print("\n[5/5] Evaluating on test set ...")
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")

    print(f"\n  Accuracy  : {acc * 100:.2f}%")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Habitable", "Habitable"]))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # ── Save Artifacts ──
    print(f"\n  Saving artifacts ...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(sc,  SCALER_PATH)
    joblib.dump(le,  ENCODER_PATH)
    print(f"  [OK] trained_model.pkl -> {MODEL_PATH}")
    print(f"  [OK] scaler.pkl        -> {SCALER_PATH}")
    print(f"  [OK] encoder.pkl       -> {ENCODER_PATH}")

    print("\n" + "=" * 62)
    print("  Training complete. Run 'python app.py' to start the API.")
    print("=" * 62)


if __name__ == "__main__":
    main()

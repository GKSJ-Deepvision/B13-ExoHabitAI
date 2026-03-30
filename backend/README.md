# 🛰️ ExoHabitAI — Backend (Flask REST API)

> Python + Flask backend for the ExoHabitAI project. Serves ML predictions and ranked exoplanet data via REST API.

---

## ⚙️ Tech Stack

| Layer      | Tool                          |
|------------|-------------------------------|
| Framework  | Flask 2.3.2                   |
| ML Model   | Scikit-learn — Random Forest  |
| Data       | Pandas, NumPy                 |
| Serialization | Joblib                     |
| CORS       | Flask-CORS                    |

---

--

## 🧠 ML Model Details

- **Model:** Tuned Random Forest Classifier
- **File:** `models/random_forest.pkl` (loaded via `joblib`)
- **Prediction Threshold:** `0.9` — probabilities below 0.9 are classified as non-habitable
- **Input Features:** 22 total (see below)

### 📋 Required Features (22)

| Feature | Description |
|---|---|
| `pl_rade` | Planet radius (Earth radii) |
| `pl_bmasse` | Planet mass (Earth masses) |
| `pl_orbper` | Orbital period (days) |
| `pl_orbsmax` | Semi-major axis (AU) |
| `pl_eqt` | Equilibrium temperature (K) |
| `st_teff` | Stellar effective temperature (K) |
| `st_rad` | Stellar radius (Solar radii) |
| `st_mass` | Stellar mass (Solar masses) |
| `st_met` | Stellar metallicity |
| `st_lum` | Stellar luminosity |
| `spec_F/G/K/M/m` | Spectral type (binary: 0 or 1) |
| `temp_score` | Temperature habitability score |
| `dist_score` | Distance habitability score |
| `lum_score` | Luminosity habitability score |
| `stellar_temp_score` | Stellar temperature score |
| `stellar_mass_score` | Stellar mass score |
| `stellar_compatibility_index` | Overall stellar compatibility |
| `orbital_stability` | Orbital stability score |

---

## ✅ Input Validation Rules

- All 22 features are **required**
- Numeric features must be valid **float** values, non-null
- Spectral type features (`spec_F`, `spec_G`, `spec_K`, `spec_M`, `spec_m`) must be **0 or 1**

---

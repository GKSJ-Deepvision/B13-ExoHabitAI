# 🪐 ExoHabitAI — Exoplanet Habitability Prediction System

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple)
![Render](https://img.shields.io/badge/Deployed-Render.com-brightgreen)

> An AI-powered web application that predicts the habitability of exoplanets
> using a trained XGBoost classifier on 33,799 planets from the
> NASA Exoplanet Archive.

## 🌍 Live Demo

**https://habitability-prediction-ml-11.onrender.com**

---

## 📌 Project Overview

ExoHabitAI analyses planetary and stellar parameters to determine
whether an exoplanet could potentially support life. The system
uses a binary classification model — Habitable (1) vs Not Habitable (0).

| Metric | Value |
|--------|-------|
| Total Planets Analysed | 33,799 |
| Habitable Planets | 9,328 (27.6%) |
| Not Habitable | 24,471 (72.4%) |
| Model Features | 304 |
| Best Model | XGBoost Classifier |
| Spectral Type OHE Columns | 291 |

---

## ✨ Features

- 🔭 **Real-time Prediction** — Enter planet parameters and get
  instant habitability prediction with probability percentage
- 🌍 **Planet Naming** — Name your planet and see it displayed
  in the result card
- 🎲 **Sample Data** — Auto-fill form with real planets from
  the training dataset
- 📊 **Batch Upload** — Upload CSV or JSON files to predict
  up to 500 planets at once and download results
- 🏆 **Rankings Table** — View the top 10 most habitable
  planets from the dataset
- 🚀 **Space UI** — Animated starfield, glassmorphism cards,
  SVG probability arc, orbital loading animation

---

## 🗂️ Project Structure
```
HABITABILITY-PREDICTION-ML-/
├── backend/
│   ├── app.py                    # Flask REST API
│   ├── utils.py                  # Preprocessing pipeline
│   ├── requirements.txt          # Python dependencies
│   ├── models/
│   │   └── xgboost.zip           # Compressed trained model
│   └── data/
│       ├── habitability_ranked.zip   # Full ranked dataset
│       ├── sample_planets.json       # Curated UI samples
│       └── feature_cols.json         # 304 model feature names
├── frontend/
│   ├── index.html                # 3-page space UI
│   ├── style.css                 # Glassmorphism theme
│   └── script.js                 # API integration
└── requirements.txt              # Root level (ignored by Render)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check and model status |
| POST | `/api/predict` | Single planet prediction |
| POST | `/api/predict/batch` | Batch prediction (max 500) |
| GET | `/api/rank?n=10` | Top N habitable planets |
| GET | `/api/sample` | Sample planets for auto-fill |
| GET | `/api/sample/random` | One random planet |

### Example — POST /api/predict

**Request:**
```json
{
  "pl_rade":    -0.673,
  "pl_bmasse":  -0.187,
  "pl_orbper":   0.660,
  "pl_orbsmax":  0.875,
  "pl_eqt":     -2.198,
  "pl_dens":    -0.120,
  "st_teff":     0.201,
  "st_lum":      0.058,
  "st_met":     -1.519,
  "st_spectype": "G0 V"
}
```

**Response:**
```json
{
  "habitable": 1,
  "probability": 97.42,
  "label": "Potentially Habitable",
  "confidence_band": "High",
  "status": "success"
}
```

---

## 📥 Input Fields

> All numeric values must be **StandardScaler-normalised**
> (same scale as training data).
> Use the **Samples** or **Random** button in the UI to
> auto-fill from the actual training dataset.

| Field | Description | Example |
|-------|-------------|---------|
| pl_rade | Planet radius (Earth radii, scaled) | -0.673 |
| pl_bmasse | Planet mass (Earth masses, scaled) | -0.187 |
| pl_orbper | Orbital period (days, scaled) | 0.660 |
| pl_orbsmax | Semi-major axis (AU, scaled) | 0.875 |
| pl_eqt | Equilibrium temperature (K, scaled) | -2.198 |
| pl_dens | Planet density (g/cm³, scaled) | -0.120 |
| st_teff | Star temperature (K, scaled) | 0.201 |
| st_lum | Star luminosity (solar, scaled) | 0.058 |
| st_met | Star metallicity [Fe/H], scaled | -1.519 |
| st_spectype | Spectral type (string) | G0 V |

---

## ⚙️ How the 304-Feature Mapping Works

The model was trained on **304 features** but users only
input 10 values. `utils.py` handles this automatically:

1. Accept 9 numeric inputs + spectral type string
2. Compute 4 engineered features (stellar compatibility,
   orbital stability)
3. Read exact OHE column names from
   `model.get_booster().feature_names`
4. Set `st_spectype_G0 V = 1`, all other 290 columns = 0
5. Align all 304 columns to exact model order
6. Return prediction-ready DataFrame

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/anusreechalla09-gif/HABITABILITY-PREDICTION-ML-.git
cd HABITABILITY-PREDICTION-ML-
```

### 2. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Add model file
Place `xgboost.pkl` inside `backend/models/`
(download from the Google Drive link in utils.py)

### 4. Run Flask server
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## ☁️ Deployment — Render.com

| Setting | Value |
|---------|-------|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |

The model is downloaded automatically from Google Drive
at startup. The CSV dataset is extracted from zip at startup.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Source | NASA Exoplanet Archive |
| Data Processing | pandas, numpy |
| ML Framework | scikit-learn |
| Model | XGBoost 2.0 |
| Backend | Flask 3.0 + Flask-CORS |
| Production Server | Gunicorn |
| Frontend | Bootstrap 5, Vanilla JS |
| Hosting | Render.com |
| Model Storage | Google Drive |
| Version Control | GitHub |

---

## 📊 Dataset

- **Source:** NASA Exoplanet Archive
- **Total records:** 33,799 exoplanets
- **Target:** Binary habitability (0 = Not Habitable, 1 = Habitable)
- **Features used:** 9 planetary/stellar inputs expanded to
  304 via feature engineering and One-Hot Encoding of
  291 spectral type categories

---

## 📁 Notebooks

| Notebook | Purpose |
|----------|---------|
| `preprocess.ipynb` | Data cleaning, feature engineering, OHE, scaling |
| `Model_training_1_.ipynb` | Model training, evaluation, comparison, export |

---

## 👩‍💻 Author

**Anusree Challa**

*Built with 🚀 XGBoost · Flask ·  NASA Data*



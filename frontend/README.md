# 🚀 ExoHabitAI — NASA Exoplanet Habitability Predictor

> An AI-powered web app that predicts whether an exoplanet is habitable using a Tuned Random Forest ML model trained on real NASA data.

---

## ✨ Features

- 🧠 **ML Prediction** — Tuned Random Forest with 98.89% accuracy
- 🌍 **Batch CSV Upload** — Predict multiple planets at once
- 📊 **Analytics Dashboard** — Interactive Chart.js visualizations
- 🏆 **Planet Rankings** — Ranked habitability scores
- ⚡ **Real-Time Results** — Flask REST API backend
- 🌌 **Animated UI** — Space-themed with canvas animations

---

## 📊 Model Performance

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 98.89%  |
| Recall    | 97.5%   |
| F1-Score  | 98.73%  |
| ROC-AUC   | 0.9994  |
| Features  | 22      |
| Threshold | 0.9     |

---

## 🛠️ Tech Stack

**Frontend:** HTML5, CSS3, Vanilla JS, Chart.js, Font Awesome, Google Fonts (Orbitron)  
**Backend:** Python, Flask REST API  
**ML Model:** Scikit-learn — Tuned Random Forest  
**Data:** NASA Exoplanet Archive


## 🔮 How It Works

1. Enter planet parameters (radius, mass, orbital period, temperature, etc.)
2. Frontend sends data to Flask REST API
3. Model predicts habitability with confidence score
4. Results displayed with visual analytics

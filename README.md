# 🚀 ExoHabitAI - AI Powered Habitability Predictor

## 🌌 Overview

ExoHabitAI is an AI/ML-based web application that predicts whether a planet is habitable or not based on planetary, stellar, and environmental parameters.

---

## ✨ Features

* Multi-section input system (Planet, Star, Environment)
* Real-time prediction using ML model
* Interactive chart visualization
* AI-generated explanation
* Animated space-themed UI
* Dark/Light mode toggle
* Prediction history tracking

---

## 🛠️ Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

### Backend

* Python
* Flask
* Flask-CORS

### Machine Learning

* Custom trained model

---

## 📁 Project Structure

```bash
ExoHabitAI/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── backend/
│   ├── app.py
│   ├── routes/
│   ├── services/
│   ├── utils/
│   └── model/
```

---

## ⚙️ Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/ExoHabitAI.git
cd ExoHabitAI
```

---

### Step 2: Backend Setup

```bash
cd backend
pip install flask flask-cors
python app.py
```

Backend runs on:
`http://127.0.0.1:5000`

---

### Step 3: Frontend Setup

* Open `frontend` folder in VS Code
* Right-click `index.html`
* Click **Open with Live Server**

Frontend runs on:
`http://127.0.0.1:5500`

---

## 🔗 API Endpoint

### POST /predict

#### Request Body

```json
{
  "mass": 1,
  "radius": 1,
  "gravity": 1,
  "temperature": 300,
  "luminosity": 1,
  "distance": 1,
  "pressure": 1,
  "water": 1
}
```

#### Response

```json
{
  "prediction": "Habitable",
  "confidence": 65
}
```

---

## 🧪 How It Works

1. User inputs data
2. Frontend sends request to backend
3. ML model processes data
4. Result is returned
5. UI displays prediction

---

## 🎯 Future Enhancements

* NASA dataset integration
* User authentication
* Cloud deployment
* Advanced analytics

---

## 👩‍💻 Author

Sneha Sharma

---

## ⭐ Contribution

Feel free to fork and contribute

---

## 📜 License

MIT License

---

## 💡 Inspiration

Exploring life beyond Earth using AI

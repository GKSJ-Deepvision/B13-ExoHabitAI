# ExoHabitAI
## Project Description
ExoHabitAI is an AI-based system that predicts the habitability potential of exoplanets using
planetary and stellar parameters.

## Tech Stack
- Python
- Machine Learning (Logistic Regression)
- Flask
- HTML, CSS, JavaScript
- Data Visualization
- Data Visualization

## Project Status
Phase 0 – Project Initialization Completed ✅
Phase 1 – Frontend Implementation Completed ✅
Phase 2 – Backend API Integration Completed ✅
Phase 3 – Full System Testing Completed ✅

## Model Performance
- **Accuracy**: 99%
- **Test Samples**: 1,089
- **Features**: 22 astronomical parameters
- **Model**: Logistic Regression with feature scaling

## Frontend Documentation
The frontend provides a complete, polished UI to interact with the ExoHabitAI prediction model.

### Usage Instructions
1. **Start the Backend server**:
   ```bash
   cd backend
   python app.py
   ```
   *The backend will run on `http://localhost:5000`.*

2. **Open the Frontend UI**:
   - Open `frontend/index.html` in your web browser
   - The frontend will connect to the backend API automatically

### Features
- **Live Dashboard**: View global metrics and analytics of analyzed exoplanets.
- **Batch Predict**: Upload JSON data representing multiple planets for batch processing.
- **Leaderboard Rankings**: View the top predicted exoplanets retrieved from the backend API.
- **Single Planet Predict**: Enter exoplanet parameters manually for instant habitability analysis.

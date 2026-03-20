# Import Libraries

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import logging
from pathlib import Path
from utils import load_model, validate_input, FEATURES


# Step 1: Create Flask Application 

app = Flask(__name__)

# Enable CORS for frontend connection
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Step 2: Load Trained ML Model

model = load_model()

try:
    # Load ranked dataset for /rank endpoint (relative to this file)
    base_dir = Path(__file__).resolve().parent
    ranked_path = base_dir / "data" / "habitability_ranked.csv"
    ranked_df = pd.read_csv(ranked_path)
    logging.info("Ranked CSV loaded successfully")

except Exception as e:
    logging.error(f"Error loading ranked CSV: {e}")
    ranked_df = pd.DataFrame()


# Home Route (Optional)

@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "ExoHabitAI Backend Running"
    })


# Step 3: /predict Endpoint


@app.route('/predict', methods=['POST'])
def predict():
    """Predict habitability from input data"""

    # Get JSON data
    json_data = request.get_json()


# Step 4: Input Validation

    is_valid, message = validate_input(json_data)

    if not is_valid:
        return jsonify({
            "status": "error",
            "message": message
        }), 400

    # Check if model loaded

    if model is None:
        return jsonify({
            "status": "error",
            "message": "ML Model is not loaded"
        }), 500

    # Convert input JSON to DataFrame
    input_df = pd.DataFrame([json_data], columns=FEATURES)

    # Convert numeric values safely,
    input_df = input_df.astype(float, errors='ignore')

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get probability (if available)
    probability = None
    if hasattr(model, "predict_proba"):
        try:
            probability = float(model.predict_proba(input_df)[0][1])
        except Exception as e:
            logging.warning(f"Probability error: {e}")
            probability = None

    # Force prediction to 0 if probability is low (less than 0.9)
    if probability is not None and probability < 0.9:
        prediction = 0
        probability = 1 - probability  # Invert for confidence in non-habitable
   
    # Step 5: Response Structure
  
    response = {
        "status": "success",
        "prediction_result": int(prediction),
        "confidence_score": probability
    }

    return jsonify(response)

# Step 6: /rank Endpoint

@app.route('/rank', methods=['GET'])
def rank():
    """Return ranked list of exoplanets"""

    # Check if data exists
    if ranked_df.empty:
        return jsonify({
            "status": "error",
            "message": "Ranked data is not available"
        }), 500  

    # Get limit from query params
    limit = request.args.get('limit', default=10, type=int)

    # Ensure Rank column exists
    if "Rank" not in ranked_df.columns:
        return jsonify({
            "status": "error",
            "message": "Rank column missing in dataset"
        }), 500

    # Sort and select top results
    top_planets_df = ranked_df.sort_values(by="Rank").head(limit)

    # Convert to JSON-safe format
    results = top_planets_df.where(pd.notnull(top_planets_df), None).to_dict(orient="records")

    
# Step 7: Response Structure

    return jsonify({
        "status": "success",
        "returned_count": len(results),
        "data": results
    })

# Run Flask App

if __name__ == '__main__':
    # Debug mode for development only
    app.run(debug=True, port=5000)
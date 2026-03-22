from flask import Flask, request, jsonify
import joblib
import pandas as pd
from utils import prepare_input

app = Flask(__name__)

# Load model
model = joblib.load("models/model.pkl")

# Load ranked dataset
ranked_data = pd.read_csv("data/habitability_ranked.csv")


# 🔹 Home Route
@app.route('/')
def home():
    return "✅ ExoHabitAI Backend Running Successfully"


# 🔹 Predict API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Use utils function
        input_df = prepare_input(data, model)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        })


# 🔹 Rank API
@app.route('/rank', methods=['GET'])
def rank():
    try:
        top = ranked_data.sort_values(
            by='habitability_score',
            ascending=False
        ).head(10)

        return jsonify(top.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)})


# 🔹 Run Server
if __name__ == "__main__":
    app.run(debug=True)
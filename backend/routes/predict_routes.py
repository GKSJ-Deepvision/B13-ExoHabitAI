from flask import Blueprint, request, jsonify
from services.prediction_service import predict_planet
from model.model_loader import load_model
import pandas as pd
from utils.logger import logger


predict_bp = Blueprint("predict", __name__)

model = load_model()

@predict_bp.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received"})

    prediction, probability = predict_planet(
        model,
        data
    )

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })
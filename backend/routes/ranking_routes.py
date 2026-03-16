from flask import Blueprint, jsonify
from services.ranking_service import rank_planets

ranking_bp = Blueprint("ranking", __name__)

@ranking_bp.route("/rank", methods=["GET"])
def rank():

    planets = rank_planets()

    return jsonify({
        "status": "success",
        "planets": planets
    })
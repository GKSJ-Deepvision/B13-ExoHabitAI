from flask import Flask, jsonify
from routes.predict_routes import predict_bp
from routes.ranking_routes import ranking_bp

app = Flask(__name__)

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Register Blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(ranking_bp)


@app.route("/")
def home():
    return jsonify({
        "message": "ExoHabitAI Backend Running"
    })


if __name__ == "__main__":
    app.run(debug=True)

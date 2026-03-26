from flask import Flask, jsonify, request
from flask_cors import CORS


import joblib
import numpy as np
import sqlite3
from datetime import datetime
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from flask import send_file
import pandas as pd
import sqlite3
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
def init_db():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS exoplanets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            planet_mass_earth REAL,
            orbital_period_days REAL,
            orbit_distance_au REAL,
            star_temperature_k REAL,
            star_radius_solar REAL,
            habitability_score REAL,
            is_habitable INTEGER,
            category TEXT
        )
    """)
    conn.commit()
    conn.close()

app = Flask(__name__)
CORS(app)
init_db()


model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Force non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

def generate_feature_importance_base64():
    if not hasattr(model, 'feature_importances_'):
        return None
    
    features = ['Mass (Earth)', 'Period (days)', 'Distance (AU)', 
                'Star Temp (K)', 'Star Radius (R☉)']
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=importances[indices], y=np.array(features)[indices], 
                ax=ax, palette='viridis')
    ax.set_title("Feature Importance – Random Forest")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_score_distribution_base64():
    conn = sqlite3.connect("database.db")
    df = pd.read_sql_query("SELECT habitability_score AS score FROM exoplanets WHERE habitability_score IS NOT NULL", conn)
    conn.close()
    
    if df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df['score'], kde=True, color='teal', ax=ax, bins=15)
    ax.set_title("Habitability Score Distribution")
    ax.set_xlabel("Score (0–1)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_correlation_base64():
    conn = sqlite3.connect("database.db")
    df = pd.read_sql_query("""
        SELECT planet_mass_earth, orbital_period_days, orbit_distance_au,
               star_temperature_k, star_radius_solar, habitability_score
        FROM exoplanets
        WHERE habitability_score IS NOT NULL
    """, conn)
    conn.close()
    
    if len(df) < 2:
        return None
    
    corr = df.corr()
    if corr.empty or corr.isna().all().all():
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("Parameter Correlations")
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

from flask import send_from_directory

@app.route("/")
def home():
    return send_from_directory("frontend", "page.html")

@app.route("/css/<path:filename>")
def css_files(filename):
    return send_from_directory("frontend/css", filename)

@app.route("/js/<path:filename>")
def js_files(filename):
    return send_from_directory("frontend/js", filename)


@app.route('/analytics_charts', methods=['GET'])
def analytics_charts():
    feat_b64 = generate_feature_importance_base64()
    dist_b64 = generate_score_distribution_base64()
    corr_b64 = generate_correlation_base64()
    
    has_data = bool(feat_b64 or dist_b64 or corr_b64)
    
    return jsonify({
        'has_data': has_data,
        'feature_importance': feat_b64,
        'score_distribution': dist_b64,
        'correlation': corr_b64
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    try:
        features = [
            float(data["planet_mass_earth"]),
            float(data["orbital_period_days"]),
            float(data["orbit_distance_au"]),
            float(data["star_temperature_k"]),
            float(data["star_radius_solar"])
        ]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing features: {str(e)}"}), 400

    features_scaled = scaler.transform([features])
    prob = model.predict_proba(features_scaled)[0][1]

    return jsonify({
        "habitability_score": float(prob),  # ensure float
        "habitable": int(prob >= 0.5)
    })


@app.route("/add_planet", methods=["POST"])
def add_planet():
    data = request.get_json()
    if not data:
        print("[ERROR] No JSON received")
        return jsonify({"error": "No JSON received"}), 400

    print("[DEBUG] FULL SAVE PAYLOAD RECEIVED:", data)

    try:
        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO exoplanets (
                name,
                planet_mass_earth,
                orbital_period_days,
                orbit_distance_au,
                star_temperature_k,
                star_radius_solar,
                habitability_score,
                is_habitable,
                category
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('name', 'Unnamed Planet'),
            data.get('planet_mass_earth'),
            data.get('orbital_period_days'),
            data.get('orbit_distance_au'),
            data.get('star_temperature_k'),
            data.get('star_radius_solar'),
            data.get('habitability_score'),  # must be coming from frontend
            data.get('is_habitable', 0),
            data.get('category', 'Custom')
        ))

        conn.commit()
        new_id = cur.lastrowid
        print(f"[SUCCESS] Planet saved! New row ID: {new_id}")

        # Quick verify insert
        cur.execute("SELECT * FROM exoplanets WHERE rowid = ?", (new_id,))
        saved = cur.fetchone()
        print("[DEBUG] Just saved row:", saved)

        return jsonify({"message": "Planet saved successfully"})

    except Exception as e:
        conn.rollback()
        print("[ERROR] Save failed:", str(e))
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route("/ranking", methods=["GET"])
def ranking():
    conn = None
    try:
        conn = sqlite3.connect("database.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Removed created_at since the column doesn't exist
        cur.execute("""
            SELECT name, habitability_score, orbit_distance_au, category
            FROM exoplanets 
            ORDER BY habitability_score DESC 
            LIMIT 10
        """)
        rows = cur.fetchall()

        result = []
        for row in rows:
            result.append({
                "name": row["name"] or "Unnamed",
                "habitability_score": float(row["habitability_score"] or 0),
                "orbit_distance_au": row["orbit_distance_au"],
                "category": row["category"] or "Custom"
                # No created_at - removed to prevent error
            })
        print("[DEBUG] Ranking returning:", result)  # optional debug
        return jsonify(result)

    except Exception as e:
        print("Ranking error:", str(e))
        return jsonify([])
    finally:
        if conn:
            conn.close()
@app.route("/export/pdf", methods=["GET"])
def export_pdf():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("SELECT name, habitability_score FROM exoplanets ORDER BY habitability_score DESC LIMIT 10")
    rows = cur.fetchall()
    conn.close()

    file_path = "Top_Exoplanets.pdf"
    c = canvas.Canvas(file_path, pagesize=A4)

    width, height = A4
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Top Habitable Exoplanets Report")
    y -= 40

    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Planet Name")
    c.drawString(300, y, "Habitability Score")
    y -= 20

    for name, score in rows:
        c.drawString(50, y, name)
        c.drawString(300, y, f"{score:.3f}" if score is not None else "N/A")
        y -= 18
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    return send_file(file_path, as_attachment=True, download_name="Top_Exoplanets_Report.pdf")


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

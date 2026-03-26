# ==========================================================
# ExoHabitAI — API Test Script
# Tests all four endpoints: /, /health, /model-info, /predict, /rank
# Run with: python test_api.py
# Requires: Flask server running on http://127.0.0.1:5000
# ==========================================================

import requests
import json

BASE_URL = "http://127.0.0.1:5000"


def print_section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def print_result(resp):
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)
    print(f"Status code: {resp.status_code}")


# ── 1. Health check ──────────────────────────────────────
print_section("1. Health Check  GET /health")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    print_result(r)
except requests.exceptions.ConnectionError:
    print("ERROR: Could not connect. Is the Flask server running?")
    exit(1)


# ── 2. Model info ─────────────────────────────────────────
print_section("2. Model Info  GET /model-info")
r = requests.get(f"{BASE_URL}/model-info", timeout=5)
print_result(r)


# ── 3. Prediction — habitable candidate ──────────────────
print_section("3. Prediction (habitable candidate)  POST /predict")
# Note: Stellar_Flux is derived server-side — do NOT include it in the payload.
# The backend computes: Stellar_Flux = Stellar_Luminosity / (Semi_Major_Axis^2 + 1e-6)
habitable_payload = {
    "Planet_Radius":      2.5,
    "Planet_Mass":        10.0,
    "Orbital_Period":     20.0,
    "Semi_Major_Axis":    0.12,
    "Planet_Density":     3.5,
    "Equilibrium_Temp":   600.0,
    "Stellar_Temp":       5200.0,
    "Stellar_Luminosity": 0.4,
    "Stellar_Metallicity": 0.0,
    "StarType_A": 0,
    "StarType_F": 0,
    "StarType_G": 0,
    "StarType_K": 1,
    "StarType_M": 0,
}
r = requests.post(f"{BASE_URL}/predict", json=habitable_payload, timeout=5)
print_result(r)


# ── 4. Prediction — non-habitable (hot Jupiter) ──────────
print_section("4. Prediction (non-habitable hot Jupiter)  POST /predict")
non_habitable_payload = {
    "Planet_Radius":      12.0,
    "Planet_Mass":        300.0,
    "Orbital_Period":     2.5,
    "Semi_Major_Axis":    0.03,
    "Planet_Density":     0.8,
    "Equilibrium_Temp":   1800.0,
    "Stellar_Temp":       5900.0,
    "Stellar_Luminosity": 1.2,
    "Stellar_Metallicity": 0.1,
    "StarType_A": 0,
    "StarType_F": 0,
    "StarType_G": 1,
    "StarType_K": 0,
    "StarType_M": 0,
}
r = requests.post(f"{BASE_URL}/predict", json=non_habitable_payload, timeout=5)
print_result(r)


# ── 5. Validation error test ─────────────────────────────
print_section("5. Validation Error (missing fields)  POST /predict")
r = requests.post(f"{BASE_URL}/predict", json={"Planet_Radius": 2.0}, timeout=5)
print_result(r)


# ── 6. Planet ranking ─────────────────────────────────────
print_section("6. Top-5 Planet Ranking  GET /rank?n=5")
r = requests.get(f"{BASE_URL}/rank?n=5", timeout=5)
data = r.json()
print(f"Status: {data.get('status')}, Count: {data.get('count')}")
if data.get("top_planets"):
    for i, p in enumerate(data["top_planets"], 1):
        print(f"  #{i} {p.get('Planet_Name', '?'):20s}  prob={p.get('Habitability_Probability', 0):.6f}")

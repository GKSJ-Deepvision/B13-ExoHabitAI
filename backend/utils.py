import numpy as np

# Earth reference
EARTH = {
    "temp": 255,
    "radius": 1,
    "mass": 1
}

# ==============================
# HABITABILITY CHECK
# ==============================
def calculate_habitability(model, temp, radius, mass, distance):
    input_data = np.array([[temp, radius, mass, distance]])
    prob = model.predict_proba(input_data)[0][1]
    return round(prob * 100, 2)


# ==============================
# FIND CLOSEST PLANET
# ==============================
def find_closest_planet(df, temp, radius, mass):
    df['distance_score'] = (
        abs((df['pl_eqt'] - temp) / 300) +
        abs((df['pl_rade'] - radius) / 2) +
        abs((df['pl_bmasse'] - mass) / 5)
    )

    closest = df.sort_values('distance_score').iloc[0]
    return closest['pl_name']


# ==============================
# EARTH SIMILARITY
# ==============================
def earth_similarity(temp, radius, mass):
    score = (
        abs((temp - EARTH["temp"]) / 300) +
        abs((radius - EARTH["radius"]) / 2) +
        abs((mass - EARTH["mass"]) / 5)
    )

    similarity = round((1 - score) * 100, 2)
    return similarity


# ==============================
# COMPARISON TEXT
# ==============================
def compare(val, earth_val):
    if val > earth_val:
        return "Higher than Earth"
    elif val < earth_val:
        return "Lower than Earth"
    else:
        return "Similar to Earth"
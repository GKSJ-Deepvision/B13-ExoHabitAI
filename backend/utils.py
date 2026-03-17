import numpy as np

# Correct feature order used during model training
FEATURE_ORDER = [
    "Planet_Radius",
    "Planet_Mass",
    "Orbital_Period",
    "Semi_Major_Axis",
    "Planet_Density",
    "Equilibrium_Temp",
    "Stellar_Temp",
    "Stellar_Luminosity",
    "Stellar_Metallicity",
    "StarType_A",
    "StarType_F",
    "StarType_G",
    "StarType_K",
    "StarType_M"
]


def validate_input(data):

    missing = [f for f in FEATURE_ORDER if f not in data]

    if missing:
        raise ValueError(f"Missing features: {missing}")


def prepare_features(data):

    validate_input(data)

    features = []

    for feature in FEATURE_ORDER:
        value = float(data[feature])
        features.append(value)

    return np.array(features).reshape(1, -1)
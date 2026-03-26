import pandas as pd
import numpy as np

# Exact feature order as trained in the model (model.feature_names_in_)
# IMPORTANT: Stellar_Flux is computed from Stellar_Luminosity / Semi_Major_Axis^2
# and must be added before prediction. Do NOT change this order.
FEATURE_ORDER = [
    "Planet_Radius",
    "Planet_Mass",
    "Orbital_Period",
    "Semi_Major_Axis",
    "Equilibrium_Temp",
    "Planet_Density",
    "Stellar_Temp",
    "Stellar_Luminosity",
    "Stellar_Metallicity",
    "Stellar_Flux",        # derived: Luminosity / Semi_Major_Axis^2
    "StarType_A",
    "StarType_F",
    "StarType_G",
    "StarType_K",
    "StarType_M",
]

# Fields the client must supply directly (Stellar_Flux is derived server-side)
REQUIRED_INPUT_FIELDS = [f for f in FEATURE_ORDER if f != "Stellar_Flux"]


def validate_input(data: dict) -> None:
    """Raise ValueError if any required input field is missing."""
    missing = [f for f in REQUIRED_INPUT_FIELDS if f not in data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")


def prepare_features(data: dict) -> pd.DataFrame:
    """
    Validate, derive Stellar_Flux, and return a one-row DataFrame with
    columns in the exact order the model was trained on.

    Stellar_Flux = Stellar_Luminosity / (Semi_Major_Axis^2)
    (inverse square law; 1e-6 epsilon guards against division by zero)
    """
    validate_input(data)

    # Cast all supplied values to float
    row = {f: float(data[f]) for f in REQUIRED_INPUT_FIELDS}

    # Derive Stellar_Flux server-side so the client never has to compute it
    row["Stellar_Flux"] = row["Stellar_Luminosity"] / (
        row["Semi_Major_Axis"] ** 2 + 1e-6
    )

    # Return a DataFrame — sklearn uses column names to align features correctly
    return pd.DataFrame([row], columns=FEATURE_ORDER)

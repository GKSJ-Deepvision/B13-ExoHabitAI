import pandas as pd

def predict_planet(model, features):

    columns = [
        "pl_orbper",
        "pl_rade",
        "pl_bmasse",
        "pl_orbsmax",
        "pl_eqt",
        "st_teff",
        "st_rad",
        "st_mass",
        "st_lum"
    ]

    df = pd.DataFrame([features], columns=columns)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability
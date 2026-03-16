import pandas as pd
import os

def rank_planets():

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(base_dir, "data", "habitability_ranked.csv")

    df = pd.read_csv(csv_path)

    ranked = df.sort_values(by="habitability_score", ascending=False)

    result = ranked[["habitability_score"]].head(10)

    return result.to_dict(orient="records")
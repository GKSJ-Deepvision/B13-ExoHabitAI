import joblib
import os

def load_model():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(BASE_DIR, "..", "models", "final_model.pkl")

    model = joblib.load(model_path)

    return model
import pandas as pd

def prepare_input(data, model):
    """
    Convert JSON input into DataFrame
    matching model feature structure
    """

    input_df = pd.DataFrame([data])

    # Add missing columns
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in correct order
    input_df = input_df[model.feature_names_in_]

    return input_df
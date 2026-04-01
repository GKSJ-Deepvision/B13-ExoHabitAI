
import pandas as pd

try:
    df = pd.read_csv('preprocessed.csv')
    print("Columns:", df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDescribe:")
    print(df.describe())
except Exception as e:
    print(f"Error reading CSV: {e}")

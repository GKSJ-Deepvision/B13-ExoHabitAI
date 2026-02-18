import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Raw Dataset
# -----------------------------
df = pd.read_csv("../data/raw/exoplanet_nasa_raw.csv")

# -----------------------------
# Select Required Features
# -----------------------------
features = [
    "planet_radius",          # Earth radii
    "planet_mass",            # Earth masses
    "orbital_period",         # days
    "semi_major_axis",        # AU
    "equilibrium_temp",       # Kelvin
    "planet_density",
    "star_temp",              # Kelvin
    "luminosity",
    "star_metallicity",
    "star_type"
]

df = df[features]

# -----------------------------
# Remove Duplicates
# -----------------------------
df.drop_duplicates(inplace=True)

# -----------------------------
# Handle Missing Values
# -----------------------------
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

# Numeric → Median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical → Mode
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# -----------------------------
# Remove Physically Impossible Values
# -----------------------------
df = df[
    (df["planet_radius"] > 0) &
    (df["planet_mass"] > 0) &
    (df["equilibrium_temp"] > 0) &
    (df["semi_major_axis"] > 0)
]

# -----------------------------
# Outlier Removal (IQR)
# -----------------------------
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
          (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# -----------------------------
# Feature Engineering
# -----------------------------
epsilon = 1e-6

# Habitability Score Index
df["habitability_score"] = (
    1 / (abs(df["equilibrium_temp"] - 288) + epsilon) +
    1 / (abs(df["planet_radius"] - 1) + epsilon) +
    1 / (abs(df["semi_major_axis"] - 1) + epsilon)
)

# Stellar Compatibility Index
df["stellar_compatibility"] = (
    df["star_temp"] / 5800 +
    df["luminosity"]
)

# Orbital Stability Factor
df["orbital_stability"] = df["orbital_period"] / (df["semi_major_axis"] + epsilon)

# -----------------------------
# Target Variable (Binary)
# -----------------------------
threshold = df["habitability_score"].median()
df["habitable"] = (df["habitability_score"] > threshold).astype(int)

# -----------------------------
# Encode Categorical Feature
# -----------------------------
df = pd.get_dummies(df, columns=["star_type"])

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

scale_cols = df.drop(columns=["habitable"]).columns
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# -----------------------------
# Save Final Preprocessed Dataset
# -----------------------------
df.to_csv("../data/processed/preprocessed.csv", index=False)

print("✅ Preprocessing completed. Dataset ready for model training.")

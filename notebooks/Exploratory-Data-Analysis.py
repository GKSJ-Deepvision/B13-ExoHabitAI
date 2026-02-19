import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# Load Dataset
# -----------------------------------
# This section imports the raw NASA Exoplanet Archive data. 
# We use 'comment=#' to skip the extensive metadata headers provided in the CSV.
original_df = pd.read_csv(
    r"C:\Users\korti\Desktop\infosys_internship\PS_2026.02.09_06.11.21.csv",
    sep=",",
    comment="#",
    engine="python"
)

print("Original Dataset Shape:", original_df.shape)

# -----------------------------------
# Data Quality Assessment
# -----------------------------------
# Data integrity check: Removing exact duplicates and filtering the 289 available 
# columns down to the 10 core physical and stellar parameters required for 
# habitability analysis (Radius, Mass, Temperature, etc.).
print("\nDuplicate Rows:", original_df.duplicated().sum())
original_df = original_df.drop_duplicates()

required_columns = [
    'pl_rade',      # Planet Radius (Earth radii)
    'pl_bmasse',    # Planet Mass (Earth masses)
    'pl_orbper',    # Orbital Period (Days)
    'pl_orbsmax',   # Semi-major axis (AU)
    'pl_eqt',       # Equilibrium Temperature (K)
    'pl_dens',      # Planet Density (g/cm^3)
    'st_teff',      # Star Effective Temperature (K)
    'st_lum',       # Star Luminosity (log10 solar units)
    'st_met',       # Star Metallicity [dex]
    'st_spectype'   # Star Spectral Type (O, B, A, F, G, K, M)
]

df = original_df[required_columns]

print("\nShape After Feature Selection:", df.shape)

# -----------------------------------
# Summary Statistics
# -----------------------------------
# Generates count, mean, std, and quartiles to identify data distribution 
# and the scale of missing values across the selected features.
print("\nSummary Statistics:\n")
print(df.describe())

# -----------------------------------
# Missing Value Heatmap
# -----------------------------------
# Visualizes the sparsity of the dataset. This helps identify which astronomical 
# parameters (like Planet Density or Star Luminosity) have the most missing entries.
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Value Heatmap")
plt.savefig("missing_value_heatmap.png")
plt.close()

print("\nMissing Value Heatmap Saved")

# -----------------------------------
# Handle Missing Values
# -----------------------------------
# Imputation Strategy:
# 1. Numeric: Use Median to avoid bias from extreme astronomical outliers.
# 2. Categorical: Use Mode (most frequent) for the Star Spectral Type.
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# -----------------------------------
# Remove Physically Impossible Values
# -----------------------------------
# Basic physics filter: Radius and Temperature cannot be zero or negative 
# in a realistic planetary model.
df = df[df['pl_rade'] > 0]
df = df[df['pl_eqt'] > 0]

print("\nShape After Removing Impossible Values:", df.shape)

# -----------------------------------
# Z-Score Outlier Analysis
# -----------------------------------
# Measures how many standard deviations data points are from the mean. 
# Large Z-scores (e.g., 198 for orbital period) indicate extreme outliers.
z_scores = np.abs(stats.zscore(df[numeric_cols]))
print("\nMaximum Z-Score per Feature:")
print(pd.DataFrame(z_scores).max())

# -----------------------------------
# IQR Outlier Capping
# -----------------------------------
# Instead of deleting outliers, we "cap" them using the Inter-Quartile Range (IQR). 
# This preserves the data size while preventing extreme values from skewing the ML model.
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

print("\nIQR Outlier Capping Completed")

# -----------------------------------
# Feature Engineering
# -----------------------------------
# Creating domain-specific indicators based on Earth-similarity:
# 1. Habitability Score: Proximity to Earth's temp (288K), radius (1), and orbit (1 AU).
# 2. Stellar Compatibility: Proximity to Sun-like temperature (5778K).
# 3. Orbital Stability: Inverse relationship with extreme orbital parameters.
df['habitability_score'] = (
    (1 / (1 + abs(df['pl_eqt'] - 288))) +
    (1 / (1 + abs(df['pl_rade'] - 1))) +
    (1 / (1 + abs(df['pl_orbsmax'] - 1))) +
    (1 / (1 + abs(df['st_lum'])))
)

df['stellar_compatibility'] = (
    (1 / (1 + abs(df['st_teff'] - 5778))) +
    (1 / (1 + abs(df['st_lum'])))
)

df['orbital_stability'] = (
    (1 / (1 + abs(df['pl_orbper']))) +
    (1 / (1 + abs(df['pl_orbsmax'])))
)

print("\nFeature Engineering Completed")

# -----------------------------------
# Simplify Star Type & One-Hot Encoding
# -----------------------------------
# Extracts the primary spectral class (first letter) and converts it into binary 
# columns (One-Hot Encoding) so the ML model can process categorical stellar data.
df['st_spectype'] = df['st_spectype'].astype(str).str[0]
df = pd.get_dummies(df, columns=['st_spectype'], drop_first=True)

print("\nCategorical Encoding Completed")
print("Shape After Encoding:", df.shape)

# -----------------------------------
# Target Variable Creation
# -----------------------------------
# Since habitability is not directly labeled in the raw data, we define the top 
# 10% of planets (based on our calculated habitability score) as "Habitable" (1), 
# creating a binary classification target.
threshold = df['habitability_score'].quantile(0.90)
df['habitability'] = (df['habitability_score'] > threshold).astype(int)

print("\nHabitability Distribution:\n")
print(df['habitability'].value_counts())

# -----------------------------------
# Feature Scaling
# -----------------------------------
# Standardizes features to a mean of 0 and variance of 1. This ensures that features 
# with large ranges (like Temperature) don't dominate those with small ranges (like AU).
scaler = StandardScaler()

features = df.drop('habitability', axis=1)
scaled_features = scaler.fit_transform(features)

scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['habitability'] = df['habitability'].values

df = scaled_df

print("\nFeature Scaling Completed")
print("Final Dataset Shape:", df.shape)

# -----------------------------------
# Save Preprocessed Dataset
# -----------------------------------
# Exporting the cleaned, engineered, and scaled data for the model training phase.
df.to_csv("preprocessed.csv", index=False)

print("\nPreprocessed dataset saved successfully!")
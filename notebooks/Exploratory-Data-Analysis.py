import os
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

os.makedirs("data/processed", exist_ok=True)
os.makedirs("plots",          exist_ok=True)
os.makedirs("models",         exist_ok=True)

# -----------------------------------
# Load Dataset
# -----------------------------------
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
print("\nDuplicate Rows:", original_df.duplicated().sum())
original_df = original_df.drop_duplicates()

required_columns = [
    'pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbsmax',
    'pl_eqt', 'pl_dens', 'st_teff', 'st_lum', 'st_met', 'st_spectype'
]
df = original_df[required_columns].copy()
print("\nShape After Feature Selection:", df.shape)

# -----------------------------------
# Summary Statistics
# -----------------------------------
print("\nSummary Statistics:\n")
print(df.describe())

# -----------------------------------
# Missing Value Heatmap
# -----------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Value Heatmap")
plt.tight_layout()
plt.savefig("plots/missing_value_heatmap.png", dpi=150)
plt.close()
print("\nMissing Value Heatmap Saved → plots/missing_value_heatmap.png")

# -----------------------------------
# Handle Missing Values
# -----------------------------------
numeric_cols     = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object', 'str']).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------------
# Remove Physically Impossible Values
# -----------------------------------
df = df[df['pl_rade'] > 0]
print("\nShape After Removing Impossible Values:", df.shape)

# -----------------------------------
# Z-Score Outlier Analysis
# -----------------------------------
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
z_scores     = np.abs(stats.zscore(df[numeric_cols]))
print("\nMaximum Z-Score per Feature:")
print(pd.DataFrame(z_scores, columns=numeric_cols).max())

# -----------------------------------
# TARGET VARIABLE CREATION
# -----------------------------------
habitable_mask   = (
    (df['pl_rade'] <  2.0)  &
    (df['st_teff'] >= 4000) & (df['st_teff'] <= 7000)
)
df['habitability'] = habitable_mask.astype(int)

print("\nHabitability Distribution:")
print(df['habitability'].value_counts())
print(f"Habitability Rate: {df['habitability'].mean()*100:.2f}%")

# -----------------------------------
# IQR Outlier Capping
# -----------------------------------
feature_cols = [c for c in numeric_cols if c != 'habitability']
for col in feature_cols:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

print("\nIQR Outlier Capping Completed (features only)")

# -----------------------------------
# Feature Engineering
# -----------------------------------
df['stellar_compatibility'] = (
    (1 / (1 + abs(df['st_teff']  - 5778))) +
    (1 / (1 + abs(df['st_lum'])))
)
df['orbital_stability'] = (
    (1 / (1 + abs(df['pl_orbper']))) +
    (1 / (1 + abs(df['pl_orbsmax'])))
)
print("\nFeature Engineering Completed")

# -----------------------------------
# Spectral Type Encoding
# -----------------------------------
df['st_spectype'] = df['st_spectype'].astype(str).str[0]
df = pd.get_dummies(df, columns=['st_spectype'], drop_first=True)
print("\nCategorical Encoding Completed")
print("Shape After Encoding:", df.shape)

# -----------------------------------
# Drop Zero-Variance Columns
# -----------------------------------
zero_var = [c for c in df.columns if c != 'habitability' and df[c].nunique() <= 1]
if zero_var:
    df.drop(columns=zero_var, inplace=True)
    print(f"\nDropped zero-variance columns: {zero_var}")

# -----------------------------------
# Feature Scaling
# -----------------------------------
# Drop BOTH habitability AND pl_rade before fitting scaler.
# pl_rade defines the target so it must not be a feature.
# This ensures scaler and ML model see the same 16 features.
scaler   = StandardScaler()
features = df.drop(['habitability', 'pl_rade'], axis=1)

scaled_features           = scaler.fit_transform(features)
scaled_df                 = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['habitability'] = df['habitability'].values
df                        = scaled_df

print("\nFeature Scaling Completed")
print("Final Dataset Shape:", df.shape)
print(f"Feature columns ({len(features.columns)}): {list(features.columns)}")

# -----------------------------------
# Save Preprocessed Dataset
# -----------------------------------
df.to_csv("data/processed/exohabit_ml.csv", index=False)
print("\nPreprocessed dataset saved → data/processed/exohabit_ml.csv")

# -----------------------------------
# Save Scaler and Feature Columns
# -----------------------------------
joblib.dump(scaler,                  "models/scaler.pkl")
joblib.dump(list(features.columns),  "models/feature_cols.pkl")
print("Scaler saved        → models/scaler.pkl")
print(f"Feature cols saved  → models/feature_cols.pkl ({len(features.columns)} features)")
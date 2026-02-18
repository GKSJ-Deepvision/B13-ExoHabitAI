import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def preprocess_exoplanet_data(file_path, output_path="data/preprocessed/preprocessed.csv"):
    print("🚀 Starting Production-Grade Preprocessing Pipeline...\n")

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    try:
        df = pd.read_csv(file_path, comment="#", low_memory=False)
        print(f"✅ Raw data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None

    # ============================================================
    # 2. FILTER DEFAULT PARAMETERS
    # ============================================================
    if "default_flag" in df.columns:
        df = df[df["default_flag"] == 1].copy()
        print(f"✅ Filtered default_flag = 1. Shape: {df.shape}")

    # ============================================================
    # 3. SELECT REQUIRED FEATURES
    # ============================================================
    required_columns = {
        'pl_name': 'Planet_Name',
        'pl_rade': 'Planet_Radius',           # Earth radii
        'pl_bmasse': 'Planet_Mass',           # Earth mass
        'pl_orbper': 'Orbital_Period',        # Days
        'pl_orbsmax': 'Semi_Major_Axis',      # AU
        'pl_eqt': 'Equilibrium_Temp',         # Kelvin
        'pl_dens': 'Planet_Density',
        'st_teff': 'Stellar_Temp',            # Kelvin
        'st_lum': 'Stellar_Luminosity',       # log10(L/L☉)
        'st_met': 'Stellar_Metallicity',
        'st_spectype': 'Stellar_Type'
    }

    available_cols = [c for c in required_columns if c in df.columns]
    df = df[available_cols].rename(columns=required_columns).copy()

    print(f"✅ Selected features. Shape: {df.shape}")

    # Remove fully empty rows
    df.dropna(how="all", inplace=True)

    # ============================================================
    # 4. DATA QUALITY REPORT
    # ============================================================
    print("\n📊 Data Quality Assessment")
    print("-" * 40)
    print("Duplicates:", df.duplicated().sum())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nSummary Statistics:\n", df.describe())

    # Missing value heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Missing Values Heatmap")
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/missing_values_heatmap.png")
    plt.close()
    print("✅ Missing value heatmap saved.")

    # ============================================================
    # 5. REMOVE PHYSICALLY IMPOSSIBLE VALUES
    # ============================================================
    df = df[
        (df["Planet_Radius"] > 0) &
        (df["Planet_Mass"] > 0) &
        (df["Equilibrium_Temp"] > 0) &
        (df["Equilibrium_Temp"] < 5000)
    ].copy()

    print(f"✅ Physically invalid values removed. Shape: {df.shape}")

    # ============================================================
    # 6. HANDLE MISSING DATA
    # ============================================================
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = ["Stellar_Type"] if "Stellar_Type" in df.columns else []

    # Median imputation for numeric
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Mode imputation for categorical
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    print("✅ Missing values handled.")

    # ============================================================
    # 7. OUTLIER DETECTION (Z-SCORE + IQR)
    # ============================================================
    for col in numerical_cols:
        # Z-score filtering
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z < 3]

        # IQR capping
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)

    print("✅ Outliers handled.")

    # ============================================================
    # 8. UNIT STANDARDIZATION
    # ============================================================
    # Stellar luminosity conversion from log scale
    df["Stellar_Luminosity"] = 10 ** df["Stellar_Luminosity"]

    print("✅ Units standardized.")

    # ============================================================
    # 9. FEATURE ENGINEERING
    # ============================================================

    # Flux estimation
    flux = df["Stellar_Luminosity"] / (df["Semi_Major_Axis"]**2 + 1e-6)

    # Habitability Score Index
    R_earth, T_earth, S_earth = 1.0, 288.0, 1.0

    h_r = 1 - abs((df["Planet_Radius"] - R_earth) /
                  (df["Planet_Radius"] + R_earth))

    h_t = 1 - abs((df["Equilibrium_Temp"] - T_earth) /
                  (df["Equilibrium_Temp"] + T_earth))

    h_s = 1 - abs((flux - S_earth) / (flux + S_earth))

    df["Habitability_Score_Index"] = (h_r * h_t * h_s) ** (1/3)

    # Stellar Compatibility Index
    df["Stellar_Compatibility_Index"] = np.exp(
        -0.5 * ((df["Stellar_Temp"] - 5778) / 1500) ** 2
    )

    # Orbital Stability Factor
    P_yr = df["Orbital_Period"] / 365.25
    df["Orbital_Stability_Factor"] = np.abs(
        np.log10((df["Semi_Major_Axis"]**3 + 1e-6) / (P_yr**2 + 1e-6))
    )

    print("✅ Feature engineering complete.")

    # ============================================================
    # 10. CATEGORICAL ENCODING (One-Hot)
    # ============================================================
    if "Stellar_Type" in df.columns:
        df["Star_Class"] = df["Stellar_Type"].astype(str).str[0].str.upper()
        valid_classes = ["O", "B", "A", "F", "G", "K", "M"]
        df["Star_Class"] = df["Star_Class"].apply(
            lambda x: x if x in valid_classes else "Other"
        )
        df = pd.get_dummies(df, columns=["Star_Class"], prefix="StarType")
        df.drop(columns=["Stellar_Type"], inplace=True)

    print("✅ Categorical encoding complete.")

    # ============================================================
    # 11. TARGET CREATION (Binary)
    # ============================================================
    is_rocky = (df["Planet_Radius"] <= 1.6) | (df["Planet_Mass"] <= 10)
    is_temp = (df["Equilibrium_Temp"] >= 180) & (df["Equilibrium_Temp"] <= 320)
    is_flux = (flux >= 0.25) & (flux <= 2.2)

    df["Target_Habitable"] = np.where(is_rocky & is_temp & is_flux, 1, 0)

    print("\nTarget Distribution:")
    print(df["Target_Habitable"].value_counts(normalize=True))

    # ============================================================
    # 12. SAVE DATASET
    # ============================================================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n💾 Preprocessed dataset saved to: {output_path}")
    print("🎉 Preprocessing Completed Successfully!")

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "PS_2026.02.13_04.47.45.csv")

    preprocess_exoplanet_data(file_path)

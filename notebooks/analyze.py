import pandas as pd

# 🔹 Correctly load NASA CSV (skip metadata lines starting with #)
df = pd.read_csv(
    "PS_2026.02.07_05.49.09.csv",
    comment="#",
    sep=",",
    low_memory=False
)

# 🔹 Remove any unnamed columns (if exists)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# 🔹 Show shape
rows, cols = df.shape
print("Total Rows:", rows)
print("Total Columns:", cols)

# 🔹 Create column summary
summary = []

for col in df.columns:
    summary.append({
        "Column Name": col,
        "Data Type": str(df[col].dtype),
        "Non-Null Count": df[col].count(),
        "Null Count": df[col].isnull().sum(),
        "Unique Values": df[col].nunique()
    })

summary_df = pd.DataFrame(summary)

# 🔹 Save description file
summary_df.to_csv("dataset_description.csv", index=False)

# 🔹 Save overview text
with open("dataset_overview.txt", "w") as f:
    f.write(f"Total Rows: {rows}\n")
    f.write(f"Total Columns: {cols}\n\n")
    f.write(summary_df.to_string())

print("✅ Dataset description created successfully!")
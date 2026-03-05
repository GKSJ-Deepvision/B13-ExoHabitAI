import os
import warnings
import numpy as np
import pandas as pd
import joblib

# FIX: Use non-GUI backend to avoid tkinter runtime error
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

DATA_PATH = "data/processed/exohabit_ml.csv"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
PROCESSED_DIR = "data/processed"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ============================================================
# 2️⃣ REMOVE LEAKAGE FEATURES
# ============================================================
print("[2] Removing leakage-prone composite features...")

LEAKAGE_COLS = [
    "habitability_score",
    "orbital_stability",
    "stellar_compatibility"
]

for col in LEAKAGE_COLS:
    if col in df.columns:
        df.drop(columns=col, inplace=True)
        print("Dropped:", col)

# Remove zero-variance columns
df = df.loc[:, df.nunique() > 1]

# ============================================================
# 3️⃣ SPLIT DATA
# ============================================================
X = df.drop(columns=["habitability"])
y = df["habitability"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ============================================================
# 4️⃣ DEFINE MODELS
# ============================================================
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

models = {
    "Logistic Regression": lr,
    "Random Forest": rf
}

if XGBOOST_AVAILABLE:
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    models["XGBoost"] = xgb

# ============================================================
# 5️⃣ TRAIN & EVALUATE MODELS
# ============================================================
print("\n[3] Training and evaluating models...")

results = []
model_predictions = {}
model_probabilities = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="f1"
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"\n{name}")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1       :", round(f1, 4))
    print("ROC-AUC  :", round(roc, 4))
    print("CV Mean F1:", round(cv_mean, 4))
    print("CV Std     :", round(cv_std, 4))

    if abs(cv_mean - f1) > 0.05:
        print("⚠ Possible Overfitting Detected")
    else:
        print("✓ Model Generalizes Well")

    results.append((name, f1, rec))
    model_predictions[name] = y_pred
    model_probabilities[name] = y_prob

# ============================================================
# 6️⃣ SELECT BEST MODEL
# ============================================================
best_model_name = sorted(
    results,
    key=lambda x: (x[1], x[2]),
    reverse=True
)[0][0]

best_model = models[best_model_name]

print("\nBest Model Selected:", best_model_name)

# ============================================================
# 7️⃣ SAVE MODELS
# ============================================================
if "Random Forest" in models:
    joblib.dump(models["Random Forest"], os.path.join(MODELS_DIR, "random_forest.pkl"))

if "XGBoost" in models:
    joblib.dump(models["XGBoost"], os.path.join(MODELS_DIR, "xgboost.pkl"))

joblib.dump(best_model, os.path.join(MODELS_DIR, "final_model.pkl"))

# ============================================================
# 8️⃣ CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, model_predictions[best_model_name])

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
plt.close()

# ============================================================
# 9️⃣ ROC CURVE
# ============================================================
fpr, tpr, _ = roc_curve(y_test, model_probabilities[best_model_name])

plt.figure(figsize=(6, 5))
plt.plot(
    fpr,
    tpr,
    label=f"{best_model_name} (AUC={roc_auc_score(y_test, model_probabilities[best_model_name]):.4f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
plt.close()

# ============================================================
# 🔟 HABITABILITY RANKING
# ============================================================
X_full = X.copy()
habit_prob = best_model.predict_proba(X_full)[:, 1]

ranked_df = X_full.copy()
ranked_df["habitability_probability"] = habit_prob

ranked_df = ranked_df.sort_values(
    "habitability_probability",
    ascending=False
)

ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))

ranked_df.to_csv(
    os.path.join(PROCESSED_DIR, "habitability_ranked.csv"),
    index=False
)

# ============================================================
# 1️⃣1️⃣ FEATURE IMPORTANCE
# ============================================================
if hasattr(best_model, "feature_importances_"):

    importances = best_model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, 6))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=feat_df.head(10)
    )

    plt.title("Top 10 Feature Importances")
    plt.tight_layout()

    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"))
    plt.close()

print("\nPipeline Completed Successfully.")
#Added ML pipeline for ExoHabitAI habitability classification

import os
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost model.")

warnings.filterwarnings("ignore")

# -----------------------------------
# Configuration
# -----------------------------------
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5

DATA_PATH     = "data/processed/exohabit_ml.csv"
MODELS_DIR    = "models"
PLOTS_DIR     = "plots"
PROCESSED_DIR = "data/processed"

os.makedirs(MODELS_DIR,    exist_ok=True)
os.makedirs(PLOTS_DIR,     exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================
# STEP 1 — LOAD DATASET
# Doc ref: Section 2 — Dataset Preparation for ML
# ============================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Shape        : {df.shape}")
print(f"Habitable    : {(df['habitability']==1).sum()}")
print(f"Non-Habitable: {(df['habitability']==0).sum()}")
print(f"Habit. Rate  : {df['habitability'].mean()*100:.2f}%")

# ============================================================
# STEP 2 — SEPARATE FEATURES AND TARGET
# Doc ref: Section 2 — Separate features (X) and target (y)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Separating Features and Target")
print("=" * 60)

# Safety: drop leakage columns if present
leakage_cols = ["habitability_score"]
for col in leakage_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)
        print(f"  Dropped leakage column: {col}")

# Drop zero-variance columns
zero_var = [c for c in df.columns if c != "habitability" and df[c].nunique() <= 1]
if zero_var:
    df.drop(columns=zero_var, inplace=True)
    print(f"  Dropped zero-variance: {zero_var}")

X = df.drop(columns=["habitability"])
y = df["habitability"]

# -------------------------------------------------------------------
# CRITICAL: Drop pl_rade from features
# The target was defined as (pl_rade < 2.0) in preprocessing.
# Keeping pl_rade as a feature means the model trivially learns
# "small radius = habitable" — inverting our own label formula.
# This causes AUC = 1.0 with 0.83 feature importance on pl_rade,
# which is not real learning. Dropping it forces the model to find
# genuine signal in stellar and orbital properties instead.
# -------------------------------------------------------------------
if "pl_rade" in X.columns:
    X = X.drop(columns=["pl_rade"])
    print("  Dropped 'pl_rade' — directly defines the target label")

print(f"\n  Features (X) : {X.shape[1]} columns")
print(f"  Features used: {list(X.columns)}")
print(f"  Target   (y) : {y.shape[0]} rows")

# ============================================================
# STEP 3 — TRAIN / TEST SPLIT
# Doc ref: Section 2 — 80% training, 20% testing
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Train-Test Split (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"  Training set : {X_train.shape[0]} rows")
print(f"  Test set     : {X_test.shape[0]} rows")
print(f"  Train balance — 0: {(y_train==0).sum()} | 1: {(y_train==1).sum()}")
print(f"  Test balance  — 0: {(y_test==0).sum()}  | 1: {(y_test==1).sum()}")

# ============================================================
# STEP 4 — DEFINE MODELS
# Doc ref: Section 3 — Baseline: Logistic Regression, Decision Tree
#          Section 4 — Primary: Random Forest, XGBoost
#          Section 5 — Pipelines with StandardScaler
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Defining Models")
print("=" * 60)

# Baseline 1: Logistic Regression (in Pipeline with scaler)
logistic_regression = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

# Baseline 2: Decision Tree shallow (in Pipeline with scaler)
decision_tree = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

# Primary 1: Random Forest
random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

models = {
    "Logistic Regression": logistic_regression,
    "Decision Tree"      : decision_tree,
    "Random Forest"      : random_forest,
}

# Primary 2: XGBoost
if XGBOOST_AVAILABLE:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgboost = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    models["XGBoost"] = xgboost

print(f"  Models defined: {list(models.keys())}")

# ============================================================
# STEP 5 — TRAIN AND EVALUATE ALL MODELS
# Doc ref: Section 6 — Fit on training data
#          Section 7 — Accuracy, Precision, Recall, F1, AUC
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Training and Evaluating All Models")
print("=" * 60)

print(f"\n  {'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV F1':>8} {'CV Std':>7}")
print("  " + "-" * 76)

results     = {}
model_preds = {}
model_probs = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="f1")
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    overfit = "⚠  overfit" if abs(cv_mean - f1) > 0.05 else "✓  stable"

    print(f"  {name:<25} {acc:>6.4f} {prec:>6.4f} {rec:>6.4f} {f1:>6.4f} "
          f"{auc:>6.4f} {cv_mean:>8.4f} {cv_std:>7.4f}  {overfit}")

    results[name]     = {"acc": acc, "prec": prec, "rec": rec,
                         "f1": f1, "auc": auc, "cv_mean": cv_mean, "cv_std": cv_std}
    model_preds[name] = y_pred
    model_probs[name] = y_prob

# ============================================================
# STEP 6 — HYPERPARAMETER TUNING
# Doc ref: Section 8 — RandomizedSearchCV after baseline
#          RF: n_estimators, max_depth
#          XGBoost: learning_rate, max_depth, n_estimators
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Hyperparameter Tuning")
print("=" * 60)

# Tune Random Forest
print("\n  Tuning Random Forest...")

rf_param_dist = {
    "n_estimators"    : [100, 200, 300],
    "max_depth"       : [5, 8, 10, 15, None],
    "min_samples_leaf": [2, 5, 10],
    "max_features"    : ["sqrt", "log2"]
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    param_distributions=rf_param_dist,
    n_iter=20,
    scoring="f1",
    cv=CV_FOLDS,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

y_pred_rf  = best_rf.predict(X_test)
y_prob_rf  = best_rf.predict_proba(X_test)[:, 1]
f1_rf      = f1_score(y_test, y_pred_rf, zero_division=0)
auc_rf     = roc_auc_score(y_test, y_prob_rf)

print(f"  Best params : {rf_search.best_params_}")
print(f"  F1: {f1_rf:.4f} | AUC: {auc_rf:.4f}")

models["RF Tuned"]      = best_rf
model_preds["RF Tuned"] = y_pred_rf
model_probs["RF Tuned"] = y_prob_rf
results["RF Tuned"]     = {
    "acc" : accuracy_score(y_test, y_pred_rf),
    "prec": precision_score(y_test, y_pred_rf, zero_division=0),
    "rec" : recall_score(y_test, y_pred_rf, zero_division=0),
    "f1"  : f1_rf, "auc": auc_rf,
    "cv_mean": rf_search.best_score_, "cv_std": 0.0
}

# Tune XGBoost
if XGBOOST_AVAILABLE:
    print("\n  Tuning XGBoost...")

    xgb_param_dist = {
        "n_estimators" : [100, 200, 300],
        "max_depth"    : [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1]
    }

    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        param_distributions=xgb_param_dist,
        n_iter=15,
        scoring="f1",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
    f1_xgb     = f1_score(y_test, y_pred_xgb, zero_division=0)
    auc_xgb    = roc_auc_score(y_test, y_prob_xgb)

    print(f"  Best params : {xgb_search.best_params_}")
    print(f"  F1: {f1_xgb:.4f} | AUC: {auc_xgb:.4f}")

    models["XGBoost Tuned"]      = best_xgb
    model_preds["XGBoost Tuned"] = y_pred_xgb
    model_probs["XGBoost Tuned"] = y_prob_xgb
    results["XGBoost Tuned"]     = {
        "acc" : accuracy_score(y_test, y_pred_xgb),
        "prec": precision_score(y_test, y_pred_xgb, zero_division=0),
        "rec" : recall_score(y_test, y_pred_xgb, zero_division=0),
        "f1"  : f1_xgb, "auc": auc_xgb,
        "cv_mean": xgb_search.best_score_, "cv_std": 0.0
    }

# ============================================================
# STEP 7 — MODEL COMPARISON AND SELECTION
# Doc ref: Section 9 — Best F1 + Recall, stable, no overfitting
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Model Comparison and Selection")
print("=" * 60)

best_name  = max(results, key=lambda n: (results[n]["f1"], results[n]["rec"]))
best_model = models[best_name]
best_res   = results[best_name]

print(f"\n  Best Model : {best_name}")
print(f"  F1         : {best_res['f1']:.4f}")
print(f"  Recall     : {best_res['rec']:.4f}")
print(f"  ROC-AUC    : {best_res['auc']:.4f}")
print(f"  Accuracy   : {best_res['acc']:.4f}")
print(f"  Precision  : {best_res['prec']:.4f}")

# ============================================================
# STEP 8 — CLASSIFICATION REPORT
# Doc ref: Section 7 — Mandatory Output
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 8: Classification Report — {best_name}")
print("=" * 60)

print(classification_report(
    y_test,
    model_preds[best_name],
    target_names=["Non-Habitable (0)", "Habitable (1)"]
))

# ============================================================
# STEP 9 — SAVE MODELS
# Doc ref: Section 6 — models/random_forest.pkl, xgboost.pkl
# ============================================================
print("=" * 60)
print("STEP 9: Saving Models")
print("=" * 60)

joblib.dump(models["Random Forest"], os.path.join(MODELS_DIR, "random_forest.pkl"))
joblib.dump(best_rf,                 os.path.join(MODELS_DIR, "random_forest_tuned.pkl"))
joblib.dump(best_model,              os.path.join(MODELS_DIR, "final_model.pkl"))

if XGBOOST_AVAILABLE:
    joblib.dump(models["XGBoost"],   os.path.join(MODELS_DIR, "xgboost.pkl"))
    if "XGBoost Tuned" in models:
        joblib.dump(models["XGBoost Tuned"],
                                     os.path.join(MODELS_DIR, "xgboost_tuned.pkl"))

print(f"  random_forest.pkl       → models/")
print(f"  random_forest_tuned.pkl → models/")
print(f"  xgboost.pkl             → models/")
print(f"  final_model.pkl         → models/  [{best_name}]")

# ============================================================
# STEP 10 — CONFUSION MATRIX
# Doc ref: Section 7 — Mandatory Output: Confusion Matrix
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Confusion Matrix")
print("=" * 60)

cm = confusion_matrix(y_test, model_preds[best_name])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Non-Habitable", "Habitable"],
    yticklabels=["Non-Habitable", "Habitable"]
)
plt.title(f"Confusion Matrix — {best_name}", fontsize=13)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

tn, fp, fn, tp = cm.ravel()
print(f"  Saved → plots/confusion_matrix.png")
print(f"  True Positives  (Habitable correctly detected)    : {tp}")
print(f"  False Negatives (Habitable missed)                : {fn}")
print(f"  False Positives (Non-Habitable wrongly flagged)   : {fp}")
print(f"  True Negatives  (Non-Habitable correctly rejected): {tn}")

# ============================================================
# STEP 11 — ROC CURVE
# Doc ref: Section 7 — Mandatory Output: ROC Curve
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: ROC Curve")
print("=" * 60)

plt.figure(figsize=(8, 6))
for name, y_prob in model_probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['auc']:.3f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random Classifier")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve — All Models", fontsize=13)
plt.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
plt.close()
print(f"  Saved → plots/roc_curve.png")

# ============================================================
# STEP 12 — FEATURE IMPORTANCE
# Doc ref: Section 11 — Mandatory: Feature importance plot +
#          explanation of top features + scientific reasoning
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: Feature Importance")
print("=" * 60)

if hasattr(best_model, "feature_importances_"):
    feat_df = pd.DataFrame({
        "Feature"   : X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(
        x="Importance", y="Feature",
        data=feat_df.head(10), palette="Blues_r"
    )
    plt.title(f"Top 10 Feature Importances — {best_name}", fontsize=13)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"  Saved → plots/feature_importance.png")

    print(f"\n  {'Feature':<30} {'Importance':>10}")
    print("  " + "-" * 42)
    for _, row in feat_df.head(10).iterrows():
        print(f"  {row['Feature']:<30} {row['Importance']:>10.4f}")

    print("\n  Scientific interpretation:")
    print("  → st_teff: star temperature controls UV/radiation levels.")
    print("  → pl_orbsmax: orbital distance determines energy received.")
    print("  → stellar_compatibility: sun-like stars are most stable.")
    print("  → orbital_stability: stable orbits allow life to develop.")
    print("  → st_met: higher metallicity means more rocky planet formation.")

# ============================================================
# STEP 13 — HABITABILITY SCORING AND RANKING
# Doc ref: Section 10 — Rank by predicted probability
#          Output: data/processed/habitability_ranked.csv
# ============================================================
print("\n" + "=" * 60)
print("STEP 13: Habitability Scoring and Ranking")
print("=" * 60)

habit_prob = best_model.predict_proba(X)[:, 1]

ranked_df = X.copy()
ranked_df["habitability_probability"] = habit_prob
ranked_df["predicted_label"]          = best_model.predict(X)
ranked_df["actual_habitability"]      = y.values
ranked_df.sort_values("habitability_probability", ascending=False, inplace=True)
ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))

out_path = os.path.join(PROCESSED_DIR, "habitability_ranked.csv")
ranked_df.to_csv(out_path, index=False)

print(f"  Saved → {out_path}")
print(f"\n  Top 5 most habitable planets:")
print(ranked_df[["rank", "habitability_probability", "actual_habitability"]].head().to_string(index=False))

# ============================================================
# STEP 14 — FINAL RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

summary_df = pd.DataFrame(results).T
summary_df = summary_df[["acc", "prec", "rec", "f1", "auc", "cv_mean", "cv_std"]]
summary_df.columns = ["Accuracy", "Precision", "Recall", "F1", "AUC", "CV_F1", "CV_Std"]
print(summary_df.round(4).to_string())

print(f"\n{'='*60}")
print(f"  WINNER  : {best_name}")
print(f"  F1      : {best_res['f1']:.4f}")
print(f"  AUC     : {best_res['auc']:.4f}")
print(f"  Recall  : {best_res['rec']:.4f}")
print(f"{'='*60}")
print("\nPipeline completed. All outputs saved.")
print("  models/final_model.pkl")
print("  models/random_forest.pkl")
print("  models/xgboost.pkl")
print("  plots/confusion_matrix.png")
print("  plots/roc_curve.png")
print("  plots/feature_importance.png")
print("  data/processed/habitability_ranked.csv")
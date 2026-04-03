"""
MULTICLASS EXOPLANET CLASSIFICATION — v2.1

KEY CHANGES FROM v2:
  v2 defined classes using eq_temp_k + radius_earth, excluded both from features.
  
  v2.1 adds mass_earth to the class definition (Gas-Giant: mass > 300 M⊕ OR radius > 6 R⊕)
  and EXCLUDES it from features. This prevents misclassifying 14,000 M⊕ planets as
  Rocky-Habitable and catches RV-detected gas giants that lack radius measurements.

EXCLUDED FROM FEATURES:
  - eq_temp_k (defines temperature zones)
  - radius_earth (defines Gas-Giant size threshold)
  - mass_earth (defines Gas-Giant mass threshold) ← NEW in v2.1
"""

import copy
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed.csv"
MODELS_PATH = PROJECT_ROOT / "models"
MODELS_PATH.mkdir(exist_ok=True)

RANDOM_STATE = 42

print(f"Data   : {DATA_PATH}")
print(f"Models : {MODELS_PATH}")

# Load & Validate
df = pd.read_csv(DATA_PATH)
print(f"\nShape  : {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

REQUIRED = [
    "eq_temp_k", "radius_earth", "mass_earth",
    "log_stellar_flux", "bulk_density_gcc",
    "habitable_binary", "label_is_measured",
    "orbital_period", "semimajor_axis",
    "star_temp_k", "star_luminosity", "star_metallicity",
]

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise RuntimeError(f"\nSTOP: missing columns — {missing}\n"
                      f"Re-run the corrected main.ipynb first.")

print("All required columns present. Correct CSV confirmed.\n")

# Class Definitions — ZERO feature overlap
T_COLD = 200.0   # K
T_HOT = 350.0    # K
R_GAS_GIANT = 6.0  # R_earth
M_GAS_GIANT = 300.0  # M_earth ← NEW in v2.1

CLASS_NAMES = {
    0: "Cold",
    1: "Rocky-Habitable",
    2: "Hot",
    3: "Gas-Giant"
}


def assign_class(row):
    r = row["radius_earth"]
    t = row["eq_temp_k"]
    m = row["mass_earth"]
    
    # Priority 1 — Gas-Giant by size OR mass
    if r > R_GAS_GIANT or m > M_GAS_GIANT:
        return 3
    # Priority 2 — hot zone
    if t > T_HOT:
        return 2
    # Priority 3 — cold zone
    if t < T_COLD:
        return 0
    # Default — temperate + rocky
    return 1


df["planet_class"] = df.apply(assign_class, axis=1)

print("Full-dataset class distribution:")
for k, name in CLASS_NAMES.items():
    n = (df["planet_class"] == k).sum()
    print(f"  Class {k}  {name:<18s}: {n:5,d}  ({n / len(df) * 100:.1f}%)")

# Feature Set — EXCLUDES all class-defining features + log_stellar_flux
STAR_DUMMIES = [c for c in df.columns if c.startswith("star_class_")]

X_COLS = [
    "orbital_period",
    "semimajor_axis",
    # mass_earth REMOVED — now used in class definition
    "star_temp_k",
    "star_luminosity",
    "star_metallicity",    "log_surface_gravity",
    "bulk_density_gcc",
] + STAR_DUMMIES

missing_x = [c for c in X_COLS if c not in df.columns]
if missing_x:
    raise RuntimeError(f"Missing feature columns: {missing_x}")

print(f"\nFeature count : {len(X_COLS)}")
print(f"Features      : {X_COLS}")
print("\nExplicitly excluded from X:")
print("  eq_temp_k    — defines Cold / Hot / Habitable zone boundaries")
print("  radius_earth — defines Gas-Giant size boundary")
print("  mass_earth   — defines Gas-Giant mass boundary (NEW in v2.1)")
print("  log_stellar_flux — algebraically equivalent to eq_temp_k (LEAKAGE)")

# Measured vs. Inferred split
measured = df[df["label_is_measured"] == 1].copy()
infer_df = df[df["label_is_measured"] == 0].copy()

print(f"\nMeasured rows (training + eval) : {len(measured):,}")
print(f"Inferred rows (scoring only)    : {len(infer_df):,}\n")

print("Measured-set class distribution:")
for k, name in CLASS_NAMES.items():
    n = (measured["planet_class"] == k).sum()
    print(f"  Class {k}  {name:<18s}: {n:5,d}  ({n / len(measured) * 100:.1f}%)")

# Merge classes with < 6 samples
MIN_SAMPLES = 6
rare = [k for k in CLASS_NAMES if (measured["planet_class"] == k).sum() < MIN_SAMPLES]

if rare:
    print(f"\n⚠  Classes with <{MIN_SAMPLES} samples in measured set:")
    for k in rare:
        n = (measured["planet_class"] == k).sum()
        print(f"    Class {k} ({CLASS_NAMES[k]}) — {n} sample(s)")
    print("   Merging into Class 1 (Rocky-Habitable) for training/eval.")
    for k in rare:
        measured.loc[measured["planet_class"] == k, "planet_class"] = 1

active_classes = sorted(measured["planet_class"].unique())
active_class_names = [CLASS_NAMES[k] for k in active_classes]
print(f"\nActive classes : {active_classes}  →  {active_class_names}")

# Remap classes to consecutive integers starting from 0
class_mapping = {old: new for new, old in enumerate(active_classes)}
reverse_mapping = {new: old for old, new in class_mapping.items()}

print(f"\nClass remapping for XGBoost:")
for old, new in class_mapping.items():
    print(f"  {CLASS_NAMES[old]} (original {old}) → {new}")

measured["planet_class_mapped"] = measured["planet_class"].map(class_mapping)

# Train / Test Split
X = measured[X_COLS]
y = measured["planet_class_mapped"]

min_class_count = y.value_counts().min()

if min_class_count >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print("Stratified split applied.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print("Random split (insufficient samples for stratification).")

print(f"\nTrain : {len(X_train):,} rows")
print(f"Test  : {len(X_test):,} rows\n")

# Display train/test distribution
print("── Train class distribution ──────────────────────────────────────")
train_counts = pd.Series(y_train).value_counts().sort_index()
for label, count in train_counts.items():
    original_class = reverse_mapping.get(int(label), int(label))
    name = CLASS_NAMES.get(original_class, f"remapped_label_{label}")
    print(f"  Label {label} ({name:<18s}): {count:4d}  ({count / len(y_train) * 100:.1f}%)")

print("\n── Test class distribution ───────────────────────────────────────")
test_counts = pd.Series(y_test).value_counts().sort_index()
for label, count in test_counts.items():
    original_class = reverse_mapping.get(int(label), int(label))
    name = CLASS_NAMES.get(original_class, f"remapped_label_{label}")
    print(f"  Label {label} ({name:<18s}): {count:4d}  ({count / len(y_test) * 100:.1f}%)")

# Build Pipeline
train_counts = pd.Series(y_train).value_counts()
k_neighbors = max(1, min(5, train_counts.min() - 1))
num_class = len(active_classes)

print(f"\nSMOTE k_neighbors : {k_neighbors}")
print(f"XGBoost num_class : {num_class}")

pipeline = ImbPipeline([
    ("smote", SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=k_neighbors,
        sampling_strategy="not majority",
    )),
    ("xgb", xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )),
])

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "weighted_f1": make_scorer(f1_score, average="weighted", zero_division=0),
}

print("\nRunning 5-fold stratified cross-validation...")
cv_res = cross_validate(
    copy.deepcopy(pipeline),
    X_train, y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1,
)

print("\n── CV Results ─────────────────────────────────────────────")
for label, key in [
    ("Macro F1         ", "test_macro_f1"),
    ("Balanced Accuracy", "test_balanced_accuracy"),
    ("Weighted F1      ", "test_weighted_f1"),
]:
    vals = cv_res[key]
    fold_str = ", ".join(f"{v:.3f}" for v in vals)
    print(f"  {label}: {vals.mean():.4f} ± {vals.std():.4f}  [{fold_str}]")

print()
print("NOTE: High CV std on Macro F1 is expected with small minority classes.")
print("      It tells you more than the mean alone.")

# Fit on Full Training Set
pipeline.fit(X_train, y_train)
print("\nModel fitted on full training set.")

# Test-Set Evaluation
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Remap predictions back to original class labels for reporting
y_pred_original = np.array([reverse_mapping[p] for p in y_pred])
y_test_original = np.array([reverse_mapping[t] for t in y_test])

bal_acc = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
raw_acc = (y_pred == y_test).mean()

print("\n── Test-Set Results ───────────────────────────────────────")
print(f"  Balanced Accuracy : {bal_acc:.4f}   ← PRIMARY METRIC")
print(f"  Macro F1          : {macro_f1:.4f}   ← PRIMARY METRIC")
print(f"  Weighted F1       : {weighted_f1:.4f}")
print(f"  Raw Accuracy      : {raw_acc:.4f}  (misleading — Gas-Giants dominate)")

test_names = [CLASS_NAMES[k] for k in active_classes]
print(f"\n── Per-Class Report ───────────────────────────────────────")
print(classification_report(
    y_test_original, y_pred_original,
    labels=active_classes,
    target_names=test_names,
    zero_division=0,
))

print("SANITY CHECK — naive baseline (always predict Gas-Giant):")
gas_giant_mapped = class_mapping.get(3, class_mapping[max(active_classes)])
naive_bal = balanced_accuracy_score(y_test, np.full(len(y_test), gas_giant_mapped))
naive_macro = f1_score(y_test, np.full(len(y_test), gas_giant_mapped), average="macro", zero_division=0)
print(f"  Balanced Accuracy : {naive_bal:.4f}")
print(f"  Macro F1          : {naive_macro:.4f}")
print(f"  (model improvement: ΔBalAcc={bal_acc - naive_bal:+.4f}, "
      f"ΔMacroF1={macro_f1 - naive_macro:+.4f})")

# Confusion Matrix
cm = confusion_matrix(y_test_original, y_pred_original, labels=active_classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_names, yticklabels=test_names, ax=axes[0])
axes[0].set_title("Confusion Matrix — Raw Counts", fontweight="bold")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=test_names, yticklabels=test_names,
            vmin=0, vmax=1, ax=axes[1])
axes[1].set_title("Confusion Matrix — Row-Normalised (Recall per Class)",
                  fontweight="bold")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig(MODELS_PATH / "multiclass_v2.1_confusion.png", dpi=150)
plt.show()

# Feature Importance
inner_model = pipeline.named_steps["xgb"]
fi_df = (
    pd.DataFrame({
        "Feature": X_COLS,
        "Importance": inner_model.feature_importances_
    })
    .sort_values("Importance", ascending=True)
)

key_features = {
    "log_stellar_flux", "bulk_density_gcc",
    "semimajor_axis", "orbital_period"
}
colors = ["#e74c3c" if f in key_features else "#3498db" for f in fi_df["Feature"]]

plt.figure(figsize=(9, 7))
plt.barh(fi_df["Feature"], fi_df["Importance"], color=colors)
plt.xlabel("Feature Importance (Gain)", fontsize=11)
plt.title("Feature Importance — Multiclass XGBoost v2.1\n"
          "(red = expected physics-driven discriminators)",
          fontweight="bold")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(MODELS_PATH / "multiclass_v2.1_feature_importance.png", dpi=150)
plt.show()

print("\nTop 5 features:")
print(fi_df.tail(5)[["Feature", "Importance"]].to_string(index=False))

top_feature = fi_df.iloc[-1]["Feature"]
top_importance = fi_df.iloc[-1]["Importance"]

if top_importance > 0.5:
    print(f"\n⚠  '{top_feature}' accounts for {top_importance:.1%} of importance.")
    print("   Check whether it's implicitly defining a class boundary.")
else:
    print(f"\n✓  Importance is spread across features (top: {top_feature} "
          f"at {top_importance:.1%}). No single-feature dominance.")

# Probability Distribution per Class
n_active = len(active_classes)
fig, axes = plt.subplots(1, n_active, figsize=(5 * n_active, 4))
if n_active == 1:
    axes = [axes]

for idx, class_id in enumerate(active_classes):
    ax = axes[idx]
    name = CLASS_NAMES[class_id]
    mapped_id = class_mapping[class_id]
    probs = y_pred_proba[:, mapped_id]
    
    ax.hist(probs[y_test_original == class_id], bins=20, alpha=0.7, color="green",
            label=f"Actual {name}")
    ax.hist(probs[y_test_original != class_id], bins=20, alpha=0.5, color="red",
            label=f"Not {name}")
    ax.axvline(0.5, color="black", linestyle="--", lw=1, label="0.5 threshold")
    ax.set_title(f"Class {class_id}: {name}", fontweight="bold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Predicted Probability Distributions by Class", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(MODELS_PATH / "multiclass_v2.1_prob_distributions.png", dpi=150,
            bbox_inches="tight")
plt.show()

# Candidate Scoring with Sanity Check
if len(infer_df) > 0:
    X_infer = infer_df[X_COLS]
    infer_proba = pipeline.predict_proba(X_infer)
    infer_pred = pipeline.predict(X_infer)
    
    # Remap predictions back to original class labels
    infer_pred_original = np.array([reverse_mapping[p] for p in infer_pred])
    
    cands = infer_df[["planet_name", "host_star_name"]].copy()
    cands["predicted_class"] = infer_pred_original
    cands["predicted_label"] = [CLASS_NAMES.get(c, f"class_{c}") for c in infer_pred_original]
    
    for i, class_id in enumerate(active_classes):
        col = f"prob_{CLASS_NAMES[class_id].lower().replace('-', '_').replace(' ', '_')}"
        cands[col] = infer_proba[:, i].round(4)
    
    cands["confidence"] = infer_proba.max(axis=1).round(4)
    
    print("\n── Candidate Scoring Summary ──────────────────────────────")
    for k in active_classes:
        n = (cands["predicted_class"] == k).sum()
        print(f"  Predicted {CLASS_NAMES[k]:<18s}: {n:,}")
    
    habitable_cands = (
        cands[cands["predicted_class"] == 1]
        .sort_values("confidence", ascending=False)
        .reset_index(drop=True)
    )
    
    print(f"\nTotal candidates scored    : {len(cands):,}")
    print(f"Predicted Rocky-Habitable  : {len(habitable_cands):,}")
    print(f"High-confidence (≥0.70)    : {(habitable_cands['confidence'] >= 0.70).sum()}")
    print(f"\nTop-20 Rocky-Habitable candidates:")
    print(habitable_cands.head(20).to_string(index=False))
    
    cands.to_csv(MODELS_PATH / "multiclass_v2.1_candidates.csv", index=False)
    print("\nFull candidate table → models/multiclass_v2.1_candidates.csv")
    
    # Candidate Sanity Check
    print("\n" + "="*80)
    print("── Candidate Sanity Check ────────────────────────────────────────")
    print("="*80)
    
    # Training Rocky-Habitable feature ranges
    rh_mapped_label = class_mapping.get(1, 0)
    train_rh_mask = (y_train == rh_mapped_label)
    train_rh = X_train[train_rh_mask]
    
    # High-confidence Rocky-Habitable candidates
    high_conf_mask = (cands["predicted_class"] == 1) & (cands["confidence"] >= 0.70)
    rh_cands_idx = cands[high_conf_mask].index
    rh_cands = infer_df.loc[rh_cands_idx, X_COLS]
    
    check_features = [
        "log_stellar_flux", "bulk_density_gcc",
        "semimajor_axis", "orbital_period"
    ]
    
    print(f"\nComparing {len(rh_cands)} high-confidence candidates vs. {len(train_rh)} training Rocky-Habitable planets")
    print(f"\n{'Feature':<22s} {'Train RH (p5–p95)':<25s} {'Candidates (p5–p95)':<25s} {'Status'}")
    print("─" * 85)
    
    issues = []
    for feat in check_features:
        if feat not in train_rh.columns or len(rh_cands) == 0:
            continue
        
        t5, t95 = train_rh[feat].quantile([0.05, 0.95])
        c5, c95 = rh_cands[feat].quantile([0.05, 0.95])
        
        # Overlap check
        overlap = min(t95, c95) - max(t5, c5)
        ok = "✓ Overlap" if overlap > 0 else "⚠ NO OVERLAP"
        
        if overlap <= 0:
            issues.append(feat)
        
        print(f"  {feat:<20s} [{t5:8.3f} – {t95:8.3f}]    [{c5:8.3f} – {c95:8.3f}]    {ok}")
    
    if issues:
        print(f"\n⚠  Features with no range overlap: {issues}")
        print("   Candidates may be extrapolating outside the training distribution.")
    else:
        print("\n✓  All checked features overlap with training Rocky-Habitable range.")
        print("   High-confidence candidates are within the learned distribution.")
    
    # Top 5 candidates
    if len(habitable_cands) >= 5:
        print("\n" + "─"*85)
        print("── Top-5 Candidates — Raw Features ─────────────────────────────")
        print("─"*85)
        top5_idx = habitable_cands.head(5).index
        top5 = infer_df.loc[top5_idx, ["planet_name"] + check_features].copy()
        top5["confidence"] = habitable_cands.loc[top5_idx, "confidence"].values
        print(top5[["planet_name", "confidence"] + check_features].to_string(index=False))
    
else:
    print("No inference rows — all planets had measured labels.")

# Save Artifacts
joblib.dump(pipeline, MODELS_PATH / "multiclass_v2.1_pipeline.pkl")
joblib.dump(X_COLS, MODELS_PATH / "multiclass_v2.1_features.pkl")
joblib.dump(CLASS_NAMES, MODELS_PATH / "multiclass_v2.1_class_names.pkl")
joblib.dump(class_mapping, MODELS_PATH / "multiclass_v2.1_class_mapping.pkl")
joblib.dump(reverse_mapping, MODELS_PATH / "multiclass_v2.1_reverse_mapping.pkl")
joblib.dump({
    "thresholds": {
        "T_COLD": T_COLD,
        "T_HOT": T_HOT,
        "R_GAS_GIANT": R_GAS_GIANT,
        "M_GAS_GIANT": M_GAS_GIANT,
    },
    "active_classes": active_classes,
    "class_mapping": class_mapping,
    "reverse_mapping": reverse_mapping,
    "cv_macro_f1_mean": float(cv_res["test_macro_f1"].mean()),
    "cv_macro_f1_std": float(cv_res["test_macro_f1"].std()),
    "test_balanced_accuracy": float(bal_acc),
    "test_macro_f1": float(macro_f1),
}, MODELS_PATH / "multiclass_v2.1_metadata.pkl")

print("\nSaved artifacts:")
for f in sorted(MODELS_PATH.glob("multiclass_v2.1*")):
    print(f"  {f.name}")

print("\n" + "="*70)
print("MULTICLASS v2.1 — Training Complete!")
print("="*70)
print(f"\nKey improvement: mass_earth removed from features")
print(f"  - Now used to define Gas-Giant boundary (>300 M⊕)")
print(f"  - Prevents misclassifying massive planets as Rocky-Habitable")
print(f"\nRun the model with:")
print(f"  python {__file__}")
print(f"\nOr import in backend:")
print(f"  joblib.load('models/multiclass_v2.1_pipeline.pkl')")

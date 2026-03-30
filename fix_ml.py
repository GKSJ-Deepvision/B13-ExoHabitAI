"""
Patch script: fixes 5 bugs in notebooks/ML.ipynb
Run from any directory — uses absolute path.
"""
import json, pathlib, sys

NB_PATH = pathlib.Path(r'c:\Users\konal\OneDrive\Documents\GitHub\B13-ExoHabitAI\notebooks\ML.ipynb')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
fixed = []

for cell in cells:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if ('from imblearn.over_sampling import SMOTE' in src
                and 'import pandas as pd' in src
                and 'RANDOM_STATE = 42' in src):
            cell['source'] = [
                "# Import required libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import os\n",
                "import joblib\n",
                "from pathlib import Path\n",
                "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n",
                "from sklearn.svm import SVC\n",
                "from sklearn.metrics import (\n",
                "    accuracy_score, recall_score, f1_score, roc_auc_score, \n",
                "    confusion_matrix, classification_report, precision_recall_curve,\n",
                "    roc_curve\n",
                ")\n",
                "from xgboost import XGBClassifier\n",
                "from imblearn.over_sampling import SMOTE\n",
                "from imblearn.pipeline import Pipeline as ImbPipeline\n",
                "\n",
                "# Set random seed for reproducibility\n",
                "RANDOM_STATE = 42\n",
                "np.random.seed(RANDOM_STATE)\n",
                "\n",
                "# Define paths using pathlib for robustness\n",
                "PROJECT_ROOT = Path.cwd().parent\n",
                "DATA_PATH    = PROJECT_ROOT / 'data' / 'processed' / 'preprocessed.csv'\n",
                "MODELS_PATH  = PROJECT_ROOT / 'models'\n",
                "\n",
                "# Create models directory if it doesn't exist\n",
                "os.makedirs(MODELS_PATH, exist_ok=True)\n",
                "\n",
                "print(f'Project root: {PROJECT_ROOT}')\n",
                "print(f'Data path:    {DATA_PATH}')\n",
                "print(f'Models path:  {MODELS_PATH}')"
            ]
            cell['outputs'] = []   # clear stale output
            fixed.append('Bug 1 — PROJECT_ROOT + pathlib import + ImbPipeline import')
            break

for cell in cells:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if ('stellar_flux_scaled' in src
                and 'scaler.fit_transform' in src
                and 'semimajor_axis' in src):
            cell['source'] = [
                "# Create stellar flux feature (if not already present)\n",
                "# Stellar flux = Luminosity / Distance^2  (energy reaching the planet)\n",
                "#\n",
                "# FIX (Bug 2): Do NOT scale here — scaling must happen AFTER the train/test\n",
                "# split to avoid the test distribution leaking into the scaler.\n",
                "# The StandardScaler in Section 7 will handle this column.\n",
                "if 'stellar_flux' not in df.columns:\n",
                "    df['stellar_flux'] = (10**df['star_luminosity']) / (df['semimajor_axis']**2)\n",
                "    \n",
                "    # Add RAW (unscaled) column — scaler will normalise it in Section 7\n",
                "    X_cols.append('stellar_flux')\n",
                "    X = df[X_cols]\n",
                "\n",
                "print(f'Updated feature count: {len(X_cols)}')\n",
                "print(df[X_cols].isnull().sum())"
            ]
            cell['outputs'] = []
            fixed.append('Bug 2 — removed premature StandardScaler from feature engineering cell')
            break

for cell in cells:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'X_train_scaled = scaler.fit_transform(X_train_res)' in src:
            cell['source'] = [
                "# Scale features for models that benefit from scaling\n",
                "#\n",
                "# FIX (Bug 4): Fit the scaler ONLY on real training data (X_train),\n",
                "# NOT on the SMOTE-augmented set (X_train_res).  Synthetic points must not\n",
                "# influence the scaler's mean/std — that would be a subtle form of leakage.\n",
                "scaler = StandardScaler()\n",
                "scaler.fit(X_train)                                # Fit on real data only\n",
                "X_train_scaled = scaler.transform(X_train_res)    # Apply to SMOTE-augmented set\n",
                "X_test_scaled  = scaler.transform(X_test)          # Apply to held-out test set\n",
                "\n",
                "# Save scaler — its parameters now reflect the real data distribution\n",
                "joblib.dump(scaler, MODELS_PATH / 'scaler.pkl')\n",
                "print('Scaler fitted on real training data and saved to models/scaler.pkl')"
            ]
            cell['outputs'] = []
            fixed.append('Bug 4 — scaler now fitted on X_train (real data), not X_train_res (SMOTE data)')
            break

for cell in cells:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if ('cross_val_score(model, X_train_scaled, y_train_res' in src
                or 'cross_val_score(model, X_train_res, y_train_res' in src):
            cell['source'] = [
                "# Train and evaluate all models with cross-validation\n",
                "results      = []\n",
                "trained_models = {}\n",
                "predictions  = {}\n",
                "probabilities = {}\n",
                "\n",
                "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)\n",
                "\n",
                "for name, model in models.items():\n",
                "    print(f\"\\n{'='*50}\")\n",
                "    print(f'Training {name}...')\n",
                "    print('='*50)\n",
                "    \n",
                "    # ── FIX (Bug 3): imblearn Pipeline applies SMOTE INSIDE each CV fold ──\n",
                "    # Without this, synthetic points generated from the whole training set\n",
                "    # can appear in both the train and validation splits, inflating CV scores.\n",
                "    if name in ['Logistic Regression', 'SVM']:\n",
                "        cv_pipe = ImbPipeline([\n",
                "            ('smote',  SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)),\n",
                "            ('scaler', StandardScaler()),\n",
                "            ('clf',    model)\n",
                "        ])\n",
                "    else:\n",
                "        cv_pipe = ImbPipeline([\n",
                "            ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)),\n",
                "            ('clf',   model)\n",
                "        ])\n",
                "    \n",
                "    # CV on ORIGINAL (non-augmented) training data\n",
                "    cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=skf, scoring='f1')\n",
                "    \n",
                "    # Final fit on full SMOTE-augmented data for held-out evaluation\n",
                "    if name in ['Logistic Regression', 'SVM']:\n",
                "        model.fit(X_train_scaled, y_train_res)\n",
                "        y_pred = model.predict(X_test_scaled)\n",
                "        y_prob = model.predict_proba(X_test_scaled)[:, 1]\n",
                "    else:\n",
                "        model.fit(X_train_res, y_train_res)\n",
                "        y_pred = model.predict(X_test)\n",
                "        y_prob = model.predict_proba(X_test)[:, 1]\n",
                "    \n",
                "    # Store predictions\n",
                "    trained_models[name] = model\n",
                "    predictions[name]    = y_pred\n",
                "    probabilities[name]  = y_prob\n",
                "    \n",
                "    # ── FIX (Bug 5): Persist every trained model to disk ──\n",
                "    fname = name.lower().replace(' ', '_') + '.pkl'\n",
                "    joblib.dump(model, MODELS_PATH / fname)\n",
                "    print(f'  ✓ Saved → models/{fname}')\n",
                "    \n",
                "    # Calculate metrics\n",
                "    acc = accuracy_score(y_test, y_pred)\n",
                "    rec = recall_score(y_test, y_pred)\n",
                "    f1  = f1_score(y_test, y_pred)\n",
                "    auc = roc_auc_score(y_test, y_prob)\n",
                "    \n",
                "    results.append({\n",
                "        'Model':      name,\n",
                "        'Accuracy':   acc,\n",
                "        'Recall':     rec,\n",
                "        'F1-Score':   f1,\n",
                "        'AUC-ROC':    auc,\n",
                "        'CV_F1_Mean': cv_scores.mean(),\n",
                "        'CV_F1_Std':  cv_scores.std()\n",
                "    })\n",
                "    \n",
                "    print(f'Accuracy:      {acc:.4f}')\n",
                "    print(f'Recall:        {rec:.4f}')\n",
                "    print(f'F1-Score:      {f1:.4f}')\n",
                "    print(f'AUC-ROC:       {auc:.4f}')\n",
                "    print(f'CV F1 Score:   {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})')\n",
                "    print(f'\\nConfusion Matrix:')\n",
                "    print(confusion_matrix(y_test, y_pred))\n",
                "    print(f'\\nClassification Report:')\n",
                "    print(classification_report(y_test, y_pred, target_names=['Non-Habitable', 'Habitable']))"
            ]
            cell['outputs'] = []
            fixed.append('Bug 3 — CV now uses ImbPipeline (SMOTE inside each fold)')
            fixed.append('Bug 5 — all trained models saved to models/<name>.pkl')
            break

for cell in cells:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'Best model by F1-Score' in src and 'results_df' in src and 'best_model_name' in src:
            last = cell['source'][-1]
            if not last.endswith('\n'):
                cell['source'][-1] = last + '\n'
            cell['source'].extend([
                "\n",
                "# Save the best model separately for easy inference\n",
                "joblib.dump(trained_models[best_model_name], MODELS_PATH / 'best_model.pkl')\n",
                "print(f\"Best model '{best_model_name}' also saved to models/best_model.pkl\")"
            ])
            cell['outputs'] = []
            fixed.append('Bug 5 (cont.) — best model saved to models/best_model.pkl')
            break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('\n' + '='*60)
print('✅  ML.ipynb patched successfully!')
print('='*60)
for msg in fixed:
    print(f'  • {msg}')

if len(fixed) < 5:
    missing = 5 - len(fixed)
    print(f'\n⚠  WARNING: only {len(fixed)}/5 patches applied. '
          f'{missing} cell(s) not matched — check cell content.')
    sys.exit(1)

# ExoHabitAI Notebooks

## Overview
This directory contains Jupyter notebooks for the ExoHabitAI project - a machine learning pipeline for exoplanet habitability classification.

## Notebooks

### 1. `main.ipynb` - Data Preprocessing
**Purpose**: Load, clean, and preprocess raw exoplanet data

**Key Steps**:
- Load raw data from NASA Exoplanet Archive
- Handle missing values with MICE imputation
- Feature engineering (stellar flux, surface gravity, bulk density)
- Create target variable (habitable_binary)
- Save preprocessed data

**Output**: `data/processed/preprocessed.csv`

**Run Time**: ~5-10 minutes

---

### 2. `ML.ipynb` - Binary Classification
**Purpose**: Train binary classification models (habitable vs non-habitable)

**Key Steps**:
- Load preprocessed data
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Handle class imbalance with SMOTE
- Evaluate with multiple metrics (F1, Precision, Recall, AUROC, AUPRC)
- Save best model

**Output**: `models/*.pkl` files

**Run Time**: ~5-10 minutes

---

### 3. `multiclass_ML.ipynb` - Multi-Class Classification ⭐ NEW
**Purpose**: Train multi-class classification model (4 planet types)

**Class Definitions**:
- **Class 0: Cold Planets** - Low stellar flux (< 0.25 Earth flux)
- **Class 1: Rocky-Habitable** - Sweet spot for life (temperate, rocky)
- **Class 2: Hot Planets** - High stellar flux (> 2.2 Earth flux)
- **Class 3: Gas Giants** - High mass planets (> 10 Earth masses)

**Key Features**:
- Excludes `radius_earth` to prevent leakage
- Uses only measured labels for reliable training
- SMOTE for handling class imbalance
- XGBoost multi-class classifier
- Comprehensive evaluation and visualization

**Output**: 
- `models/multiclass_xgboost.pkl`
- `models/multiclass_confusion_matrix.png`
- `models/multiclass_feature_importance.png`
- `models/multiclass_probability_distributions.png`

**Run Time**: ~5-10 minutes

---

## Workflow

### Standard Pipeline (Binary Classification)
```
1. Run: main.ipynb
   → Preprocesses data
   → Output: data/processed/preprocessed.csv

2. Run: ML.ipynb
   → Trains binary models
   → Output: models/*.pkl
```

### Multi-Class Pipeline
```
1. Run: main.ipynb
   → Preprocesses data
   → Output: data/processed/preprocessed.csv

2. Run: multiclass_ML.ipynb
   → Trains multi-class model
   → Output: models/multiclass_*.pkl and visualizations
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib
```

Or use the project's `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Key Improvements in Multi-Class Approach

### Advantages
✅ More nuanced classification (4 classes vs 2)
✅ Separates gas giants from rocky planets
✅ Distinguishes hot vs cold non-habitable planets
✅ Avoids radius_earth leakage
✅ Uses only measured labels

### Disadvantages
⚠️ More complex model
⚠️ Requires more training data
⚠️ Harder to interpret

---

## Data Files

### Input
- `data/raw/RawData.csv` - Raw NASA Exoplanet Archive data

### Output
- `data/processed/preprocessed.csv` - Cleaned and preprocessed data

### Models
- `models/*.pkl` - Trained models (binary and multi-class)
- `models/*.png` - Visualizations

---

## Documentation

### PDFs
- `Detailed Dataset Explanation.pdf` - Dataset documentation
- `Milestone-1PDF.pdf` - Project milestone 1
- `Milestone-2Pdf.pdf` - Project milestone 2

---

## Notes

### Binary vs Multi-Class
- **Binary**: Simpler, easier to interpret, good for yes/no decisions
- **Multi-Class**: More detailed, better for understanding planet types

### Feature Selection
- Both approaches exclude `radius_earth` to prevent leakage
- Both exclude `eq_temp_k` (defines target)
- Multi-class uses purist feature set for cleaner results

### Class Imbalance
- Binary: ~0.4% positive class (severe imbalance)
- Multi-Class: More balanced across 4 classes
- Both use SMOTE for handling imbalance

---

## Troubleshooting

### Issue: StratifiedKFold Error
```
ValueError: The least populated class in y has only 1 member...
```
**Solution**: The notebooks handle this automatically with adaptive CV

### Issue: Missing Data
**Solution**: `main.ipynb` handles imputation with MICE

### Issue: Out of Memory
**Solution**: Reduce dataset size or use sampling

---

## Citation

If using these notebooks, please cite:
```
ExoHabitAI: Machine Learning Pipeline for Exoplanet Habitability Classification
Multi-class and binary classification approaches with SMOTE and XGBoost
```

---

**Last Updated**: March 20, 2026
**Status**: Production Ready ✅
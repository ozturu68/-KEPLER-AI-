# 📜 Scripts Directory

Kepler Exoplanet Classification Project - Python Scripts

**Author:** sulegogh  
**Last Updated:** 2025-11-11 20:20:42 UTC  
**Version:** 1.0

---

## 📚 Table of Contents

- [Overview](#overview)
- [Script Categories](#script-categories)
- [Data Pipeline Scripts](#data-pipeline-scripts)
- [Model Training Scripts](#model-training-scripts)
- [Analysis & Inference Scripts](#analysis--inference-scripts)
- [Usage Examples](#usage-examples)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

Bu klasör, Kepler Exoplanet sınıflandırma projesinin tüm Python script'lerini içerir.

**Script Kategorileri:**

1. **Data Pipeline** - Veri indirme, preprocessing, feature engineering
2. **Model Training** - Model eğitimi (v1 baseline, v2 weighted)
3. **Analysis & Inference** - Model analizi ve production inference

**Toplam Script Sayısı:** 11

---

## 📂 Script Categories

### 🗂️ File Structure

```
scripts/
├── download_nasa_data.py              # NASA data downloader
├── preprocess_data.py                 # Data preprocessing
├── scale_features.py                  # Feature scaling
├── engineer_features.py               # Feature engineering
├── select_features.py                 # Feature selection
├── train_model.py                     # v1 baseline training
├── train_model_v2_class_weights.py    # v2 weighted training
├── analyze_model_v2.py                # ✨ v2 model analysis
├── predict_v2.py                      # ✨ Production inference
└── README.md                          # This file
```

---

## 🔄 Data Pipeline Scripts

### 1. `download_nasa_data.py`

**Amaç:** NASA Exoplanet Archive'dan veri indir

**Usage:**

```bash
python scripts/download_nasa_data.py
```

**Output:**

- `data/raw/cumulative.csv` (NASA raw data)

**Features:**

- Automatic retry mechanism
- Progress bar
- Data validation
- Error handling

---

### 2. `preprocess_data.py`

**Amaç:** Veriyi temizle ve preprocess et

**Usage:**

```bash
python scripts/preprocess_data.py
```

**Input:**

- `data/raw/cumulative.csv`

**Output:**

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

**Processing Steps:**

1. Remove duplicates
2. Handle missing values
3. Remove low-variance features
4. Encode target (CANDIDATE=0, CONFIRMED=1, FALSE POSITIVE=2)
5. Train/Val/Test split (70/15/15)

**Key Features:**

- Imbalanced class handling
- Statistical outlier detection
- Data quality checks

---

### 3. `scale_features.py`

**Amaç:** Feature'ları scale et (StandardScaler)

**Usage:**

```bash
python scripts/scale_features.py
```

**Input:**

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

**Output:**

- `data/scaled/train_scaled.csv`
- `data/scaled/val_scaled.csv`
- `data/scaled/test_scaled.csv`

**Scaling Method:**

- StandardScaler (mean=0, std=1)
- Fit on train, transform on val/test
- Scaler saved to `models/scaler.pkl`

---

### 4. `engineer_features.py`

**Amaç:** Yeni feature'lar oluştur

**Usage:**

```bash
python scripts/engineer_features.py
```

**Input:**

- `data/scaled/*.csv`

**Output:**

- `data/engineered/*.csv`

**Engineered Features:**

- `snr_per_transit` = `koi_model_snr / koi_count`
- `transit_depth_to_duration_ratio`
- `stellar_temperature_ratio`
- Polynomial features (degree=2)
- Interaction terms
- 20+ additional features

**Total Features:** ~120 features

---

### 5. `select_features.py`

**Amaç:** En önemli 50 feature'ı seç

**Usage:**

```bash
python scripts/select_features.py
```

**Input:**

- `data/engineered/*.csv`

**Output:**

- `data/selected/*.csv` (50 features)
- `logs/feature_selection.log`

**Selection Methods:**

- Mutual Information
- Recursive Feature Elimination (RFE)
- Variance Threshold
- Correlation analysis

**Final Features:** 50 selected features

---

## 🎓 Model Training Scripts

### 6. `train_model.py`

**Amaç:** v1 baseline model eğit

**Usage:**

```bash
python scripts/train_model.py
```

**Model:** CatBoost Classifier (no class weights)

**Output:**

- `models/baseline/catboost_baseline.pkl`
- `logs/baseline_creation_*.log`

**Metrics:**

- Overall Accuracy: ~81%
- CANDIDATE Recall: ~11% (very poor)

**Note:** ⚠️ Imbalanced data problemi var

---

### 7. `train_model_v2_class_weights.py`

**Amaç:** v2 weighted model eğit (5 strategi)

**Usage:**

```bash
python scripts/train_model_v2_class_weights.py
```

**Strategies:**

1. **Balanced** - sklearn balanced weights
2. **Inverse Frequency** - 1/class_frequency
3. **Manual Aggressive** - [5.0, 3.0, 1.0] ⭐ **BEST**
4. **Sqrt Inverse** - 1/sqrt(class_frequency)
5. **Log Inverse** - 1/log(1 + class_frequency)

**Best Model:**

- **Manual Aggressive** (Strategy 3)
- CANDIDATE Recall: 90.91%
- Overall Accuracy: 75.92%

**Output:**

- `models/v2_class_weights/*.pkl` (5 models)
- `models/v2_final/catboost_v2_final.pkl` (best model)
- `logs/v2_class_weights_*.log`

**Training Time:** ~15-20 minutes (all 5 models)

---

## 🔍 Analysis & Inference Scripts

### 8. `analyze_model_v2.py` ✨

**Amaç:** v2 Final model kapsamlı analizi

**Usage:**

```bash
python scripts/analyze_model_v2.py
```

**Features:**

- Confusion matrix visualization
- Feature importance (top 20)
- Classification report
- Error analysis (per-class)
- Performance metrics

**Output Files:**

```
logs/analysis_v2/
├── confusion_matrix.png               # Confusion matrix heatmap
├── feature_importance_top20.png       # Top 20 features
├── feature_importance_full.csv        # All features CSV
├── class_distribution.png             # True vs predicted
├── error_analysis.png                 # Error breakdown
└── analysis_report.txt                # Comprehensive report
```

**Runtime:** ~30 seconds

**Example Output:**

```
✅ ANALYSIS COMPLETE!

Generated Files:
  1. confusion_matrix.png
  2. feature_importance_top20.png
  3. feature_importance_full.csv
  4. class_distribution.png
  5. error_analysis.png
  6. analysis_report.txt
```

---

### 9. `predict_v2.py` ✨

**Amaç:** Production inference (CLI tool)

**Usage:**

**Basic Prediction:**

```bash
python scripts/predict_v2.py \
    --input data/selected/test_selected.csv \
    --output predictions.csv
```

**With Confidence Scores:**

```bash
python scripts/predict_v2.py \
    --input data/selected/test_selected.csv \
    --output predictions.csv \
    --confidence
```

**Verbose Mode:**

```bash
python scripts/predict_v2.py \
    --input data/selected/test_selected.csv \
    --output predictions.csv \
    --confidence \
    --verbose
```

**Arguments:**

- `--input, -i` : Input CSV file (required)
- `--output, -o` : Output CSV file (required)
- `--confidence, -c` : Include confidence scores (optional)
- `--verbose, -v` : Verbose output (optional)
- `--model` : Custom model path (optional)

**Output Format (without --confidence):**

```csv
koi_fpflag_nt,koi_fpflag_ss,...,prediction,prediction_label
0.0,0.0,...,0,CANDIDATE
0.0,1.0,...,2,FALSE POSITIVE
1.0,0.0,...,1,CONFIRMED
```

**Output Format (with --confidence):**

```csv
...,prediction,prediction_label,confidence,confidence_candidate,confidence_confirmed,confidence_false_positive
...,0,CANDIDATE,0.9234,0.9234,0.0521,0.0245
...,2,FALSE POSITIVE,0.8756,0.0123,0.1121,0.8756
...,1,CONFIRMED,0.8912,0.0345,0.8912,0.0743
```

**Features:**

- ✅ Validation mode (auto-detects target column)
- ✅ Missing value handling
- ✅ Feature validation (50 features)
- ✅ Colored terminal output
- ✅ Progress indicators
- ✅ Error handling

**Runtime:** ~5 seconds (1,435 samples)

---

## 🚀 Usage Examples

### Complete Pipeline (from scratch)

```bash
# 1. Download data
python scripts/download_nasa_data.py

# 2. Preprocess
python scripts/preprocess_data.py

# 3. Scale features
python scripts/scale_features.py

# 4. Engineer features
python scripts/engineer_features.py

# 5. Select features
python scripts/select_features.py

# 6. Train v2 model
python scripts/train_model_v2_class_weights.py

# 7. Analyze model
python scripts/analyze_model_v2.py

# 8. Make predictions
python scripts/predict_v2.py \
    --input data/selected/test_selected.csv \
    --output predictions.csv \
    --confidence
```

**Total Runtime:** ~25-30 minutes

---

### Quick Analysis (model already trained)

```bash
# Analyze existing model
python scripts/analyze_model_v2.py

# Make predictions
python scripts/predict_v2.py \
    --input data/selected/test_selected.csv \
    --output predictions.csv \
    --confidence \
    --verbose
```

**Total Runtime:** ~35 seconds

---

### Custom Prediction

```bash
# Predict on custom data
python scripts/predict_v2.py \
    --input my_custom_data.csv \
    --output my_predictions.csv \
    --confidence

# Note: my_custom_data.csv must have same 50 features as training data
```

---

## 📦 Dependencies

### Core Libraries

```python
# Data manipulation
pandas >= 1.5.0
numpy >= 1.23.0

# Machine learning
scikit-learn >= 1.2.0
catboost >= 1.2.0

# Visualization
matplotlib >= 3.6.0
seaborn >= 0.12.0

# Utilities
tqdm >= 4.65.0
```

### Installation

```bash
# Using requirements.txt
pip install -r requirements.txt

# Or install individually
pip install pandas numpy scikit-learn catboost matplotlib seaborn tqdm
```

### Python Version

```
Python >= 3.10
```

---

## ⚙️ Configuration

### Hardcoded Paths

Script'ler project root'a göre relative path kullanır:

```python
PROJECT_ROOT = Path(__file__).parent.parent

# Model path
MODEL_PATH = PROJECT_ROOT / "models/v2_final/catboost_v2_final.pkl"

# Data paths
TRAIN_DATA = PROJECT_ROOT / "data/selected/train_selected.csv"
TEST_DATA = PROJECT_ROOT / "data/selected/test_selected.csv"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "logs/analysis_v2"
```

### Değiştirilebilir Parametreler

**`train_model_v2_class_weights.py`:**

```python
# Class weights (line ~50)
STRATEGIES = {
    'manual_aggressive': [5.0, 3.0, 1.0],  # Değiştirilebilir
    # ...
}

# CatBoost hyperparameters (line ~120)
model = CatBoostModel(
    iterations=1000,        # Artırılabilir: 2000, 3000
    learning_rate=0.1,      # Azaltılabilir: 0.05, 0.01
    depth=6,                # Değiştirilebilir: 4, 8, 10
    l2_leaf_reg=3,
    # ...
)
```

**`analyze_model_v2.py`:**

```python
# Top N features to plot (line ~180)
plot_feature_importance(importance_df, top_n=20)  # 20 -> 30, 50
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'src'**

```bash
# Solution: Make sure you're in project root
cd ~/Projeler/kepler-new
python scripts/analyze_model_v2.py
```

**2. FileNotFoundError: Model not found**

```bash
# Solution: Train model first
python scripts/train_model_v2_class_weights.py

# Then run analysis
python scripts/analyze_model_v2.py
```

**3. ValueError: Feature count mismatch**

```bash
# Input data must have exactly 50 features
# Check feature selection:
python scripts/select_features.py
```

**4. MemoryError during training**

```bash
# Solution: Reduce batch size or iterations
# Edit train_model_v2_class_weights.py:
# iterations=1000 -> iterations=500
```

**5. Permission denied (executable scripts)**

```bash
# Solution: Make scripts executable
chmod +x scripts/*.py
```

---

## 📊 Performance Benchmarks

### Script Runtimes (on Intel i7-10700K, 32GB RAM)

| Script                            | Runtime | Memory Usage |
| --------------------------------- | ------- | ------------ |
| `download_nasa_data.py`           | ~30s    | ~50 MB       |
| `preprocess_data.py`              | ~5s     | ~100 MB      |
| `scale_features.py`               | ~3s     | ~80 MB       |
| `engineer_features.py`            | ~10s    | ~200 MB      |
| `select_features.py`              | ~15s    | ~300 MB      |
| `train_model.py`                  | ~5 min  | ~500 MB      |
| `train_model_v2_class_weights.py` | ~20 min | ~800 MB      |
| `analyze_model_v2.py`             | ~30s    | ~150 MB      |
| `predict_v2.py`                   | ~5s     | ~100 MB      |

**Total Pipeline Runtime:** ~25-30 minutes

---

## 🎯 Best Practices

### 1. Always use virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 2. Run scripts from project root

```bash
cd ~/Projeler/kepler-new
python scripts/analyze_model_v2.py
```

### 3. Check logs for errors

```bash
# View latest log
ls -lt logs/ | head -n 5
cat logs/v2_class_weights_*.log
```

### 4. Validate data before prediction

```bash
# Check feature count
python -c "import pandas as pd; print(pd.read_csv('data.csv').shape)"
# Should output: (n_samples, 50) or (n_samples, 51) if target exists
```

### 5. Use verbose mode for debugging

```bash
python scripts/predict_v2.py --input data.csv --output pred.csv --verbose
```

---

## 📚 Additional Resources

### Documentation

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Project Documentation

- `README.md` (project root)
- `data/README.md` (data dictionary)
- `models/README.md` (model details)
- `logs/README.md` (log explanations)

---

## 🤝 Contributing

**Author:** sulegogh  
**Date:** 2025-11-11  
**Version:** 1.0

---

## 📝 Notes

- Tüm script'ler Python 3.10+ ile test edilmiştir
- CatBoost GPU support opsiyoneldir (CPU ile de çalışır)
- Log dosyaları otomatik olarak `logs/` klasörüne kaydedilir
- Model dosyaları `models/` klasöründe saklanır
- Data pipeline her adımda ara sonuçları kaydeder

---

**Last Updated:** 2025-11-11 20:20:42 UTC  
**Maintained by:** sulegogh

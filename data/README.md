# 📊 Data Directory

NASA Kepler Exoplanet Dataset - Data Processing Pipeline

**Author:** sulegogh  
**Last Updated:** 2025-11-11 20:22:34 UTC  
**Version:** 1.0  
**Source:** NASA Exoplanet Archive  

---

## 📚 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Dataset Information](#dataset-information)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Feature Dictionary](#feature-dictionary)
- [Class Distribution](#class-distribution)
- [Usage Examples](#usage-examples)
- [Data Quality](#data-quality)

---

## 🎯 Overview

Bu klasör, NASA Kepler Exoplanet sınıflandırma projesinin tüm veri setlerini içerir.

**Data Pipeline:**
```
raw → processed → scaled → engineered → selected
```

**Total Samples:** 9,564  
**Classes:** 3 (CANDIDATE, CONFIRMED, FALSE POSITIVE)  
**Final Features:** 50 (selected from 120+ engineered features)  

---

## 🗂️ Directory Structure

```
data/
├── raw/                    # NASA raw data (original)
│   └── cumulative.csv      # Downloaded from NASA (9,564 samples)
│
├── processed/              # Cleaned & preprocessed
│   ├── train.csv           # 6,694 samples (70%)
│   ├── val.csv             # 1,435 samples (15%)
│   └── test.csv            # 1,435 samples (15%)
│
├── scaled/                 # Scaled features (StandardScaler)
│   ├── train_scaled.csv    # Mean=0, Std=1
│   ├── val_scaled.csv      # Transformed using train scaler
│   └── test_scaled.csv     # Transformed using train scaler
│
├── engineered/             # Engineered features (~120 features)
│   ├── train_engineered.csv
│   ├── val_engineered.csv
│   └── test_engineered.csv
│
├── selected/               # Final features (50 selected) ⭐
│   ├── train_selected.csv  # Ready for training
│   ├── val_selected.csv    # Ready for validation
│   └── test_selected.csv   # Ready for testing
│
└── README.md               # This file
```

---

## 📊 Dataset Information

### Basic Statistics

| Property | Value |
|----------|-------|
| **Source** | NASA Exoplanet Archive |
| **Mission** | Kepler Space Telescope |
| **URL** | https://exoplanetarchive.ipac.caltech.edu/ |
| **Total Samples** | 9,564 |
| **Training Samples** | 6,694 (70%) |
| **Validation Samples** | 1,435 (15%) |
| **Test Samples** | 1,435 (15%) |
| **Original Features** | 40+ |
| **Engineered Features** | 120+ |
| **Final Features** | 50 |
| **Classes** | 3 |
| **Imbalance Ratio** | 1 : 1.39 : 2.45 |

### File Sizes

| File | Size | Rows | Columns |
|------|------|------|---------|
| `raw/cumulative.csv` | ~2.1 MB | 9,564 | 50 |
| `processed/train.csv` | ~1.5 MB | 6,694 | 41 |
| `scaled/train_scaled.csv` | ~1.5 MB | 6,694 | 41 |
| `engineered/train_engineered.csv` | ~3.2 MB | 6,694 | 121 |
| `selected/train_selected.csv` | ~1.2 MB | 6,694 | 51 |

---

## 🔄 Data Processing Pipeline

### Stage 1: Raw Data (`raw/`)

**Source:** NASA Exoplanet Archive  
**File:** `cumulative.csv`  
**Size:** 9,564 samples × 50 columns  

**Description:**
- Original NASA Kepler cumulative dataset
- Contains all KOI (Kepler Object of Interest) candidates
- Includes disposition labels (CANDIDATE, CONFIRMED, FALSE POSITIVE)
- Raw NASA feature names and values

**Download Script:**
```bash
python scripts/download_nasa_data.py
```

---

### Stage 2: Processed Data (`processed/`)

**Script:** `scripts/preprocess_data.py`  
**Output:** `train.csv`, `val.csv`, `test.csv`  

**Processing Steps:**

1. **Remove Duplicates**
   - Duplicated rows removed
   - Unique samples retained

2. **Handle Missing Values**
   - Missing values < 30%: Median imputation
   - Missing values > 30%: Column dropped

3. **Remove Low-Variance Features**
   - Variance threshold: 0.01
   - Constant/near-constant features removed

4. **Encode Target Variable**
   - CANDIDATE → 0
   - CONFIRMED → 1
   - FALSE POSITIVE → 2

5. **Train/Val/Test Split**
   - Stratified split (preserve class distribution)
   - Train: 70% (6,694 samples)
   - Val: 15% (1,435 samples)
   - Test: 15% (1,435 samples)

**Output Features:** 41 features + 1 target

---

### Stage 3: Scaled Data (`scaled/`)

**Script:** `scripts/scale_features.py`  
**Method:** StandardScaler (mean=0, std=1)  

**Scaling Process:**

1. **Fit on Training Data**
   ```python
   scaler = StandardScaler()
   scaler.fit(X_train)
   ```

2. **Transform All Splits**
   ```python
   X_train_scaled = scaler.transform(X_train)
   X_val_scaled = scaler.transform(X_val)
   X_test_scaled = scaler.transform(X_test)
   ```

3. **Save Scaler**
   - `models/scaler.pkl` (for production)

**Why Scaling?**
- Improves convergence speed
- Prevents feature domination
- Required for some algorithms

**Output Features:** 41 scaled features + 1 target

---

### Stage 4: Engineered Data (`engineered/`)

**Script:** `scripts/engineer_features.py`  
**Features:** 120+ total features  

**Feature Engineering Methods:**

#### 1. Domain-Specific Features

**SNR per Transit:**
```python
snr_per_transit = koi_model_snr / koi_count
```
*Measures signal quality per individual transit*

**Transit Depth to Duration Ratio:**
```python
depth_duration_ratio = koi_depth / koi_duration
```
*Indicates transit shape characteristics*

**Stellar Temperature Ratio:**
```python
temp_ratio = koi_steff / koi_teq
```
*Compares stellar and planetary temperatures*

#### 2. Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
```

- Degree: 2
- Includes interactions: x1 * x2
- Includes squares: x1²

#### 3. Statistical Features

- **Error Ratios:** `koi_period_err1 / koi_period`
- **Asymmetry Measures:** `(err1 - err2) / (err1 + err2)`
- **Relative Uncertainties:** `error / value`

**Output Features:** 121 features (41 original + 80 engineered) + 1 target

---

### Stage 5: Selected Data (`selected/`) ⭐

**Script:** `scripts/select_features.py`  
**Final Features:** 50 best features  

**Selection Methods:**

1. **Mutual Information**
   ```python
   from sklearn.feature_selection import mutual_info_classif
   mi_scores = mutual_info_classif(X, y)
   ```

2. **Recursive Feature Elimination (RFE)**
   ```python
   from sklearn.feature_selection import RFE
   rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=50)
   ```

3. **Variance Threshold**
   - Remove features with variance < 0.01

4. **Correlation Analysis**
   - Remove highly correlated features (r > 0.95)

**Selection Criteria:**
- High mutual information with target
- Low correlation with other features
- High variance
- Domain relevance

**Output Features:** 50 selected features + 1 target

---

## 📖 Feature Dictionary

### Top 10 Most Important Features

| Rank | Feature | Importance | Description | Unit |
|------|---------|------------|-------------|------|
| 1 | `koi_score` | 29.59% | NASA disposition score | 0-1 |
| 2 | `koi_max_mult_ev` | 4.81% | Max multiple event statistic | - |
| 3 | `koi_count` | 3.65% | Number of transits observed | count |
| 4 | `koi_model_snr` | 2.34% | Transit signal-to-noise ratio | - |
| 5 | `koi_period` | 2.12% | Orbital period | days |
| 6 | `koi_depth` | 1.98% | Transit depth | ppm |
| 7 | `koi_duration` | 1.87% | Transit duration | hours |
| 8 | `koi_prad` | 1.76% | Planetary radius | Earth radii |
| 9 | `koi_teq` | 1.65% | Equilibrium temperature | Kelvin |
| 10 | `koi_steff` | 1.54% | Stellar effective temperature | Kelvin |

---

### Complete Feature List (50 Features)

#### False Positive Flags (4 features)

| Feature | Description | Values |
|---------|-------------|--------|
| `koi_fpflag_nt` | Not Transit-Like Flag | 0 or 1 |
| `koi_fpflag_ss` | Stellar Eclipse Flag | 0 or 1 |
| `koi_fpflag_co` | Centroid Offset Flag | 0 or 1 |
| `koi_fpflag_ec` | Ephemeris Match Flag | 0 or 1 |

**Description:**
- Binary flags indicating potential false positive scenarios
- 1 = potential false positive indicator
- 0 = no indication of this type of false positive

---

#### Orbital Parameters (9 features)

| Feature | Description | Unit | Typical Range |
|---------|-------------|------|---------------|
| `koi_period` | Orbital period | days | 0.5 - 500 |
| `koi_period_err1` | Period uncertainty (+) | days | 0.00001 - 1 |
| `koi_period_err2` | Period uncertainty (-) | days | -1 - -0.00001 |
| `koi_time0bk` | Transit epoch | BKJD | 120 - 1600 |
| `koi_time0bk_err1` | Epoch uncertainty (+) | days | 0.0001 - 1 |
| `koi_time0bk_err2` | Epoch uncertainty (-) | days | -1 - -0.0001 |
| `koi_impact` | Impact parameter | - | 0 - 1 |
| `koi_impact_err1` | Impact uncertainty (+) | - | 0.001 - 0.5 |
| `koi_impact_err2` | Impact uncertainty (-) | - | -0.5 - -0.001 |

**Notes:**
- `err1` = positive uncertainty
- `err2` = negative uncertainty
- BKJD = Barycentric Kepler Julian Date

---

#### Transit Parameters (9 features)

| Feature | Description | Unit | Typical Range |
|---------|-------------|------|---------------|
| `koi_duration` | Transit duration | hours | 0.5 - 15 |
| `koi_duration_err1` | Duration uncertainty (+) | hours | 0.01 - 2 |
| `koi_duration_err2` | Duration uncertainty (-) | hours | -2 - -0.01 |
| `koi_depth` | Transit depth | ppm | 10 - 100,000 |
| `koi_depth_err1` | Depth uncertainty (+) | ppm | 1 - 10,000 |
| `koi_depth_err2` | Depth uncertainty (-) | ppm | -10,000 - -1 |
| `koi_ror` | Planet-star radius ratio | - | 0.001 - 0.3 |
| `koi_ror_err1` | Radius ratio uncertainty (+) | - | 0.0001 - 0.05 |
| `koi_ror_err2` | Radius ratio uncertainty (-) | - | -0.05 - -0.0001 |

**Notes:**
- ppm = parts per million
- Larger depth = larger planet or grazing transit

---

#### Planetary Properties (6 features)

| Feature | Description | Unit | Typical Range |
|---------|-------------|------|---------------|
| `koi_prad` | Planetary radius | Earth radii | 0.5 - 30 |
| `koi_prad_err1` | Radius uncertainty (+) | Earth radii | 0.01 - 5 |
| `koi_prad_err2` | Radius uncertainty (-) | Earth radii | -5 - -0.01 |
| `koi_teq` | Equilibrium temperature | Kelvin | 200 - 3000 |
| `koi_insol` | Insolation flux | Earth flux | 0.1 - 10,000 |
| `koi_sma` | Semi-major axis | AU | 0.01 - 2 |

**Notes:**
- 1 Earth radius ≈ 6,371 km
- 1 AU ≈ 150 million km
- Earth flux = amount of stellar radiation Earth receives

---

#### Stellar Properties (9 features)

| Feature | Description | Unit | Typical Range |
|---------|-------------|------|---------------|
| `koi_steff` | Stellar effective temperature | Kelvin | 3000 - 7000 |
| `koi_steff_err1` | Stellar temp uncertainty (+) | Kelvin | 10 - 500 |
| `koi_steff_err2` | Stellar temp uncertainty (-) | Kelvin | -500 - -10 |
| `koi_slogg` | Stellar surface gravity | log10(cm/s²) | 3.5 - 5.0 |
| `koi_slogg_err1` | Surface gravity uncertainty (+) | dex | 0.01 - 0.5 |
| `koi_slogg_err2` | Surface gravity uncertainty (-) | dex | -0.5 - -0.01 |
| `koi_srad` | Stellar radius | Solar radii | 0.5 - 3 |
| `koi_smass` | Stellar mass | Solar masses | 0.5 - 2 |
| `koi_kepmag` | Kepler-band magnitude | mag | 8 - 18 |

**Notes:**
- Sun temperature ≈ 5,778 K
- Sun surface gravity ≈ 4.44 (log scale)
- Brighter stars have lower magnitude

---

#### Signal Quality (6 features)

| Feature | Description | Unit | Typical Range |
|---------|-------------|------|---------------|
| `koi_model_snr` | Transit signal-to-noise | - | 5 - 500 |
| `koi_tce_plnt_num` | Planet number in system | - | 1 - 8 |
| `koi_count` | Number of transits | count | 3 - 100+ |
| `koi_num_transits` | Observed transits | count | 3 - 100+ |
| `koi_max_sngle_ev` | Max single event statistic | - | 5 - 100 |
| `koi_max_mult_ev` | Max multiple event statistic | - | 10 - 500 |

**Notes:**
- Higher SNR = more confident detection
- More transits = better orbit determination
- Multi-planet systems have `tce_plnt_num > 1`

---

#### Disposition Metrics (2 features)

| Feature | Description | Unit | Typical Range |
|---------|-------------|------|---------------|
| `koi_score` | Disposition score | 0-1 | 0.0 - 1.0 |
| `koi_pdisposition` | Pipeline disposition | string | CANDIDATE/FALSE POSITIVE |

**Notes:**
- `koi_score` closer to 1.0 = more likely planet
- `koi_score` closer to 0.0 = more likely false positive
- **Most important feature** (29.59% importance)

---

#### Engineered Features (5 examples)

| Feature | Description | Formula |
|---------|-------------|---------|
| `snr_per_transit` | SNR normalized by transit count | `koi_model_snr / koi_count` |
| `depth_duration_ratio` | Transit shape indicator | `koi_depth / koi_duration` |
| `stellar_temp_ratio` | Star-planet temp ratio | `koi_steff / koi_teq` |
| `period_error_ratio` | Relative period uncertainty | `(err1 + abs(err2)) / koi_period` |
| `radius_error_asymmetry` | Error asymmetry measure | `(err1 + err2) / (err1 - err2)` |

---

## 📊 Class Distribution

### Overall Distribution

| Class | Train | Val | Test | Total | Percentage |
|-------|-------|-----|------|-------|------------|
| **CANDIDATE** (0) | 1,385 | 297 | 297 | 1,979 | 20.69% |
| **CONFIRMED** (1) | 1,922 | 412 | 412 | 2,746 | 28.71% |
| **FALSE POSITIVE** (2) | 3,387 | 726 | 726 | 4,839 | 50.60% |
| **Total** | 6,694 | 1,435 | 1,435 | 9,564 | 100.00% |

### Imbalance Analysis

**Class Ratios:**
- CANDIDATE : CONFIRMED : FALSE POSITIVE
- 1.00 : 1.39 : 2.45

**Imbalance Factor:**
- Minority class (CANDIDATE): 20.69%
- Majority class (FALSE POSITIVE): 50.60%
- Imbalance ratio: 2.45:1

**Impact:**
- Without class weights: Model biased towards FALSE POSITIVE
- Baseline model: CANDIDATE recall = 11% ❌
- With class weights: CANDIDATE recall = 90.91% ✅

---

## 💻 Usage Examples

### Load Data in Python

```python
import pandas as pd

# Load training data (final selected features)
df_train = pd.read_csv('data/selected/train_selected.csv')

# Separate features and target
X_train = df_train.drop(columns=['koi_disposition'])
y_train = df_train['koi_disposition'].values

print(f"Training samples: {len(X_train)}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {sorted(y_train.unique())}")

# Output:
# Training samples: 6694
# Features: 50
# Classes: [0, 1, 2]
```

### Load Test Data

```python
# Load test data
df_test = pd.read_csv('data/selected/test_selected.csv')
X_test = df_test.drop(columns=['koi_disposition'])
y_test = df_test['koi_disposition'].values

print(f"Test samples: {len(X_test)}")
```

### Class Distribution

```python
# Check class distribution
import numpy as np

unique, counts = np.unique(y_train, return_counts=True)
for class_id, count in zip(unique, counts):
    class_name = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'][class_id]
    percentage = count / len(y_train) * 100
    print(f"{class_name}: {count} ({percentage:.2f}%)")

# Output:
# CANDIDATE: 1385 (20.69%)
# CONFIRMED: 1922 (28.71%)
# FALSE POSITIVE: 3387 (50.60%)
```

### Feature Statistics

```python
# Feature statistics
print(X_train.describe())

# Check for missing values
print(f"Missing values: {X_train.isnull().sum().sum()}")

# Feature correlation
corr_matrix = X_train.corr()
print(f"Max correlation: {corr_matrix.abs().unstack().sort_values(ascending=False)[1]}")
```

---

## 🔍 Data Quality

### Quality Checks

✅ **No Duplicate Rows**
- All samples are unique

✅ **No Missing Values**
- Missing values handled in preprocessing
- Median imputation applied

✅ **Consistent Splits**
- Train/Val/Test splits are stratified
- Class distribution preserved across splits

✅ **Feature Scaling**
- All features scaled to mean=0, std=1
- Scaler fitted on training data only

✅ **No Data Leakage**
- Validation and test sets never seen during training
- Scaler/selector fitted on training data only

### Data Validation Script

```python
# Validate data integrity
def validate_data():
    import pandas as pd

    # Load all splits
    train = pd.read_csv('data/selected/train_selected.csv')
    val = pd.read_csv('data/selected/val_selected.csv')
    test = pd.read_csv('data/selected/test_selected.csv')

    # Check shapes
    assert train.shape[1] == val.shape[1] == test.shape[1] == 51  # 50 features + 1 target

    # Check class distribution
    train_dist = train['koi_disposition'].value_counts(normalize=True)
    val_dist = val['koi_disposition'].value_counts(normalize=True)
    test_dist = test['koi_disposition'].value_counts(normalize=True)

    # Distributions should be similar (within 2%)
    for cls in [0, 1, 2]:
        assert abs(train_dist[cls] - val_dist[cls]) < 0.02
        assert abs(train_dist[cls] - test_dist[cls]) < 0.02

    print("✅ All validation checks passed!")

validate_data()
```

---

## 📚 References

### Data Source
- **NASA Exoplanet Archive:** https://exoplanetarchive.ipac.caltech.edu/
- **Kepler Mission:** https://www.nasa.gov/mission_pages/kepler/main/index.html
- **Data Documentation:** https://exoplanetarchive.ipac.caltech.edu/docs/data.html

### Related Papers
- Batalha et al. (2013) - "Planetary Candidates Observed by Kepler III"
- Thompson et al. (2018) - "Planetary Candidates Observed by Kepler VIII"
- NASA Exoplanet Science Institute Documentation

### Feature Descriptions
- **KOI:** Kepler Object of Interest
- **TCE:** Threshold Crossing Event
- **SNR:** Signal-to-Noise Ratio
- **ppm:** Parts per million
- **BKJD:** Barycentric Kepler Julian Date

---

## 🔗 Related Documentation

- `README.md` (project root) - Project overview
- `scripts/README.md` - Script usage guide
- `models/README.md` - Model documentation
- `logs/README.md` - Log file explanations

---

## 📝 Notes

### Important Considerations

1. **Imbalanced Classes**
   - FALSE POSITIVE is majority class (50.60%)
   - CANDIDATE is minority class (20.69%)
   - Class weights are essential for good performance

2. **Feature Selection**
   - 50 features selected from 120+ engineered features
   - Selection based on mutual information and RFE
   - Redundant features removed

3. **Data Splits**
   - Stratified split maintains class distribution
   - Random state fixed for reproducibility
   - No overlap between splits

4. **Missing Values**
   - All missing values imputed during preprocessing
   - Final datasets have NO missing values

5. **Scaling**
   - StandardScaler ensures mean=0, std=1
   - Essential for distance-based algorithms
   - Scaler saved for production use

---

## 🤝 Contributing

**Author:** sulegogh  
**Date:** 2025-11-11 20:22:34 UTC  
**Version:** 1.0  

---

**Last Updated:** 2025-11-11 20:22:34 UTC  
**Maintained by:** sulegogh

# 🎨 Feature Engineering Documentation

**Last Updated:** 2025-11-13  
**Status:** ✅ Production Ready

## Overview

Kepler Exoplanet ML feature engineering modülü, ham gezegen verilerinden güçlü özellikler (features) oluşturur.

## Table of Contents

- [Feature Types](#feature-types)
- [Feature Engineering](#feature-engineering)
- [Feature Scaling](#feature-scaling)
- [Feature Selection](#feature-selection)
- [Pipeline Integration](#pipeline-integration)
- [Best Practices](#best-practices)

---

## Feature Types

### Original Features (Base)

NASA Kepler verilerinden gelen orijinal özellikler:

| Feature      | Description                   | Unit         | Range       |
| ------------ | ----------------------------- | ------------ | ----------- |
| `koi_period` | Orbital period                | days         | 0.5 - 730   |
| `koi_prad`   | Planetary radius              | Earth radii  | 0.5 - 30    |
| `koi_teq`    | Equilibrium temperature       | Kelvin       | 200 - 3000  |
| `koi_insol`  | Insolation flux               | Earth flux   | 0.1 - 1000  |
| `koi_steff`  | Stellar effective temperature | Kelvin       | 3000 - 8000 |
| `koi_slogg`  | Stellar surface gravity       | log10(cm/s²) | 2.5 - 5.0   |
| `koi_srad`   | Stellar radius                | Solar radii  | 0.3 - 5.0   |
| `koi_smass`  | Stellar mass                  | Solar masses | 0.5 - 2.0   |

### Planetary Features (Derived)

Fiziksel ilişkilerden türetilen özellikler:

```python
# 1. Planetary Density
density = mass / (radius³)

# 2. Orbital Distance (Semi-major axis)
a = (G × M_star × period²)^(1/3) / (4π²)

# 3. Escape Velocity
v_esc = sqrt(2 × G × M_planet / R_planet)

# 4. Surface Gravity
g = G × M_planet / R_planet²

# 5. Habitable Zone Score
HZ_score = function(teq, stellar_temp, distance)
```

### Interaction Features

Özellikler arası etkileşimler:

```python
# 1. Period × Radius
period_radius_interaction = koi_period × koi_prad

# 2. Temperature × Insolation
temp_insol_interaction = koi_teq × koi_insol

# 3. Stellar Temp × Planetary Radius
stellar_planetary_interaction = koi_steff × koi_prad
```

### Polynomial Features

Yüksek dereceli ilişkiler:

```python
# Degree 2 example
koi_period²
koi_prad²
koi_period × koi_prad
```

---

## Feature Engineering

### Module: `src.features.engineering`

#### ExoplanetFeatureEngineer

**Class:** Ana feature engineering sınıfı

**Initialization:**

```python
from src.features.engineering import ExoplanetFeatureEngineer

engineer = ExoplanetFeatureEngineer(
    create_planetary=True,      # Planetary features oluştur
    create_interactions=True,   # Interaction features oluştur
    create_polynomial=False,    # Polynomial features (opsiyonel)
    degree=2,                   # Polynomial degree
    interaction_only=False      # Sadece interaction (degree=2 için)
)
```

#### Create Planetary Features

**Method:** `create_planetary_features(df: pd.DataFrame) -> pd.DataFrame`

Fiziksel yasalara dayalı özellikler oluşturur:

```python
from src.features.engineering import ExoplanetFeatureEngineer

engineer = ExoplanetFeatureEngineer()
df_enhanced = engineer.create_planetary_features(df)

# Oluşturulan features:
# - planetary_density
# - orbital_distance
# - escape_velocity
# - surface_gravity
# - habitable_zone_score
```

**Example:**

```python
import pandas as pd

# Original data
df = pd.DataFrame({
    'koi_period': [3.52, 10.85, 365.25],
    'koi_prad': [1.5, 2.0, 1.0],
    'koi_teq': [300, 400, 288],
    'koi_steff': [5800, 6000, 5778],
    'koi_srad': [1.0, 1.2, 1.0],
    'koi_smass': [1.0, 1.1, 1.0]
})

engineer = ExoplanetFeatureEngineer()
df_features = engineer.create_planetary_features(df)

print(df_features.columns.tolist())
# Output:
# ['koi_period', 'koi_prad', ..., 'planetary_density',
#  'orbital_distance', 'escape_velocity', ...]
```

---

#### Create Interaction Features

**Method:** `create_interaction_features(df: pd.DataFrame) -> pd.DataFrame`

Manuel olarak tanımlanmış önemli etkileşimler:

```python
engineer = ExoplanetFeatureEngineer()
df_interactions = engineer.create_interaction_features(df)

# Default interactions:
# - koi_period × koi_prad
# - koi_teq × koi_insol
```

**Custom Interactions:**

```python
custom_pairs = [
    ('koi_period', 'koi_steff'),
    ('koi_prad', 'koi_srad'),
    ('koi_teq', 'koi_slogg')
]

df_custom = engineer.create_interaction_features(
    df,
    interaction_pairs=custom_pairs
)
```

---

#### Create Polynomial Features

**Method:** `create_polynomial_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, PolynomialFeatures]`

Sklearn PolynomialFeatures kullanarak otomatik feature oluşturma:

```python
engineer = ExoplanetFeatureEngineer()

# Degree 2 polynomial
df_poly, poly_transformer = engineer.create_polynomial_features(
    df,
    columns=['koi_period', 'koi_prad', 'koi_teq'],
    degree=2,
    interaction_only=False
)

# interaction_only=True sadece çarpımları alır (kare terimleri olmaz)
df_inter, inter_transformer = engineer.create_polynomial_features(
    df,
    columns=['koi_period', 'koi_prad'],
    degree=2,
    interaction_only=True
)
```

**Example:**

```python
# Before: 3 features
# [koi_period, koi_prad, koi_teq]

# After (degree=2, interaction_only=False):
# [1, koi_period, koi_prad, koi_teq,
#  koi_period², koi_period×koi_prad, koi_period×koi_teq,
#  koi_prad², koi_prad×koi_teq,
#  koi_teq²]
# Total: 10 features

# After (degree=2, interaction_only=True):
# [koi_period, koi_prad, koi_teq,
#  koi_period×koi_prad, koi_period×koi_teq,
#  koi_prad×koi_teq]
# Total: 6 features
```

---

#### Fit Transform (Complete Pipeline)

**Method:** `fit_transform(df: pd.DataFrame) -> pd.DataFrame`

Tüm feature engineering adımlarını uygular:

```python
from src.features.engineering import ExoplanetFeatureEngineer

# Full pipeline
engineer = ExoplanetFeatureEngineer(
    create_planetary=True,
    create_interactions=True,
    create_polynomial=False
)

# Transform
df_engineered = engineer.fit_transform(df)

print(f"Original features: {df.shape[1]}")
print(f"Engineered features: {df_engineered.shape[1]}")
print(f"New features: {df_engineered.shape[1] - df.shape[1]}")
```

---

#### Engineer Train/Val/Test

**Function:** `engineer_train_val_test(train, val, test, **kwargs)`

Aynı transformer'ı train/val/test'e uygula:

```python
from src.features.engineering import engineer_train_val_test

# Engineer all splits consistently
train_eng, val_eng, test_eng, engineer = engineer_train_val_test(
    train_df,
    val_df,
    test_df,
    create_planetary=True,
    create_interactions=True,
    create_polynomial=False
)

# All splits have same features
assert list(train_eng.columns) == list(val_eng.columns) == list(test_eng.columns)
```

---

## Feature Scaling

### Module: `src.features.scalers`

#### FeatureScaler

**Class:** Feature scaling için unified interface

**Supported Methods:**

- `standard`: StandardScaler (mean=0, std=1)
- `minmax`: MinMaxScaler (range=[0, 1])
- `robust`: RobustScaler (median, IQR based)

**Initialization:**

```python
from src.features.scalers import FeatureScaler

scaler = FeatureScaler(
    method='standard',           # standard, minmax, robust
    exclude_cols=None            # Scale edilmeyecek kolonlar
)
```

#### Standard Scaling

**Formula:** `z = (x - μ) / σ`

```python
scaler = FeatureScaler(method='standard')
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**Use Case:**

- Normal dağılım
- Outlier az
- Neural networks, SVM

---

#### MinMax Scaling

**Formula:** `x_scaled = (x - x_min) / (x_max - x_min)`

```python
scaler = FeatureScaler(method='minmax')
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)  # Range: [0, 1]
```

**Use Case:**

- Bounded range gerekli
- Neural networks (sigmoid/tanh)
- Image-like data

---

#### Robust Scaling

**Formula:** `x_scaled = (x - median) / IQR`

```python
scaler = FeatureScaler(method='robust')
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
```

**Use Case:**

- Outlier çok
- Skewed distribution
- Robust model gerekli

---

#### Fit Transform

```python
scaler = FeatureScaler(method='standard')

# Combined fit + transform
X_train_scaled = scaler.fit_transform(X_train)

# Only transform (already fitted)
X_val_scaled = scaler.transform(X_val)
```

---

#### Inverse Transform

```python
# Scale
X_scaled = scaler.fit_transform(X)

# Inverse (get original values back)
X_original = scaler.inverse_transform(X_scaled)

# Verify
assert np.allclose(X, X_original)
```

---

#### Exclude Columns

```python
# Don't scale target and ID columns
scaler = FeatureScaler(
    method='standard',
    exclude_cols=['koi_disposition', 'kepid']
)

df_scaled = scaler.fit_transform(df)

# 'koi_disposition' and 'kepid' remain unchanged
```

---

#### Scale Train/Val/Test

**Function:** `scale_train_val_test(train, val, test, method='standard', **kwargs)`

```python
from src.features.scalers import scale_train_val_test

# Scale all splits consistently
train_s, val_s, test_s, scaler = scale_train_val_test(
    train_df,
    val_df,
    test_df,
    method='standard',
    exclude_cols=['koi_disposition']
)

# Scaler is fitted only on train_df
# Then applied to val and test
```

---

#### Get Feature Statistics

```python
scaler = FeatureScaler(method='standard')
scaler.fit(X_train)

# Get scaling statistics
stats = scaler.get_feature_stats()

print(stats)
# Output:
#      feature       mean       std       min       max
# 0  koi_period   15.234    25.123    0.523   730.456
# 1     koi_prad    2.156     3.421    0.612    29.876
```

---

#### Compare Scalers

**Function:** `compare_scalers(df: pd.DataFrame) -> dict`

Farklı scaler'ları karşılaştır:

```python
from src.features.scalers import compare_scalers

results = compare_scalers(df)

for method, stats in results.items():
    print(f"\n{method.upper()}:")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std: {stats['std']}")
    print(f"  Range: [{stats['min']}, {stats['max']}]")
```

**Output:**

```
STANDARD:
  Mean: ~0.0
  Std: ~1.0
  Range: [-3.5, 3.5]

MINMAX:
  Mean: ~0.5
  Std: ~0.28
  Range: [0.0, 1.0]

ROBUST:
  Mean: ~0.0
  Std: ~1.0
  Range: [-5.0, 5.0]
```

---

## Feature Selection

### Module: `src.features.selection`

#### FeatureSelector

**Class:** Feature selection için unified interface

**Supported Methods:**

- `variance`: Low variance features'ları çıkar
- `correlation`: Highly correlated features'ları çıkar
- `importance`: Tree-based importance
- `mutual_info`: Mutual information
- `rfe`: Recursive Feature Elimination

**Initialization:**

```python
from src.features.selection import FeatureSelector

selector = FeatureSelector(
    method='importance',         # Selection method
    n_features=50,              # Number of features to select
    threshold=None,             # Alternative: importance threshold
    exclude_cols=['koi_disposition']  # Don't select from these
)
```

---

#### Remove Low Variance Features

**Function:** `remove_low_variance_features(df, threshold=0.01)`

Varyansı düşük (sabit/quasi-constant) özellikleri çıkar:

```python
from src.features.selection import remove_low_variance_features

# Remove features with variance < 0.01
to_remove = remove_low_variance_features(
    df,
    threshold=0.01,
    exclude_cols=['koi_disposition']
)

print(f"Low variance features: {to_remove}")
# Output: ['constant_column', 'quasi_constant_column']

df_filtered = df.drop(columns=to_remove)
```

---

#### Remove High Correlation Features

**Function:** `remove_high_correlation_features(df, threshold=0.95)`

Yüksek korelasyonlu (redundant) özellikleri çıkar:

```python
from src.features.selection import remove_high_correlation_features

# Remove features with correlation > 0.95
to_remove = remove_high_correlation_features(
    df,
    threshold=0.95,
    exclude_cols=['koi_disposition']
)

print(f"High correlation features: {to_remove}")
df_filtered = df.drop(columns=to_remove)
```

---

#### Get Feature Importance

**Function:** `get_feature_importance(df, target_col, method='importance')`

Feature importance hesapla:

```python
from src.features.selection import get_feature_importance

# Tree-based importance (Random Forest)
importance_df = get_feature_importance(
    df,
    target_col='koi_disposition',
    method='importance',
    n_estimators=100
)

print(importance_df.head(10))
# Output:
#           feature  importance
# 0      koi_period      0.234
# 1         koi_prad      0.187
# 2  planetary_density      0.156
```

**Methods:**

| Method        | Algorithm           | Use Case             |
| ------------- | ------------------- | -------------------- |
| `importance`  | Random Forest       | Tree-based models    |
| `mutual_info` | Mutual Information  | Any model            |
| `correlation` | Pearson correlation | Linear relationships |
| `chi2`        | Chi-squared test    | Categorical target   |

---

#### Select Top Features

**Function:** `select_top_features(importance_df, n_features=50, threshold=None)`

En önemli N feature'ı seç:

```python
from src.features.selection import (
    get_feature_importance,
    select_top_features
)

# 1. Get importance
importance_df = get_feature_importance(df, 'koi_disposition')

# 2. Select top 20
top_features = select_top_features(
    importance_df,
    n_features=20
)

print(f"Selected features: {top_features}")

# 3. Filter dataframe
df_selected = df[top_features + ['koi_disposition']]
```

**Alternative: Threshold-based**

```python
# Select features with importance > 0.01
top_features = select_top_features(
    importance_df,
    threshold=0.01
)
```

---

#### Select Features (Complete)

**Method:** `select_features(df, target_col, n_features=50, method='importance')`

Complete feature selection pipeline:

```python
selector = FeatureSelector()

selected_features, info = selector.select_features(
    df,
    target_col='koi_disposition',
    n_features=50,
    method='importance'
)

print(f"Selected: {len(selected_features)} features")
print(f"Feature names: {selected_features}")

# Info dictionary
print(f"Removed (low variance): {info['removed_variance']}")
print(f"Removed (correlation): {info['removed_correlation']}")
print(f"Final features: {info['selected_features']}")
```

**Returns:**

```python
(
    selected_features: List[str],  # Feature names
    info: Dict[str, Any]           # Selection metadata
)
```

---

#### Transform (Apply Selection)

```python
# 1. Select features
selector = FeatureSelector()
selected_features, info = selector.select_features(
    train_df,
    target_col='koi_disposition',
    n_features=50
)

# 2. Transform
train_selected = selector.transform(train_df)
val_selected = selector.transform(val_df)
test_selected = selector.transform(test_df)

# All have same columns (selected + target)
assert list(train_selected.columns) == list(val_selected.columns)
```

---

#### Select Features Train/Val/Test

**Function:** `select_features_train_val_test(train, val, test, n_features=50)`

```python
from src.features.selection import select_features_train_val_test

# Select and transform all splits
train_s, val_s, test_s, selector, info = select_features_train_val_test(
    train_df,
    val_df,
    test_df,
    n_features=50,
    method='importance',
    target_col='koi_disposition'
)

# Consistent features across splits
assert list(train_s.columns) == list(val_s.columns) == list(test_s.columns)
```

---

## Pipeline Integration

### Complete Feature Engineering + Scaling + Selection

```python
from src.features.engineering import engineer_train_val_test
from src.features.scalers import scale_train_val_test
from src.features.selection import select_features_train_val_test

# 1. Engineer features
train_eng, val_eng, test_eng, engineer = engineer_train_val_test(
    train_df, val_df, test_df,
    create_planetary=True,
    create_interactions=True
)

# 2. Scale features
train_scaled, val_scaled, test_scaled, scaler = scale_train_val_test(
    train_eng, val_eng, test_eng,
    method='standard',
    exclude_cols=['koi_disposition']
)

# 3. Select features
train_final, val_final, test_final, selector, info = select_features_train_val_test(
    train_scaled, val_scaled, test_scaled,
    n_features=50,
    method='importance'
)

print(f"✅ Pipeline complete!")
print(f"Original features: {train_df.shape[1]}")
print(f"After engineering: {train_eng.shape[1]}")
print(f"After selection: {train_final.shape[1]}")
```

---

## Best Practices

### ✅ DO

```python
# 1. Engineer on all data first (no data leakage)
engineer = ExoplanetFeatureEngineer()
df_engineered = engineer.fit_transform(df)

# 2. Then split
train, val, test = split_data(df_engineered)

# 3. Fit scaler only on train
scaler.fit(train)
train_scaled = scaler.transform(train)
val_scaled = scaler.transform(val)

# 4. Select features using train
selector.select_features(train_scaled, target_col='koi_disposition')
train_selected = selector.transform(train_scaled)
val_selected = selector.transform(val_scaled)
```

### ❌ DON'T

```python
# 1. Don't fit on validation/test
scaler.fit(val)  # ❌ Data leakage!

# 2. Don't select different features per split
train_features = selector.select_features(train)
val_features = selector.select_features(val)  # ❌ Inconsistent!

# 3. Don't scale target column
scaler.fit(df)  # ❌ Will scale target too!

# 4. Don't forget to exclude target
engineer.fit_transform(df)  # ✅ Target preserved
```

---

## Performance Metrics

### Benchmark Results

**Dataset:** 6,000 rows, 8 features → 50 features

| Operation          | Time      | Memory    |
| ------------------ | --------- | --------- |
| Planetary Features | 0.05s     | +2 MB     |
| Interactions       | 0.02s     | +1 MB     |
| Polynomial (deg=2) | 0.15s     | +10 MB    |
| Scaling            | 0.03s     | +5 MB     |
| Selection          | 0.20s     | +3 MB     |
| **Total**          | **0.45s** | **21 MB** |

---

## Testing

```bash
# Run feature tests
pytest tests/test_features/ -v

# Specific module
pytest tests/test_features/test_engineering.py -v
pytest tests/test_features/test_scalers.py -v
pytest tests/test_features/test_selection.py -v

# With coverage
pytest tests/test_features/ --cov=src.features --cov-report=html
```

**Test Coverage:**

- `engineering.py`: 84.47%
- `scalers.py`: 91.34%
- `selection.py`: 92.86%

---

## References

- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Exoplanet Physical Properties](https://exoplanets.nasa.gov/)

---

**Status:** ✅ Production Ready  
**Maintainer:** sulegogh  
**Last Updated:** 2025-11-13

# 📊 Data Pipeline Documentation

**Last Updated:** 2025-11-13  
**Status:** ✅ Production Ready

## Overview

Kepler Exoplanet ML veri işleme pipeline'ı, NASA Kepler verilerini model eğitimine hazır hale getiren kapsamlı bir
sistemdir.

## Table of Contents

- [Data Flow](#data-flow)
- [Data Cleaning](#data-cleaning)
- [Data Preprocessing](#data-preprocessing)
- [Data Validation](#data-validation)
- [Usage Examples](#usage-examples)

---

## Data Flow

```
Raw Data (CSV/Parquet)
    ↓
Data Loading
    ↓
Data Validation
    ↓
Data Cleaning
    ├─ Remove Duplicates
    ├─ Handle Outliers
    ├─ Convert Dtypes
    └─ Validate Target
    ↓
Missing Value Handling
    ├─ Analyze Missing
    ├─ Drop High Missing Columns
    └─ Impute Remaining
    ↓
Data Splitting
    ├─ Train (60%)
    ├─ Validation (20%)
    └─ Test (20%)
    ↓
Ready for Feature Engineering
```

---

## Data Cleaning

### Module: `src.data.cleaners`

#### Remove Duplicates

**Function:** `remove_duplicates(df: pd.DataFrame) -> pd.DataFrame`

```python
from src.data.cleaners import remove_duplicates

# Remove exact duplicate rows
df_clean = remove_duplicates(df)

# Check results
print(f"Original: {len(df)} rows")
print(f"After cleaning: {len(df_clean)} rows")
print(f"Removed: {len(df) - len(df_clean)} duplicates")
```

**Test Coverage:** 97%

---

#### Outlier Detection

**Function:** `detect_outliers_iqr(df, columns, multiplier=1.5)`

Interquartile Range (IQR) metodunu kullanarak outlier tespiti.

```python
from src.data.cleaners import detect_outliers_iqr

# Detect outliers
outliers, bounds = detect_outliers_iqr(
    df,
    columns=['koi_period', 'koi_prad'],
    multiplier=1.5
)

print(f"Outlier indices: {outliers}")
print(f"Bounds: {bounds}")
```

**Formula:**

```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower Bound = Q1 - (multiplier × IQR)
Upper Bound = Q3 + (multiplier × IQR)
```

---

#### Handle Outliers

**Function:** `handle_outliers(df, columns, method='clip', multiplier=1.5)`

**Methods:**

- `clip`: Outlier'ları bounds'a kırp
- `remove`: Outlier içeren satırları çıkar

```python
from src.data.cleaners import handle_outliers

# Clip outliers
df_clipped = handle_outliers(
    df,
    columns=['koi_period', 'koi_prad'],
    method='clip',
    multiplier=1.5
)

# Remove outliers
df_removed = handle_outliers(
    df,
    columns=['koi_period', 'koi_prad'],
    method='remove',
    multiplier=2.0  # More conservative
)
```

**Comparison:**

| Method | Pros                  | Cons                     | Use Case            |
| ------ | --------------------- | ------------------------ | ------------------- |
| clip   | Preserves data points | May distort distribution | Large datasets      |
| remove | Clean distribution    | Loses data               | Small outlier ratio |

---

#### Convert Data Types

**Function:** `convert_dtypes(df: pd.DataFrame) -> pd.DataFrame`

Memory optimizasyonu için veri tiplerini dönüştürür:

- `float64` → `float32` (50% memory reduction)
- `int64` → `int32` (50% memory reduction)
- `object` → `category` (up to 90% reduction for low cardinality)

```python
from src.data.cleaners import convert_dtypes

# Before
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Convert
df_optimized = convert_dtypes(df)

# After
print(f"Memory usage: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

---

#### Validate Target Column

**Function:** `validate_target_column(df, target_col='koi_disposition')`

Target sütununu doğrular:

- Sütun var mı?
- Missing value var mı?
- Expected classes mevcut mu?

```python
from src.data.cleaners import validate_target_column
from src.core.exceptions import DataValidationError

try:
    validate_target_column(df, target_col='koi_disposition')
    print("✅ Target column is valid")
except DataValidationError as e:
    print(f"❌ Validation failed: {e}")
```

---

#### Clean Data (All-in-One)

**Function:** `clean_data(df, handle_outliers=True, method='clip', **kwargs)`

Tüm cleaning adımlarını tek seferde uygular:

```python
from src.data.cleaners import clean_data

df_clean = clean_data(
    df,
    handle_outliers=True,
    method='clip',
    outlier_multiplier=1.5,
    optimize_dtypes=True,
    validate_target=True
)
```

**Pipeline:**

1. ✅ Remove duplicates
2. ✅ Validate target column
3. ✅ Drop rows with missing target
4. ✅ Handle outliers (optional)
5. ✅ Convert dtypes (optional)
6. ✅ Final validation

---

## Data Preprocessing

### Module: `src.data.preprocessors`

#### Missing Value Analysis

**Function:** `analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame`

```python
from src.data.preprocessors import analyze_missing_values

# Analyze
report = analyze_missing_values(df)

print(report)
# Output:
#        column  missing_count  missing_pct     dtype
# 0   koi_period             15         1.5   float64
# 1      koi_prad             23         2.3   float64
```

---

#### Missing Value Handler

**Class:** `MissingValueHandler`

**Initialization:**

```python
from src.data.preprocessors import MissingValueHandler

handler = MissingValueHandler(
    threshold=0.7,                        # Drop columns with >70% missing
    numerical_strategy='median',          # median, mean, most_frequent
    categorical_strategy='most_frequent'  # most_frequent, constant
)
```

**Methods:**

```python
# Fit on training data
handler.fit(train_df)

# Transform (fill missing values)
train_filled = handler.transform(train_df)
val_filled = handler.transform(val_df)
test_filled = handler.transform(test_df)

# Or fit_transform
train_filled = handler.fit_transform(train_df)
```

**Strategies:**

| Strategy        | Description   | Best For                      |
| --------------- | ------------- | ----------------------------- |
| `mean`          | Column mean   | Normal distribution           |
| `median`        | Column median | Skewed distribution, outliers |
| `most_frequent` | Mode value    | Categorical, discrete         |
| `constant`      | Fixed value   | Domain knowledge              |

---

#### Simple Imputer

**Class:** `SimpleImputer` (sklearn wrapper)

```python
from src.data.preprocessors import SimpleImputer

# Create imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform
X = df.drop(columns=['koi_disposition'])
imputer.fit(X)
X_filled = imputer.transform(X)  # Returns numpy array

# Convert back to DataFrame
X_filled_df = pd.DataFrame(
    X_filled,
    columns=X.columns,
    index=X.index
)
```

---

#### Data Splitting

**Function:** `split_data(df, target_col, train_size, val_size, test_size, stratify=True)`

```python
from src.data.preprocessors import split_data

# Split with stratification
train_df, val_df, test_df = split_data(
    df,
    target_col='koi_disposition',
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    random_state=42,
    stratify=True
)

# Check class distribution
print("Train distribution:")
print(train_df['koi_disposition'].value_counts(normalize=True))
```

**Output:**

```
Split Statistics:
  Train: 3,600 samples (60.0%)
  Val:   1,200 samples (20.0%)
  Test:  1,200 samples (20.0%)

Target Distribution:
         Train    Val   Test
CANDIDATE   35%    35%    35%
CONFIRMED   52%    52%    52%
FALSE POS   13%    13%    13%
```

---

#### Save Splits

**Function:** `save_splits(train_df, val_df, test_df, output_dir='data/processed')`

```python
from src.data.preprocessors import save_splits

# Save to CSV
save_splits(
    train_df,
    val_df,
    test_df,
    output_dir='data/processed'
)

# Files created:
# data/processed/train.csv
# data/processed/val.csv
# data/processed/test.csv
```

---

#### Preprocess Data (All-in-One)

**Function:** `preprocess_data(df, handle_missing=True, split=True) -> dict`

Complete preprocessing pipeline:

```python
from src.data.preprocessors import preprocess_data

# Run full pipeline
result = preprocess_data(
    df,
    handle_missing=True,
    split=True
)

# Access results
train_df = result['train']
val_df = result['val']
test_df = result['test']
missing_handler = result['missing_handler']
missing_report = result['missing_report']
```

**Returns:**

```python
{
    'train': pd.DataFrame,
    'val': pd.DataFrame,
    'test': pd.DataFrame,
    'missing_handler': MissingValueHandler,
    'missing_report': pd.DataFrame
}
```

---

## Data Validation

### Validation Checklist

```python
def validate_data(df: pd.DataFrame) -> bool:
    """Comprehensive data validation."""

    checks = {
        'no_duplicates': len(df) == len(df.drop_duplicates()),
        'target_exists': 'koi_disposition' in df.columns,
        'no_missing_target': df['koi_disposition'].isnull().sum() == 0,
        'valid_classes': set(df['koi_disposition'].unique()).issubset(
            {'CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'}
        ),
        'numeric_features': len(df.select_dtypes(include=[np.number]).columns) > 0,
        'no_infinite': not np.isinf(df.select_dtypes(include=[np.number])).any().any(),
        'reasonable_size': 100 <= len(df) <= 1_000_000
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}: {passed}")

    return all(checks.values())
```

---

## Usage Examples

### Example 1: Basic Pipeline

```python
from src.data.cleaners import clean_data
from src.data.preprocessors import MissingValueHandler, split_data

# 1. Load data
df = pd.read_csv('data/raw/kepler_exoplanet_data.csv')

# 2. Clean
df_clean = clean_data(df, handle_outliers=True, method='clip')

# 3. Handle missing
handler = MissingValueHandler(threshold=0.7)
df_filled = handler.fit_transform(df_clean)

# 4. Split
train_df, val_df, test_df = split_data(df_filled, stratify=True)

print(f"✅ Pipeline complete!")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

---

### Example 2: Advanced Pipeline with Validation

```python
from src.data.cleaners import clean_data, validate_target_column
from src.data.preprocessors import (
    analyze_missing_values,
    MissingValueHandler,
    split_data,
    save_splits
)

# 1. Load
df = pd.read_csv('data/raw/kepler_exoplanet_data.csv')
print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

# 2. Validate
validate_target_column(df)

# 3. Analyze missing
missing_report = analyze_missing_values(df)
print("\nMissing Values:")
print(missing_report)

# 4. Clean
df_clean = clean_data(
    df,
    handle_outliers=True,
    method='clip',
    outlier_multiplier=2.0,  # Conservative
    optimize_dtypes=True
)

# 5. Handle missing
handler = MissingValueHandler(
    threshold=0.5,  # Drop >50% missing
    numerical_strategy='median',
    categorical_strategy='most_frequent'
)
df_filled = handler.fit_transform(df_clean)

# 6. Split
train_df, val_df, test_df = split_data(
    df_filled,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    stratify=True
)

# 7. Save
save_splits(train_df, val_df, test_df, output_dir='data/processed')

print("\n✅ Complete pipeline executed successfully!")
```

---

### Example 3: Custom Processing

```python
from src.data.cleaners import remove_duplicates, handle_outliers
from src.data.preprocessors import SimpleImputer
import pandas as pd
import numpy as np

# Custom pipeline
df = pd.read_csv('data/raw/kepler_exoplanet_data.csv')

# Step 1: Remove duplicates
df = remove_duplicates(df)

# Step 2: Handle outliers on specific columns
numeric_cols = ['koi_period', 'koi_prad', 'koi_teq']
df = handle_outliers(df, columns=numeric_cols, method='clip', multiplier=3.0)

# Step 3: Custom missing value handling
X = df.drop(columns=['koi_disposition'])
y = df['koi_disposition']

# Different strategies for different column groups
temporal_cols = ['koi_period', 'koi_duration']
physical_cols = ['koi_prad', 'koi_teq', 'koi_insol']
stellar_cols = ['koi_steff', 'koi_slogg', 'koi_srad']

# Temporal: median (robust to outliers)
imputer_temporal = SimpleImputer(strategy='median')
X[temporal_cols] = imputer_temporal.fit_transform(X[temporal_cols])

# Physical: mean (assume normal distribution)
imputer_physical = SimpleImputer(strategy='mean')
X[physical_cols] = imputer_physical.fit_transform(X[physical_cols])

# Stellar: median (skewed distribution)
imputer_stellar = SimpleImputer(strategy='median')
X[stellar_cols] = imputer_stellar.fit_transform(X[stellar_cols])

# Combine
df_processed = pd.concat([X, y], axis=1)

print(f"✅ Custom pipeline complete: {len(df_processed)} rows")
```

---

## Performance Metrics

### Benchmark Results

**Dataset:** Kepler Exoplanet Archive (6,000 rows, 50 columns)

| Operation      | Time      | Memory    |
| -------------- | --------- | --------- |
| Load CSV       | 0.15s     | 24 MB     |
| Clean Data     | 0.08s     | +2 MB     |
| Handle Missing | 0.12s     | +3 MB     |
| Split Data     | 0.05s     | +18 MB    |
| **Total**      | **0.40s** | **47 MB** |

**Optimization Tips:**

- Use `dtype` parameter in `pd.read_csv()` for faster loading
- Convert dtypes early with `convert_dtypes()`
- Use `chunksize` for large files (>1M rows)
- Consider Parquet format for faster I/O

---

## Best Practices

### ✅ DO

```python
# 1. Always validate data first
validate_target_column(df)

# 2. Analyze before imputing
missing_report = analyze_missing_values(df)

# 3. Fit on train, transform on all
handler.fit(train_df)
train_filled = handler.transform(train_df)
val_filled = handler.transform(val_df)

# 4. Use stratified split for imbalanced data
train, val, test = split_data(df, stratify=True)

# 5. Save intermediate results
save_splits(train, val, test)
```

### ❌ DON'T

```python
# 1. Don't fit on validation/test data
handler.fit(val_df)  # ❌ Data leakage!

# 2. Don't drop target in cleaning
df = df.dropna()  # ❌ May drop target rows!

# 3. Don't ignore class distribution
train, test = train_test_split(df)  # ❌ No stratification!

# 4. Don't use mean for skewed distributions
imputer = SimpleImputer(strategy='mean')  # ❌ Use median!
```

---

## Error Handling

```python
from src.core.exceptions import DataValidationError

try:
    # Validate
    validate_target_column(df)

    # Clean
    df_clean = clean_data(df)

    # Process
    df_processed = handler.fit_transform(df_clean)

except DataValidationError as e:
    print(f"❌ Validation error: {e}")
    # Handle error...

except ValueError as e:
    print(f"❌ Value error: {e}")
    # Handle error...

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    # Log and handle...
```

---

## Testing

All data pipeline functions have comprehensive tests:

```bash
# Run data tests
pytest tests/test_data/ -v

# Specific module
pytest tests/test_data/test_cleaners.py -v
pytest tests/test_data/test_preprocessors.py -v

# With coverage
pytest tests/test_data/ --cov=src.data --cov-report=html
```

**Test Coverage:**

- `cleaners.py`: 97.10%
- `preprocessors.py`: 81.44%
- Integration tests: 100%

---

## Troubleshooting

**Issue:** Memory error with large datasets

```python
# Solution: Use chunking
chunks = pd.read_csv('large_file.csv', chunksize=10000)
processed_chunks = []
for chunk in chunks:
    processed = clean_data(chunk)
    processed_chunks.append(processed)
df = pd.concat(processed_chunks)
```

**Issue:** Stratified split fails

```python
# Solution: Check class distribution
print(df['koi_disposition'].value_counts())
# If too imbalanced, increase dataset size or use different split ratio
```

**Issue:** Too many missing values

```python
# Solution: Adjust threshold or use different strategy
handler = MissingValueHandler(
    threshold=0.9,  # More lenient
    numerical_strategy='median'  # More robust
)
```

---

## References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

---

**Status:** ✅ Production Ready  
**Maintainer:** sulegogh  
**Last Updated:** 2025-11-13

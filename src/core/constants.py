"""
Proje genelinde kullanılan sabitler.

Bu modül, proje içinde kullanılan tüm sabit değerleri merkezi olarak
tanımlar. Değişiklik yapılması gerektiğinde tek bir yerden yapılabilir.
"""

from pathlib import Path
from typing import Dict, List

# ============================================
# PROJE YOLLARI
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = RESULTS_DIR / "logs"

# Data subdirectories
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"
DATA_SAMPLE = DATA_DIR / "sample"

# Model subdirectories
MODELS_EXPERIMENTS = MODELS_DIR / "experiments"
MODELS_PRODUCTION = MODELS_DIR / "production"
MODELS_REGISTRY = MODELS_DIR / "registry"

# Results subdirectories
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_REPORTS = RESULTS_DIR / "reports"

# ============================================
# NASA API
# ============================================
NASA_API_BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
NASA_TABLE_NAME = "cumulative"
NASA_OUTPUT_FORMAT = "csv"

# ============================================
# VERI ÖZELLİKLERİ
# ============================================
TARGET_COLUMN = "koi_disposition"

TARGET_VALUES: Dict[str, int] = {
    "CONFIRMED": 2,
    "CANDIDATE": 1,
    "FALSE POSITIVE": 0
}

# Kategorik özellikler
CATEGORICAL_FEATURES: List[str] = [
    "koi_disposition",
    "koi_pdisposition",
]

# Sayısal özellikler (core)
NUMERICAL_FEATURES: List[str] = [
    "koi_period",
    "koi_time0bk",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "ra",
    "dec",
]

# Kullanılmayacak sütunlar
DROP_COLUMNS: List[str] = [
    "rowid",
    "kepid",
    "kepler_name",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]

# ============================================
# MODEL AYARLARI
# ============================================
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Class dengeleme
APPLY_SMOTE = True
SMOTE_K_NEIGHBORS = 5

# Cross-validation
CV_FOLDS = 5
CV_SCORING = "f1_weighted"

# ============================================
# MODEL HİPERPARAMETRELERİ (Baseline)
# ============================================
CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "border_count": 128,
    "random_seed": RANDOM_STATE,
    "verbose": 100,
    "task_type": "CPU",  # RTX 3050 4GB için şimdilik CPU
    "early_stopping_rounds": 50,
}

LIGHTGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "random_state": RANDOM_STATE,
}

XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "tree_method": "hist",  # CPU için
}

# ============================================
# PERFORMANS METRİKLERİ
# ============================================
METRICS_TO_TRACK: List[str] = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_ovr",
    "confusion_matrix",
]

MIN_ACCURACY = 0.85
MIN_F1_SCORE = 0.80
MIN_ROC_AUC = 0.90

# ============================================
# LOGGING
# ============================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
LOG_FILE_NAME = "exoplanet_ml.log"
LOG_FILE_PATH = LOGS_DIR / LOG_FILE_NAME

# ============================================
# VERSİYONLAMA
# ============================================
PROJECT_VERSION = "0.1.0"
MODEL_VERSION = "1.0.0"
DATA_VERSION = "1.0.0"

# ============================================
# FEATURE ENGINEERING
# ============================================
POLY_DEGREE = 2
POLY_INTERACTION_ONLY = False
POLY_INCLUDE_BIAS = False

SCALER_TYPE = "robust"
FEATURE_SELECTION_METHOD = "tree_importance"
FEATURE_SELECTION_THRESHOLD = 0.01

# ============================================
# EXPLAINABİLİTY
# ============================================
SHAP_SAMPLE_SIZE = 500
SHAP_CHECK_ADDITIVITY = False
FEATURE_IMPORTANCE_TOP_N = 20

# ============================================
# VALIDATION
# ============================================
VALIDATE_DATA_TYPES = True
VALIDATE_MISSING_VALUES = True
VALIDATE_VALUE_RANGES = True
VALIDATE_DUPLICATES = True
MISSING_VALUE_THRESHOLD = 0.70
OUTLIER_IQR_MULTIPLIER = 1.5
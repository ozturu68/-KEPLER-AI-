"""
Core modülü - Proje çekirdeği.

Bu modül, projenin temel sabitlerini, exception'larını ve
utility fonksiyonlarını içerir.
"""

from src.core.constants import (
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    DATA_RAW,
    DATA_PROCESSED,
    MODELS_PRODUCTION,
    
    # NASA API
    NASA_API_BASE_URL,
    NASA_TABLE_NAME,
    NASA_OUTPUT_FORMAT,
    
    # Target
    TARGET_COLUMN,
    TARGET_VALUES,
    
    # Features
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    
    # Model parameters
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    RANDOM_STATE,
    
    # Validation
    MISSING_VALUE_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
    
    # Versions
    PROJECT_VERSION,
    MODEL_VERSION,
)

from src.core.exceptions import (
    # Base
    ExoplanetMLError,
    
    # Data
    DataError,
    DataNotFoundError,
    DataValidationError,
    DataDownloadError,
    EmptyDataError,
    MissingColumnsError,
    
    # Model
    ModelError,
    ModelNotFoundError,
    ModelTrainingError,
    ModelPredictionError,
    ModelLoadError,
    ModelSaveError,
    
    # Feature Engineering
    FeatureEngineeringError,
    FeatureSelectionError,
    ScalingError,
    
    # API
    APIError,
    InvalidRequestError,
    PredictionServiceError,
    
    # Config
    ConfigError,
    ConfigNotFoundError,
    InvalidConfigError,
    
    # Validation
    ValidationError,
    SchemaValidationError,
    RangeValidationError,
)

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "DATA_RAW",
    "DATA_PROCESSED",
    "MODELS_PRODUCTION",
    
    # NASA API
    "NASA_API_BASE_URL",
    "NASA_TABLE_NAME",
    "NASA_OUTPUT_FORMAT",
    
    # Target
    "TARGET_COLUMN",
    "TARGET_VALUES",
    
    # Features
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "DROP_COLUMNS",
    
    # Model parameters
    "TRAIN_SIZE",
    "VAL_SIZE",
    "TEST_SIZE",
    "RANDOM_STATE",
    
    # Validation
    "MISSING_VALUE_THRESHOLD",
    "OUTLIER_IQR_MULTIPLIER",
    
    # Versions
    "PROJECT_VERSION",
    "MODEL_VERSION",
    
    # Base exceptions
    "ExoplanetMLError",
    
    # Data exceptions
    "DataError",
    "DataNotFoundError",
    "DataValidationError",
    "DataDownloadError",
    "EmptyDataError",
    "MissingColumnsError",
    
    # Model exceptions
    "ModelError",
    "ModelNotFoundError",
    "ModelTrainingError",
    "ModelPredictionError",
    "ModelLoadError",
    "ModelSaveError",
    
    # Feature Engineering exceptions
    "FeatureEngineeringError",
    "FeatureSelectionError",
    "ScalingError",
    
    # API exceptions
    "APIError",
    "InvalidRequestError",
    "PredictionServiceError",
    
    # Config exceptions
    "ConfigError",
    "ConfigNotFoundError",
    "InvalidConfigError",
    
    # Validation exceptions
    "ValidationError",
    "SchemaValidationError",
    "RangeValidationError",
]

__version__ = PROJECT_VERSION
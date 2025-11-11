"""
Data modülü - Veri işleme ve hazırlama.

Bu modül, veri yükleme, temizleme, preprocessing ve
splitting işlemlerini içerir.
"""

from src.data.cleaners import (
    clean_data,
    convert_dtypes,
    detect_outliers_iqr,
    drop_unnecessary_columns,
    handle_outliers,
    remove_duplicates,
    validate_target_column,
)
from src.data.preprocessors import MissingValueHandler, analyze_missing_values, preprocess_data, save_splits, split_data

__all__ = [
    # Cleaners
    "remove_duplicates",
    "drop_unnecessary_columns",
    "detect_outliers_iqr",
    "handle_outliers",
    "convert_dtypes",
    "validate_target_column",
    "clean_data",
    # Preprocessors
    "MissingValueHandler",
    "analyze_missing_values",
    "split_data",
    "save_splits",
    "preprocess_data",
]

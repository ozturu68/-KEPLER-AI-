"""
Veri temizleme fonksiyonları.

Bu modül, ham veriyi temizlemek için kullanılan fonksiyonları içerir:
- Duplicate kayıtları kaldırma
- Outlier tespiti ve işleme
- Veri tipi dönüşümleri
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger

from src.core import (
    DROP_COLUMNS,
    TARGET_COLUMN,
    OUTLIER_IQR_MULTIPLIER,
    DataValidationError,
)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate (tekrar eden) kayıtları kaldır.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        pd.DataFrame: Temizlenmiş DataFrame
    """
    logger.info("Duplicate kayıtlar kontrol ediliyor...")
    
    initial_rows = len(df)
    df_clean = df.drop_duplicates()
    removed_rows = initial_rows - len(df_clean)
    
    if removed_rows > 0:
        logger.warning(f"{removed_rows} duplicate kayıt kaldırıldı")
    else:
        logger.info("Duplicate kayıt bulunamadı")
    
    return df_clean


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gereksiz sütunları kaldır.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        pd.DataFrame: Temizlenmiş DataFrame
    """
    logger.info("Gereksiz sütunlar kaldırılıyor...")
    
    # Mevcut olan gereksiz sütunları bul
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    
    if cols_to_drop:
        df_clean = df.drop(columns=cols_to_drop)
        logger.info(f"{len(cols_to_drop)} sütun kaldırıldı: {cols_to_drop}")
    else:
        df_clean = df.copy()
        logger.info("Kaldırılacak sütun bulunamadı")
    
    return df_clean


def detect_outliers_iqr(
    data: pd.Series,
    multiplier: float = OUTLIER_IQR_MULTIPLIER
) -> Tuple[pd.Series, int]:
    """
    IQR (Interquartile Range) yöntemi ile outlier tespit et.
    
    Args:
        data: Pandas Series
        multiplier: IQR çarpanı (default: 1.5)
        
    Returns:
        tuple: (outlier_mask, outlier_count)
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_count = outlier_mask.sum()
    
    return outlier_mask, outlier_count


def handle_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "clip",
    multiplier: float = OUTLIER_IQR_MULTIPLIER
) -> pd.DataFrame:
    """
    Outlier'ları işle.
    
    Args:
        df: Pandas DataFrame
        columns: İşlenecek sütunlar
        method: 'clip' (sınırla) veya 'remove' (kaldır)
        multiplier: IQR çarpanı
        
    Returns:
        pd.DataFrame: İşlenmiş DataFrame
    """
    logger.info(f"Outlier işleme başladı (method: {method})...")
    
    df_clean = df.copy()
    total_outliers = 0
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        data = df_clean[col].dropna()
        if len(data) == 0:
            continue
        
        outlier_mask, outlier_count = detect_outliers_iqr(data, multiplier)
        
        if outlier_count > 0:
            total_outliers += outlier_count
            outlier_pct = (outlier_count / len(data)) * 100
            
            if method == "clip":
                # Sınırla (capping)
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                logger.debug(f"{col}: {outlier_count} outlier clipped ({outlier_pct:.1f}%)")
                
            elif method == "remove":
                # Kaldır
                df_clean = df_clean[~outlier_mask]
                logger.debug(f"{col}: {outlier_count} outlier removed ({outlier_pct:.1f}%)")
    
    if method == "clip":
        logger.info(f"Toplam {total_outliers} outlier clipped")
    else:
        removed_rows = len(df) - len(df_clean)
        logger.info(f"{removed_rows} satır kaldırıldı (outlier içeren)")
    
    return df_clean


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri tiplerini optimize et (memory kullanımını azalt).
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        pd.DataFrame: Optimize edilmiş DataFrame
    """
    logger.info("Veri tipleri optimize ediliyor...")
    
    df_clean = df.copy()
    initial_memory = df_clean.memory_usage(deep=True).sum() / (1024 * 1024)
    
    # Float64 → Float32
    float_cols = df_clean.select_dtypes(include=['float64']).columns
    df_clean[float_cols] = df_clean[float_cols].astype('float32')
    
    # Int64 → Int32 (eğer değerler uygunsa)
    int_cols = df_clean.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df_clean[col].min() >= np.iinfo(np.int32).min and \
           df_clean[col].max() <= np.iinfo(np.int32).max:
            df_clean[col] = df_clean[col].astype('int32')
    
    # Object → Category (tekrar eden string'ler için)
    object_cols = df_clean.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df_clean[col].nunique() / len(df_clean) < 0.5:  # %50'den az unique
            df_clean[col] = df_clean[col].astype('category')
    
    final_memory = df_clean.memory_usage(deep=True).sum() / (1024 * 1024)
    memory_reduction = ((initial_memory - final_memory) / initial_memory) * 100
    
    logger.info(f"Memory kullanımı: {initial_memory:.2f}MB → {final_memory:.2f}MB "
                f"({memory_reduction:.1f}% azalma)")
    
    return df_clean


def validate_target_column(df: pd.DataFrame) -> None:
    """
    Target sütununu validate et.
    
    Args:
        df: Pandas DataFrame
        
    Raises:
        DataValidationError: Target sütunu geçersizse
    """
    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(f"Target sütunu bulunamadı: {TARGET_COLUMN}")
    
    # NaN kontrolü
    nan_count = df[TARGET_COLUMN].isnull().sum()
    if nan_count > 0:
        raise DataValidationError(
            f"Target sütununda {nan_count} NaN değer var - kaldırılmalı"
        )
    
    # Değer kontrolü
    unique_values = df[TARGET_COLUMN].unique()
    logger.info(f"Target sütunu değerleri: {unique_values}")


def clean_data(df: pd.DataFrame, handle_outliers_method: str = "clip") -> pd.DataFrame:
    """
    Veriyi temizle (ana fonksiyon).
    
    Args:
        df: Pandas DataFrame
        handle_outliers_method: Outlier işleme yöntemi
        
    Returns:
        pd.DataFrame: Temizlenmiş DataFrame
    """
    logger.info("="*60)
    logger.info("VERİ TEMİZLEME BAŞLADI")
    logger.info("="*60)
    
    # 1. Duplicate'leri kaldır
    df_clean = remove_duplicates(df)
    
    # 2. Gereksiz sütunları kaldır
    df_clean = drop_unnecessary_columns(df_clean)
    
    # 3. Target'ta NaN varsa kaldır
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean[TARGET_COLUMN].notna()]
    removed_rows = initial_rows - len(df_clean)
    if removed_rows > 0:
        logger.warning(f"{removed_rows} satır kaldırıldı (target NaN)")
    
    # 4. Target validate
    validate_target_column(df_clean)
    
    # 5. Veri tiplerini optimize et
    df_clean = convert_dtypes(df_clean)
    
    logger.info("="*60)
    logger.info("VERİ TEMİZLEME TAMAMLANDI")
    logger.info(f"Sonuç: {len(df_clean)} satır, {len(df_clean.columns)} sütun")
    logger.info("="*60)
    
    return df_clean
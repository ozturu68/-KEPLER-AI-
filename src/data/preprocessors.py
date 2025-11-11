"""
Veri ön işleme (preprocessing) fonksiyonları.

Bu modül, veriyi model eğitimine hazırlamak için kullanılan fonksiyonları içerir:
- Missing value imputation (eksik değerleri doldurma)
- Feature transformation
- Data splitting
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.core import (
    MISSING_VALUE_THRESHOLD,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    DataValidationError,
)


class MissingValueHandler:
    """
    Eksik değerleri işleyen sınıf.

    Attributes:
        threshold: Sütunu drop etmek için missing value eşiği
        numerical_strategy: Sayısal sütunlar için strateji ('mean', 'median', 'most_frequent')
        categorical_strategy: Kategorik sütunlar için strateji ('most_frequent', 'constant')
    """

    def __init__(
        self,
        threshold: float = MISSING_VALUE_THRESHOLD,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
    ):
        self.threshold = threshold
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.dropped_columns = []

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        """
        Handler'ı fit et (imputer'ları eğit).

        Args:
            df: Pandas DataFrame

        Returns:
            self
        """
        logger.info("Missing value handler fit ediliyor...")

        # Yüksek missing value'lu sütunları tespit et
        missing_pct = df.isnull().sum() / len(df)
        self.dropped_columns = missing_pct[missing_pct >= self.threshold].index.tolist()

        if self.dropped_columns:
            logger.warning(f"{len(self.dropped_columns)} sütun drop edilecek (>%{self.threshold*100} missing)")
            for col in self.dropped_columns:
                logger.debug(f"  - {col}: %{missing_pct[col]*100:.1f} missing")

        # Kalan sütunlar
        df_remaining = df.drop(columns=self.dropped_columns, errors="ignore")

        # Numerical ve categorical sütunları ayır
        numerical_cols = df_remaining.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_remaining.select_dtypes(include=["object", "category"]).columns.tolist()

        # Target sütununu çıkar
        if TARGET_COLUMN in numerical_cols:
            numerical_cols.remove(TARGET_COLUMN)
        if TARGET_COLUMN in categorical_cols:
            categorical_cols.remove(TARGET_COLUMN)

        # Numerical imputer
        if numerical_cols:
            self.numerical_imputer = SimpleImputer(strategy=self.numerical_strategy)
            self.numerical_imputer.fit(df_remaining[numerical_cols])
            logger.info(
                f"Numerical imputer fit edildi: {len(numerical_cols)} sütun, strategy={self.numerical_strategy}"
            )

        # Categorical imputer
        if categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy=self.categorical_strategy)
            self.categorical_imputer.fit(df_remaining[categorical_cols])
            logger.info(
                f"Categorical imputer fit edildi: {len(categorical_cols)} sütun, strategy={self.categorical_strategy}"
            )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame'i transform et (missing value'ları doldur).

        Args:
            df: Pandas DataFrame

        Returns:
            pd.DataFrame: Transform edilmiş DataFrame
        """
        logger.info("Missing value'lar doldruluyor...")

        df_transformed = df.copy()

        # Yüksek missing value'lu sütunları drop et
        if self.dropped_columns:
            df_transformed = df_transformed.drop(columns=self.dropped_columns, errors="ignore")

        # Numerical sütunları impute et
        if self.numerical_imputer is not None:
            numerical_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
            if TARGET_COLUMN in numerical_cols:
                numerical_cols.remove(TARGET_COLUMN)

            if numerical_cols:
                df_transformed[numerical_cols] = self.numerical_imputer.transform(df_transformed[numerical_cols])
                logger.debug(f"{len(numerical_cols)} sayısal sütun impute edildi")

        # Categorical sütunları impute et
        if self.categorical_imputer is not None:
            categorical_cols = df_transformed.select_dtypes(include=["object", "category"]).columns.tolist()
            if TARGET_COLUMN in categorical_cols:
                categorical_cols.remove(TARGET_COLUMN)

            if categorical_cols:
                df_transformed[categorical_cols] = self.categorical_imputer.transform(df_transformed[categorical_cols])
                logger.debug(f"{len(categorical_cols)} kategorik sütun impute edildi")

        # Final kontrol
        remaining_missing = df_transformed.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"⚠️  Hala {remaining_missing} missing value var!")
        else:
            logger.info("✅ Tüm missing value'lar dolduruldu")

        return df_transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit ve transform'u birlikte yap.

        Args:
            df: Pandas DataFrame

        Returns:
            pd.DataFrame: Transform edilmiş DataFrame
        """
        return self.fit(df).transform(df)


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing value'ları analiz et ve rapor oluştur.

    Args:
        df: Pandas DataFrame

    Returns:
        pd.DataFrame: Missing value analiz raporu
    """
    logger.info("Missing value analizi yapılıyor...")

    missing_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isnull().sum(),
            "missing_pct": (df.isnull().sum() / len(df)) * 100,
            "dtype": df.dtypes,
        }
    )

    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_pct", ascending=False)

    total_missing = df.isnull().sum().sum()
    total_cells = np.product(df.shape)
    total_pct = (total_missing / total_cells) * 100

    logger.info(f"Toplam missing: {total_missing:,} / {total_cells:,} ({total_pct:.2f}%)")
    logger.info(f"Missing value'lu sütun sayısı: {len(missing_df)}")

    if len(missing_df) > 0:
        logger.info(
            f"En çok missing olan sütun: {missing_df.iloc[0]['column']} (%{missing_df.iloc[0]['missing_pct']:.1f})"
        )

    return missing_df


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Veriyi train/val/test olarak böl.

    Args:
        df: Pandas DataFrame
        target_col: Target sütunu
        train_size: Train oranı
        val_size: Validation oranı
        test_size: Test oranı
        random_state: Random seed
        stratify: Stratified split (class balance korusun)

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info("Veri bölünüyor (train/val/test)...")

    # Oranları kontrol et
    assert (
        abs((train_size + val_size + test_size) - 1.0) < 1e-6
    ), f"Train+Val+Test oranları 1.0 olmalı, bulundu: {train_size + val_size + test_size}"

    # Stratify için target
    stratify_col = df[target_col] if stratify else None

    # İlk split: train + (val+test)
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state, stratify=stratify_col)

    # İkinci split: val + test
    val_ratio = val_size / (val_size + test_size)
    stratify_col_temp = temp_df[target_col] if stratify else None

    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio, random_state=random_state, stratify=stratify_col_temp
    )

    logger.info(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

    # Target dağılımını kontrol et
    if stratify:
        logger.info("\nTarget dağılımı:")
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            dist = split_df[target_col].value_counts(normalize=True) * 100
            logger.info(f"  {split_name}: " + ", ".join([f"{k}={v:.1f}%" for k, v in dist.items()]))

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "data/processed"
) -> None:
    """
    Split'leri dosyaya kaydet.

    Args:
        train_df: Train DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Çıktı dizini
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Split'ler kaydediliyor: {output_dir}")

    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    logger.info(f"✅ Kaydedildi:")
    logger.info(f"  - train.csv: {len(train_df):,} satır")
    logger.info(f"  - val.csv: {len(val_df):,} satır")
    logger.info(f"  - test.csv: {len(test_df):,} satır")


def preprocess_data(df: pd.DataFrame, handle_missing: bool = True, split: bool = True) -> dict:
    """
    Veriyi preprocess et (ana fonksiyon).

    Args:
        df: Pandas DataFrame
        handle_missing: Missing value'ları işle
        split: Veriyi train/val/test'e böl

    Returns:
        dict: Preprocessed data ve metadata
    """
    logger.info("=" * 60)
    logger.info("VERİ PREPROCESSING BAŞLADI")
    logger.info("=" * 60)

    result = {}

    # Missing value analizi
    missing_report = analyze_missing_values(df)
    result["missing_report"] = missing_report

    # Missing value handling
    if handle_missing:
        handler = MissingValueHandler()
        df_processed = handler.fit_transform(df)
        result["missing_handler"] = handler
    else:
        df_processed = df.copy()
        result["missing_handler"] = None

    # Data splitting
    if split:
        train_df, val_df, test_df = split_data(df_processed)
        result["train"] = train_df
        result["val"] = val_df
        result["test"] = test_df
    else:
        result["data"] = df_processed

    logger.info("=" * 60)
    logger.info("VERİ PREPROCESSING TAMAMLANDI")
    logger.info("=" * 60)

    return result

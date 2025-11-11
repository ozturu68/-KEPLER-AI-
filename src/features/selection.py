"""
Feature selection fonksiyonları.

Bu modül, en iyi feature'ları seçmek için kullanılan fonksiyonları içerir:
- Correlation-based selection
- Variance threshold
- Tree-based importance
- Mutual information

Author: sulegogh
Date: 2025-11-11
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from src.core import RANDOM_STATE, TARGET_COLUMN, FeatureSelectionError


class FeatureSelector:
    """
    Feature selection sınıfı.

    Pipeline:
    1. Low variance features → DROP
    2. High correlation features → DROP
    3. Random Forest importance → RANK
    4. Top N features → SELECT

    Attributes:
        correlation_threshold: Correlation eşiği (default: 0.95)
        variance_threshold: Variance eşiği (default: 0.01)
        selected_features: Seçilen feature'ların listesi
        dropped_features: Kaldırılan feature'ların dict'i
    """

    def __init__(
        self, correlation_threshold: float = 0.95, variance_threshold: float = 0.01, importance_threshold: float = 0.001
    ):
        """
        FeatureSelector başlat.

        Args:
            correlation_threshold: Korelasyon eşiği (>0.95 drop edilir)
            variance_threshold: Varyans eşiği (<0.01 drop edilir)
            importance_threshold: Importance eşiği (kullanılmıyor şu an)
        """
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.importance_threshold = importance_threshold
        self.selected_features = None
        self.dropped_features = {}

        logger.info(
            f"FeatureSelector oluşturuldu: "
            f"corr_threshold={correlation_threshold}, "
            f"var_threshold={variance_threshold}"
        )

    def remove_low_variance_features(self, df: pd.DataFrame, exclude_cols: list[str] | None = None) -> list[str]:
        """
        Düşük varyans'lı feature'ları tespit et ve kaldır.

        Args:
            df: Pandas DataFrame
            exclude_cols: Kontrol dışı bırakılacak sütunlar (örn: target)

        Returns:
            list: Kaldırılan feature'ların listesi
        """
        logger.info(f"Düşük varyans feature'lar kontrol ediliyor (threshold={self.variance_threshold})...")

        if exclude_cols is None:
            exclude_cols = [TARGET_COLUMN]

        # Sadece numerical sütunlar
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]

        # Varyans hesapla
        variances = df[feature_cols].var()

        # Düşük varyans feature'ları bul
        low_variance_features = variances[variances < self.variance_threshold].index.tolist()

        if low_variance_features:
            logger.info(f"  ✓ {len(low_variance_features)} düşük varyans feature bulundu")
            for feat in low_variance_features[:5]:
                logger.debug(f"    - {feat}: var={variances[feat]:.6f}")
        else:
            logger.info("  ✓ Düşük varyans feature bulunamadı")

        self.dropped_features["low_variance"] = low_variance_features
        return low_variance_features

    def remove_high_correlation_features(self, df: pd.DataFrame, exclude_cols: list[str] | None = None) -> list[str]:
        """
        Yüksek korelasyonlu feature'ları tespit et ve kaldır.

        Mantık: İki feature arasında >0.95 korelasyon varsa birini drop et.

        Args:
            df: Pandas DataFrame
            exclude_cols: Kontrol dışı bırakılacak sütunlar

        Returns:
            list: Kaldırılan feature'ların listesi
        """
        logger.info(f"Yüksek korelasyon feature'lar kontrol ediliyor (threshold={self.correlation_threshold})...")

        if exclude_cols is None:
            exclude_cols = [TARGET_COLUMN]

        # Sadece numerical sütunlar
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]

        # Korelasyon matrisi
        corr_matrix = df[feature_cols].corr().abs()

        # Upper triangle (diagonal ve alt üçgeni ignore et)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Yüksek korelasyonlu feature'ları bul
        high_corr_features = [
            column for column in upper_triangle.columns if any(upper_triangle[column] > self.correlation_threshold)
        ]

        if high_corr_features:
            logger.info(f"  ✓ {len(high_corr_features)} yüksek korelasyon feature bulundu")
            for feat in high_corr_features[:5]:
                correlated_with = upper_triangle[feat][upper_triangle[feat] > self.correlation_threshold].index.tolist()
                if correlated_with:
                    logger.debug(
                        f"    - {feat} → {correlated_with[0]}: " f"{upper_triangle[feat][correlated_with[0]]:.3f}"
                    )
        else:
            logger.info("  ✓ Yüksek korelasyon feature bulunamadı")

        self.dropped_features["high_correlation"] = high_corr_features
        return high_corr_features

    def get_feature_importance(
        self, X: pd.DataFrame, y: pd.Series, method: str = "random_forest", n_estimators: int = 100
    ) -> pd.DataFrame:
        """
        Feature importance hesapla (tree-based veya mutual information).

        Args:
            X: Features DataFrame
            y: Target Series
            method: 'random_forest' veya 'mutual_info'
            n_estimators: Tree sayısı (RF için)

        Returns:
            pd.DataFrame: feature, importance columns ile DataFrame
        """
        logger.info(f"Feature importance hesaplanıyor (method={method})...")

        # Sadece numerical sütunlar
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_cols].copy()

        if len(X_numerical.columns) == 0:
            raise FeatureSelectionError("Numerical feature bulunamadı!")

        logger.debug(f"  Numerical features: {len(X_numerical.columns)}/{len(X.columns)}")

        # INF ve NaN temizleme
        logger.info("  INF ve NaN değerler temizleniyor...")
        X_numerical = X_numerical.replace([np.inf, -np.inf], np.nan)

        nan_count = X_numerical.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"    ⚠️  {nan_count} NaN/INF değer bulundu, median ile doldruluyor...")

            # Sütun sütun doldur (daha güvenli)
            for col in X_numerical.columns:
                if X_numerical[col].isnull().any():
                    median_val = X_numerical[col].median()
                    fill_value = 0 if pd.isna(median_val) else median_val
                    X_numerical.loc[:, col] = X_numerical[col].fillna(fill_value)

        # Final kontrol
        remaining_nan = X_numerical.isnull().sum().sum()
        remaining_inf = np.isinf(X_numerical.values).sum()

        if remaining_nan > 0 or remaining_inf > 0:
            raise FeatureSelectionError(f"Temizleme başarısız! NaN: {remaining_nan}, INF: {remaining_inf}")

        logger.info("  ✓ Temizleme tamamlandı")

        # Feature importance hesapla
        if method == "random_forest":
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
            )
            rf.fit(X_numerical, y)
            importances = rf.feature_importances_

        elif method == "mutual_info":
            importances = mutual_info_classif(X_numerical, y, random_state=RANDOM_STATE)

        else:
            raise FeatureSelectionError(f"Geçersiz method: {method}")

        # DataFrame'e dönüştür ve sırala
        importance_df = (
            pd.DataFrame({"feature": X_numerical.columns, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        logger.info("  ✓ Feature importance hesaplandı")
        logger.info("    Top 5 features:")
        for idx, row in importance_df.head(5).iterrows():
            logger.info(f"      {row['feature']}: {row['importance']:.4f}")

        return importance_df

    def select_top_features(
        self,
        importance_df: pd.DataFrame,
        n_features: int | None = None,
        importance_threshold: float | None = None,
    ) -> list[str]:
        """
        En iyi feature'ları seç (importance'a göre).

        Args:
            importance_df: Feature importance DataFrame
            n_features: Seçilecek feature sayısı
            importance_threshold: Minimum importance değeri

        Returns:
            list: Seçilen feature'ların listesi
        """
        if n_features is not None:
            selected = importance_df.head(n_features)["feature"].tolist()
            logger.info(f"  ✓ Top {n_features} feature seçildi")

        elif importance_threshold is not None:
            selected = importance_df[importance_df["importance"] >= importance_threshold]["feature"].tolist()
            logger.info(f"  ✓ {len(selected)} feature seçildi " f"(importance >= {importance_threshold})")

        else:
            raise FeatureSelectionError("n_features veya importance_threshold belirtilmeli!")

        return selected

    def select_features(
        self, df: pd.DataFrame, target_col: str = TARGET_COLUMN, method: str = "auto", n_features: int = 50
    ) -> tuple[list[str], dict]:
        """
        Feature selection pipeline'ı çalıştır.

        Pipeline:
        1. Sadece numerical feature'ları al
        2. Düşük varyans feature'ları drop et
        3. Yüksek korelasyon feature'ları drop et
        4. Random Forest ile importance hesapla
        5. Top N feature'ı seç

        Args:
            df: Pandas DataFrame
            target_col: Target sütunu adı
            method: Selection yöntemi (şu an sadece 'auto')
            n_features: Seçilecek feature sayısı

        Returns:
            tuple: (selected_features: List[str], selection_info: Dict)
        """
        logger.info("=" * 60)
        logger.info("FEATURE SELECTION")
        logger.info("=" * 60)
        logger.info(f"Method: {method}, Target features: {n_features}")

        # Target kontrolü
        if target_col not in df.columns:
            raise FeatureSelectionError(f"Target sütunu bulunamadı: {target_col}")

        # Sadece numerical + target
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_keep = [col for col in numerical_cols if col != target_col] + [target_col]
        df_numerical = df[cols_to_keep].copy()

        initial_features = len(df_numerical.columns) - 1  # Target hariç
        logger.info(f"Numerical features: {initial_features}")

        # STEP 1: Düşük varyans
        low_var_features = self.remove_low_variance_features(df_numerical, exclude_cols=[target_col])
        df_filtered = df_numerical.drop(columns=low_var_features)

        # STEP 2: Yüksek korelasyon
        high_corr_features = self.remove_high_correlation_features(df_filtered, exclude_cols=[target_col])
        df_filtered = df_filtered.drop(columns=high_corr_features)

        after_filtering = len(df_filtered.columns) - 1
        logger.info(f"\n✓ Filtreleme sonrası: {initial_features} → {after_filtering} features")

        # STEP 3: Feature importance
        X = df_filtered.drop(columns=[target_col])
        y = df_filtered[target_col]

        importance_df = self.get_feature_importance(X, y, method="random_forest")

        # STEP 4: Top N seç
        if after_filtering > n_features:
            selected_features = self.select_top_features(importance_df, n_features=n_features)
        else:
            selected_features = X.columns.tolist()
            logger.info(f"  ✓ Tüm feature'lar seçildi ({len(selected_features)} < {n_features})")

        self.selected_features = selected_features

        # Özet bilgi
        selection_info = {
            "initial_features": initial_features,
            "after_filtering": after_filtering,
            "selected_features": len(selected_features),
            "dropped_low_variance": len(low_var_features),
            "dropped_high_correlation": len(high_corr_features),
            "dropped_low_importance": after_filtering - len(selected_features),
            "importance_df": importance_df,
        }

        logger.info("=" * 60)
        logger.info("FEATURE SELECTION TAMAMLANDI")
        logger.info(f"  {initial_features} → {len(selected_features)} features")
        logger.info(f"  Dropped: {initial_features - len(selected_features)}")
        logger.info("=" * 60)

        return selected_features, selection_info

    def transform(self, df: pd.DataFrame, keep_target: bool = True) -> pd.DataFrame:
        """
        DataFrame'i transform et (sadece seçilen feature'ları tut).

        Args:
            df: Pandas DataFrame
            keep_target: Target sütununu da tut

        Returns:
            pd.DataFrame: Transform edilmiş DataFrame
        """
        if self.selected_features is None:
            raise FeatureSelectionError("Önce select_features() çağrılmalı!")

        cols_to_keep = self.selected_features.copy()

        if keep_target and TARGET_COLUMN in df.columns:
            cols_to_keep.append(TARGET_COLUMN)

        # Eksik sütun kontrolü
        missing_cols = [col for col in cols_to_keep if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️  {len(missing_cols)} sütun DataFrame'de bulunamadı, atlanıyor: " f"{missing_cols[:3]}")
            cols_to_keep = [col for col in cols_to_keep if col in df.columns]

        return df[cols_to_keep].copy()


def select_features_train_val_test(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, n_features: int = 50, method: str = "auto"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureSelector, dict]:
    """
    Train/val/test için feature selection uygula.

    ÖNEMLİ: Selection SADECE train'de yapılır!
           Val ve test'e sadece transform uygulanır (data leakage önleme).

    Args:
        train_df: Train DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        n_features: Seçilecek feature sayısı
        method: Selection yöntemi

    Returns:
        tuple: (train_selected, val_selected, test_selected, selector, info)
    """
    logger.info("=" * 60)
    logger.info("TRAIN/VAL/TEST FEATURE SELECTION")
    logger.info("=" * 60)

    # Selector oluştur ve SADECE train'de fit et
    selector = FeatureSelector()
    selected_features, selection_info = selector.select_features(train_df, method=method, n_features=n_features)

    # Transform (train, val, test)
    train_selected = selector.transform(train_df, keep_target=True)
    val_selected = selector.transform(val_df, keep_target=True)
    test_selected = selector.transform(test_df, keep_target=True)

    logger.info("=" * 60)
    logger.info("TRANSFORM TAMAMLANDI")
    logger.info(f"  Train: {len(train_selected):,} satır, {len(train_selected.columns)} sütun")
    logger.info(f"  Val:   {len(val_selected):,} satır, {len(val_selected.columns)} sütun")
    logger.info(f"  Test:  {len(test_selected):,} satır, {len(test_selected.columns)} sütun")
    logger.info("=" * 60)

    return train_selected, val_selected, test_selected, selector, selection_info

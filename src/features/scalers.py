"""
Feature scaling fonksiyonları.

Bu modül, feature'ları ölçeklendirmek için kullanılan fonksiyonları içerir:
- StandardScaler
- RobustScaler (outlier'lar için önerilir)
- MinMaxScaler
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from loguru import logger

from src.core import TARGET_COLUMN, ScalingError


class FeatureScaler:
    """
    Feature scaling sınıfı.
    
    Attributes:
        method: Scaling yöntemi ('standard', 'robust', 'minmax')
        scaler: Sklearn scaler objesi
        numerical_features: Ölçeklenecek sayısal sütunlar
    """
    
    def __init__(self, method: str = "robust"):
        """
        FeatureScaler başlat.
        
        Args:
            method: 'standard', 'robust' veya 'minmax'
        """
        if method not in ['standard', 'robust', 'minmax']:
            raise ScalingError(f"Geçersiz scaling yöntemi: {method}")
        
        self.method = method
        self.scaler = None
        self.numerical_features = None
        
        # Scaler oluştur
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        
        logger.info(f"FeatureScaler oluşturuldu: method={method}")
    
    def fit(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> 'FeatureScaler':
        """
        Scaler'ı fit et.
        
        Args:
            df: Pandas DataFrame
            exclude_cols: Hariç tutulacak sütunlar (ör: target, ID)
            
        Returns:
            self
        """
        logger.info("Feature scaler fit ediliyor...")
        
        # Exclude cols default
        if exclude_cols is None:
            exclude_cols = [TARGET_COLUMN]
        
        # Sayısal sütunları bul
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude cols'u çıkar
        self.numerical_features = [col for col in numerical_cols if col not in exclude_cols]
        
        if len(self.numerical_features) == 0:
            raise ScalingError("Ölçeklenecek sayısal feature bulunamadı!")
        
        # Fit
        self.scaler.fit(df[self.numerical_features])
        
        logger.info(f"✅ Scaler fit edildi: {len(self.numerical_features)} feature, method={self.method}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame'i transform et (scale et).
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            pd.DataFrame: Scale edilmiş DataFrame
        """
        if self.scaler is None or self.numerical_features is None:
            raise ScalingError("Scaler fit edilmemiş! Önce fit() çağırın.")
        
        logger.info(f"Feature'lar scale ediliyor (method={self.method})...")
        
        df_scaled = df.copy()
        
        # Scale et
        df_scaled[self.numerical_features] = self.scaler.transform(df[self.numerical_features])
        
        # İstatistikler
        logger.debug(f"{len(self.numerical_features)} feature scale edildi")
        
        # Örnek: İlk feature'ın yeni değer aralığı
        if len(self.numerical_features) > 0:
            sample_col = self.numerical_features[0]
            min_val = df_scaled[sample_col].min()
            max_val = df_scaled[sample_col].max()
            mean_val = df_scaled[sample_col].mean()
            logger.debug(f"Örnek ({sample_col}): min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}")
        
        logger.info("✅ Feature scaling tamamlandı")
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit ve transform'u birlikte yap.
        
        Args:
            df: Pandas DataFrame
            exclude_cols: Hariç tutulacak sütunlar
            
        Returns:
            pd.DataFrame: Scale edilmiş DataFrame
        """
        return self.fit(df, exclude_cols).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale'i geri al (orijinal değerlere dön).
        
        Args:
            df: Scale edilmiş DataFrame
            
        Returns:
            pd.DataFrame: Orijinal scale'e dönmüş DataFrame
        """
        if self.scaler is None or self.numerical_features is None:
            raise ScalingError("Scaler fit edilmemiş!")
        
        logger.info("Scale geri alınıyor...")
        
        df_inversed = df.copy()
        df_inversed[self.numerical_features] = self.scaler.inverse_transform(df[self.numerical_features])
        
        logger.info("✅ Inverse transform tamamlandı")
        
        return df_inversed
    
    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature'ların scale istatistiklerini al.
        
        Args:
            df: DataFrame (scale öncesi veya sonrası)
            
        Returns:
            pd.DataFrame: İstatistik tablosu
        """
        if self.numerical_features is None:
            raise ScalingError("Scaler fit edilmemiş!")
        
        stats = pd.DataFrame({
            'feature': self.numerical_features,
            'min': df[self.numerical_features].min().values,
            'max': df[self.numerical_features].max().values,
            'mean': df[self.numerical_features].mean().values,
            'std': df[self.numerical_features].std().values,
            'median': df[self.numerical_features].median().values,
        })
        
        return stats


def compare_scalers(df: pd.DataFrame, sample_features: Optional[List[str]] = None) -> Dict:
    """
    Farklı scaler'ları karşılaştır.
    
    Args:
        df: Pandas DataFrame
        sample_features: Karşılaştırılacak örnek feature'lar (None ise ilk 5)
        
    Returns:
        dict: Her scaler için sonuçlar
    """
    logger.info("Scaler'lar karşılaştırılıyor...")
    
    # Sayısal feature'ları al
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numerical_cols:
        numerical_cols.remove(TARGET_COLUMN)
    
    # Sample features
    if sample_features is None:
        sample_features = numerical_cols[:5]
    
    results = {}
    
    for method in ['standard', 'robust', 'minmax']:
        logger.info(f"  Testing: {method}")
        
        scaler = FeatureScaler(method=method)
        df_scaled = scaler.fit_transform(df)
        
        # İstatistikler
        stats = {
            'method': method,
            'features': {}
        }
        
        for feat in sample_features:
            if feat in df_scaled.columns:
                stats['features'][feat] = {
                    'original': {
                        'min': float(df[feat].min()),
                        'max': float(df[feat].max()),
                        'mean': float(df[feat].mean()),
                        'std': float(df[feat].std()),
                    },
                    'scaled': {
                        'min': float(df_scaled[feat].min()),
                        'max': float(df_scaled[feat].max()),
                        'mean': float(df_scaled[feat].mean()),
                        'std': float(df_scaled[feat].std()),
                    }
                }
        
        results[method] = stats
    
    logger.info("✅ Scaler karşılaştırması tamamlandı")
    
    return results


def scale_train_val_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str = "robust",
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureScaler]:
    """
    Train/val/test'i scale et.
    
    ÖNEMLİ: Scaler sadece train'de fit edilir, val/test'e transform uygulanır!
    
    Args:
        train_df: Train DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        method: Scaling yöntemi
        exclude_cols: Hariç tutulacak sütunlar
        
    Returns:
        tuple: (train_scaled, val_scaled, test_scaled, scaler)
    """
    logger.info("="*60)
    logger.info("TRAIN/VAL/TEST SCALING")
    logger.info("="*60)
    
    # Scaler oluştur ve train'de fit et
    scaler = FeatureScaler(method=method)
    train_scaled = scaler.fit_transform(train_df, exclude_cols=exclude_cols)
    
    # Val ve test'e transform uygula (fit ETME!)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
    logger.info("="*60)
    logger.info("SCALING TAMAMLANDI")
    logger.info(f"  Train: {len(train_scaled)} satır")
    logger.info(f"  Val:   {len(val_scaled)} satır")
    logger.info(f"  Test:  {len(test_scaled)} satır")
    logger.info(f"  Method: {method}")
    logger.info(f"  Scaled features: {len(scaler.numerical_features)}")
    logger.info("="*60)
    
    return train_scaled, val_scaled, test_scaled, scaler
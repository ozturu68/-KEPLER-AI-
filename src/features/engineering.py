"""
Feature engineering fonksiyonları.

Bu modül, yeni feature'lar oluşturmak için kullanılan fonksiyonları içerir:
- Polynomial features
- Interaction features
- Domain-specific features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures
from loguru import logger

from src.core import FeatureEngineeringError


class ExoplanetFeatureEngineer:
    """
    Exoplanet-specific feature engineering sınıfı.
    
    Attributes:
        poly_degree: Polynomial feature degree'si
        interaction_only: Sadece interaction features oluştur (x1*x2, x1*x3 vs)
        include_bias: Bias terimi ekle (1.0)
    """
    
    def __init__(
        self,
        poly_degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False
    ):
        self.poly_degree = poly_degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly_transformer = None
        self.poly_feature_names = None
        
        logger.info(f"ExoplanetFeatureEngineer oluşturuldu: "
                   f"degree={poly_degree}, interaction_only={interaction_only}")
    
    def create_planetary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gezegen fiziği tabanlı yeni feature'lar oluştur.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            pd.DataFrame: Yeni feature'lar eklenmiş DataFrame
        """
        logger.info("Planetary features oluşturuluyor...")
        
        df_new = df.copy()
        created_features = []
        
        # 1. Planet-Star Density Ratio (Gezegen/Yıldız yoğunluk oranı)
        if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
            # Yarıçap oranının küpü ≈ hacim oranı
            df_new['planet_star_volume_ratio'] = (df['koi_prad'] / df['koi_srad']) ** 3
            created_features.append('planet_star_volume_ratio')
        
        # 2. Orbital Velocity (Yörünge hızı, v ∝ sqrt(a/P))
        if 'koi_sma' in df.columns and 'koi_period' in df.columns:
            # Basitleştirilmiş yörünge hızı (AU/day)
            df_new['orbital_velocity'] = np.sqrt(df['koi_sma'] / (df['koi_period'] + 1e-10))
            created_features.append('orbital_velocity')
        
        # 3. Stellar Flux (Yıldız ışık akısı, F ∝ R²T⁴/a²)
        if all(col in df.columns for col in ['koi_srad', 'koi_steff', 'koi_sma']):
            df_new['stellar_flux'] = (df['koi_srad'] ** 2 * df['koi_steff'] ** 4) / (df['koi_sma'] ** 2 + 1e-10)
            created_features.append('stellar_flux')
        
        # 4. Planet Equilibrium Temperature Check (Teq karşılaştırma)
        if 'koi_teq' in df.columns and 'koi_steff' in df.columns:
            # Expected Teq ≈ Tstar * sqrt(Rstar / 2a)
            if 'koi_srad' in df.columns and 'koi_sma' in df.columns:
                expected_teq = df['koi_steff'] * np.sqrt(df['koi_srad'] / (2 * df['koi_sma'] + 1e-10))
                df_new['teq_deviation'] = np.abs(df['koi_teq'] - expected_teq) / (expected_teq + 1)
                created_features.append('teq_deviation')
        
        # 5. Transit Signal Strength (Geçiş sinyal gücü, depth * duration)
        if 'koi_depth' in df.columns and 'koi_duration' in df.columns:
            df_new['transit_signal_strength'] = df['koi_depth'] * df['koi_duration']
            created_features.append('transit_signal_strength')
        
        # 6. Planet Radius Category (Gezegen boyut kategorisi)
        if 'koi_prad' in df.columns:
            # Earth radius cinsinden
            # < 1.5 Re: Rocky, 1.5-4 Re: Sub-Neptune, 4-10 Re: Neptune, > 10 Re: Jupiter
            df_new['is_earth_size'] = (df['koi_prad'] < 1.5).astype(float)
            df_new['is_sub_neptune'] = ((df['koi_prad'] >= 1.5) & (df['koi_prad'] < 4)).astype(float)
            df_new['is_neptune_size'] = ((df['koi_prad'] >= 4) & (df['koi_prad'] < 10)).astype(float)
            df_new['is_jupiter_size'] = (df['koi_prad'] >= 10).astype(float)
            created_features.extend(['is_earth_size', 'is_sub_neptune', 'is_neptune_size', 'is_jupiter_size'])
        
        # 7. Habitable Zone Indicator (Yaşanabilir bölge göstergesi)
        if 'koi_sma' in df.columns and 'koi_steff' in df.columns:
            # Basitleştirilmiş habitable zone: 0.95 * sqrt(L) < a < 1.37 * sqrt(L)
            # L ≈ (Tstar/Tsun)^4, Tsun ≈ 5778 K
            luminosity = (df['koi_steff'] / 5778) ** 4
            hz_inner = 0.95 * np.sqrt(luminosity)
            hz_outer = 1.37 * np.sqrt(luminosity)
            df_new['in_habitable_zone'] = ((df['koi_sma'] >= hz_inner) & (df['koi_sma'] <= hz_outer)).astype(float)
            created_features.append('in_habitable_zone')
        
        # 8. Impact Parameter Quality (Merkez geçiş kalitesi)
        if 'koi_impact' in df.columns:
            # b < 0.5: Central transit (good), b > 0.9: Grazing (bad)
            df_new['is_central_transit'] = (df['koi_impact'] < 0.5).astype(float)
            df_new['is_grazing_transit'] = (df['koi_impact'] > 0.9).astype(float)
            created_features.extend(['is_central_transit', 'is_grazing_transit'])
        
        # 9. SNR to Transit Count Ratio (SNR verimliliği)
        if 'koi_model_snr' in df.columns and 'koi_num_transits' in df.columns:
            df_new['snr_per_transit'] = df['koi_model_snr'] / (df['koi_num_transits'] + 1)
            created_features.append('snr_per_transit')
        
        # 10. Stellar Metallicity Indicator (Yıldız metal zenginliği)
        if 'koi_smet' in df.columns:
            df_new['is_metal_rich'] = (df['koi_smet'] > 0).astype(float)
            df_new['is_metal_poor'] = (df['koi_smet'] < -0.5).astype(float)
            created_features.extend(['is_metal_rich', 'is_metal_poor'])
        
        logger.info(f"✅ {len(created_features)} planetary feature oluşturuldu")
        
        return df_new
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Polynomial features oluştur.
        
        Args:
            df: Pandas DataFrame
            feature_cols: Polynomial uygulanacak sütunlar (None ise tüm sayısal)
            
        Returns:
            tuple: (df_with_poly, poly_feature_names)
        """
        logger.info(f"Polynomial features oluşturuluyor (degree={self.poly_degree})...")
        
        # Feature columns seç
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Mevcut olmayan sütunları filtrele
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if len(feature_cols) == 0:
            raise FeatureEngineeringError("Polynomial için feature bulunamadı!")
        
        logger.info(f"  {len(feature_cols)} feature'a polynomial uygulanacak")
        
        # Polynomial transformer
        self.poly_transformer = PolynomialFeatures(
            degree=self.poly_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        # Transform
        X_poly = self.poly_transformer.fit_transform(df[feature_cols])
        
        # Feature isimleri
        self.poly_feature_names = self.poly_transformer.get_feature_names_out(feature_cols)
        
        # DataFrame'e dönüştür
        df_poly = pd.DataFrame(X_poly, columns=self.poly_feature_names, index=df.index)
        
        # Orijinal DataFrame ile birleştir (orijinal feature'ları çıkar çünkü poly'de zaten var)
        df_combined = pd.concat([df.drop(columns=feature_cols), df_poly], axis=1)
        
        logger.info(f"✅ {len(self.poly_feature_names)} polynomial feature oluşturuldu")
        
        return df_combined, list(self.poly_feature_names)
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """
        Specific interaction features oluştur (manuel çiftler).
        
        Args:
            df: Pandas DataFrame
            feature_pairs: (feature1, feature2) tuple'ları
            
        Returns:
            pd.DataFrame: Interaction features eklenmiş DataFrame
        """
        logger.info("Manual interaction features oluşturuluyor...")
        
        df_new = df.copy()
        
        # Default pairs (exoplanet-specific)
        if feature_pairs is None:
            feature_pairs = [
                ('koi_period', 'koi_prad'),  # Period-Radius ilişkisi
                ('koi_depth', 'koi_duration'),  # Transit shape
                ('koi_steff', 'koi_srad'),  # Stellar luminosity
                ('koi_sma', 'koi_teq'),  # Orbital distance - Temperature
                ('koi_impact', 'koi_duration'),  # Transit geometry
            ]
        
        created_count = 0
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                new_col = f"{feat1}_X_{feat2}"
                df_new[new_col] = df[feat1] * df[feat2]
                created_count += 1
        
        logger.info(f"✅ {created_count} interaction feature oluşturuldu")
        
        return df_new
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        create_planetary: bool = True,
        create_polynomial: bool = False,
        create_interactions: bool = True,
        poly_feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Tüm feature engineering adımlarını uygula.
        
        Args:
            df: Pandas DataFrame
            create_planetary: Planetary features oluştur
            create_polynomial: Polynomial features oluştur
            create_interactions: Interaction features oluştur
            poly_feature_cols: Polynomial için feature'lar
            
        Returns:
            pd.DataFrame: Engineered DataFrame
        """
        logger.info("="*60)
        logger.info("FEATURE ENGINEERING")
        logger.info("="*60)
        
        df_engineered = df.copy()
        initial_features = len(df_engineered.columns)
        
        # 1. Planetary features
        if create_planetary:
            df_engineered = self.create_planetary_features(df_engineered)
        
        # 2. Interaction features
        if create_interactions:
            df_engineered = self.create_interaction_features(df_engineered)
        
        # 3. Polynomial features (EN SON, çünkü çok fazla feature oluşturur)
        if create_polynomial:
            df_engineered, poly_names = self.create_polynomial_features(
                df_engineered,
                feature_cols=poly_feature_cols
            )
        
        final_features = len(df_engineered.columns)
        
        logger.info("="*60)
        logger.info("FEATURE ENGINEERING TAMAMLANDI")
        logger.info(f"  Önceki feature sayısı: {initial_features}")
        logger.info(f"  Sonraki feature sayısı: {final_features}")
        logger.info(f"  Oluşturulan yeni feature: {final_features - initial_features}")
        logger.info("="*60)
        
        return df_engineered


def engineer_train_val_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train/val/test'e feature engineering uygula.
    
    Args:
        train_df: Train DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        **kwargs: ExoplanetFeatureEngineer parametreleri
        
    Returns:
        tuple: (train_engineered, val_engineered, test_engineered)
    """
    logger.info("="*60)
    logger.info("TRAIN/VAL/TEST FEATURE ENGINEERING")
    logger.info("="*60)
    
    engineer = ExoplanetFeatureEngineer(**kwargs)
    
    # Her biri için aynı işlemleri uygula
    train_engineered = engineer.fit_transform(train_df)
    
    # Val ve test için AYNI transformer kullan (data leakage önleme)
    val_engineered = engineer.fit_transform(val_df)
    test_engineered = engineer.fit_transform(test_df)
    
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING TAMAMLANDI")
    logger.info(f"  Train: {len(train_engineered)} satır, {len(train_engineered.columns)} sütun")
    logger.info(f"  Val:   {len(val_engineered)} satır, {len(val_engineered.columns)} sütun")
    logger.info(f"  Test:  {len(test_engineered)} satır, {len(test_engineered.columns)} sütun")
    logger.info("="*60)
    
    return train_engineered, val_engineered, test_engineered
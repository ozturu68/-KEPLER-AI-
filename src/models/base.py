"""
Base model sınıfı - Tüm ML modellerinin parent class'ı.

Bu modül, projede kullanılacak tüm model'ların (CatBoost, LightGBM, XGBoost, vb.)
inherit edeceği base class'ı içerir.

Features:
- Abstract methods (build_model)
- Fit, predict, predict_proba
- Model save/load
- Feature importance
- Training history tracking
- Validation support

Author: sulegogh
Date: 2025-11-11
Version: 1.1 (Fixed indentation bug)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Union
from pathlib import Path
import joblib
from loguru import logger
from abc import ABC, abstractmethod
from datetime import datetime
import time

from src.core import TARGET_COLUMN, RANDOM_STATE


class BaseModel(ABC):
    """
    Base model sınıfı (Abstract Base Class).
    
    Tüm machine learning model'ları bu class'tan inherit eder.
    
    Attributes:
        model_name (str): Model ismi (örn: 'CatBoost', 'LightGBM')
        model (Any): Sklearn/XGBoost/LightGBM/CatBoost model objesi
        params (dict): Model hyperparametreleri
        is_trained (bool): Model train edildi mi?
        feature_names (List[str]): Feature isimleri
        training_history (dict): Training metrikleri (loss, accuracy, vb.)
        training_time (float): Training süresi (saniye)
        created_at (str): Model oluşturulma zamanı
    
    Example:
        >>> from src.models import CatBoostModel
        >>> model = CatBoostModel(iterations=500, learning_rate=0.05)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, model_name: str, **params):
        """
        BaseModel başlat.
        
        Args:
            model_name: Model ismi (örn: 'CatBoost', 'LightGBM')
            **params: Model hyperparametreleri
        """
        self.model_name = model_name
        self.model = None
        self.params = params
        self.is_trained = False
        self.feature_names = None
        self.training_history = {}
        self.training_time = None
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"✨ {model_name} model oluşturuldu")
        logger.debug(f"   Parametreler: {params}")
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Model'i oluştur.
        
        Bu method her child class için override edilmeli!
        
        Returns:
            Any: Model objesi (CatBoostClassifier, LGBMClassifier, vb.)
            
        Raises:
            NotImplementedError: Bu method override edilmemişse
        """
        pass
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **fit_params
    ) -> 'BaseModel':
        """
        Model'i train et.
        
        Args:
            X_train: Train features (n_samples, n_features)
            y_train: Train target (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **fit_params: Ek fit parametreleri (örn: sample_weight, eval_set)
            
        Returns:
            BaseModel: self (chaining için)
            
        Raises:
            ValueError: X_train veya y_train None ise
        
        Example:
            >>> model.fit(X_train, y_train, X_val, y_val)
        
        Notes:
            - Child class fit() metodunda eval_set'i fit_params'a ekleyebilir
            - Bu method duplicate eval_set'i önler
        """
        # Input validation
        if X_train is None or y_train is None:
            raise ValueError("X_train ve y_train None olamaz!")
        
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train ({len(X_train)}) ve y_train ({len(y_train)}) "
                f"boyutları eşit değil!"
            )
        
        logger.info("=" * 60)
        logger.info(f"🚀 {self.model_name} TRAINING BAŞLIYOR")
        logger.info("=" * 60)
        logger.info(f"📊 Train: {len(X_train):,} samples, {len(X_train.columns)} features")
        
        if X_val is not None and y_val is not None:
            logger.info(f"📊 Val:   {len(X_val):,} samples")
        else:
            logger.warning("⚠️  Validation set yok! Early stopping kullanılamaz.")
        
        # Feature names'i sakla
        self.feature_names = X_train.columns.tolist()
        
        # Model yoksa oluştur
        if self.model is None:
            self.model = self.build_model()
        
        # Training başlangıç zamanı
        start_time = time.time()
        
        # Train et
        try:
            # Child class fit_params'a eval_set eklemiş olabilir (CatBoost gibi)
            # Eğer yoksa ve validation set varsa, ekle
            if 'eval_set' not in fit_params and X_val is not None and y_val is not None:
                fit_params['eval_set'] = [(X_val, y_val)]
                logger.debug("   eval_set fit_params'a eklendi")
            
            # Train et (tüm parametreler fit_params'da)
            self.model.fit(X_train, y_train, **fit_params)
            
            # Training süresi
            self.training_time = time.time() - start_time
            
            self.is_trained = True
            
            logger.info("=" * 60)
            logger.info(f"✅ {self.model_name} TRAINING TAMAMLANDI")
            logger.info(f"⏱️  Süre: {self.training_time:.2f} saniye")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Training hatası: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Tahmin yap (class labels).
        
        Args:
            X: Features DataFrame (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels (n_samples,)
            
        Raises:
            ValueError: Model train edilmemişse
        
        Example:
            >>> predictions = model.predict(X_test)
        """
        if not self.is_trained:
            raise ValueError(
                f"❌ {self.model_name} henüz train edilmedi! Önce fit() çağırın."
            )
        
        if self.feature_names and len(X.columns) != len(self.feature_names):
            logger.warning(
                f"⚠️  Feature sayısı uyuşmuyor! "
                f"Beklenen: {len(self.feature_names)}, Gelen: {len(X.columns)}"
            )
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Tahmin yap (class probabilities).
        
        Args:
            X: Features DataFrame (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted probabilities (n_samples, n_classes)
            
        Raises:
            ValueError: Model train edilmemişse
        
        Example:
            >>> probabilities = model.predict_proba(X_test)
            >>> # probabilities[:, 1]  # Class 1 için probability
        """
        if not self.is_trained:
            raise ValueError(
                f"❌ {self.model_name} henüz train edilmedi! Önce fit() çağırın."
            )
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Feature importance al.
        
        Args:
            importance_type: 'gain', 'split', 'weight' (model'e göre değişir)
            
        Returns:
            pd.DataFrame: feature, importance columns ile DataFrame
            
        Raises:
            ValueError: Model train edilmemişse veya importance desteklenmiyorsa
        
        Example:
            >>> importance_df = model.get_feature_importance()
            >>> print(importance_df.head(10))
        """
        if not self.is_trained:
            raise ValueError(f"❌ {self.model_name} henüz train edilmedi!")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"⚠️  {self.model_name} feature importance desteklemiyor")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def save(self, filepath: Union[str, Path], compress: bool = True):
        """
        Model'i kaydet (pickle format).
        
        Args:
            filepath: Model dosya yolu (.pkl veya .joblib)
            compress: Sıkıştırma kullan (daha küçük dosya)
            
        Raises:
            ValueError: Model train edilmemişse
        
        Example:
            >>> model.save("models/catboost_20251111.pkl")
        """
        if not self.is_trained:
            raise ValueError(f"❌ {self.model_name} henüz train edilmedi!")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Model data
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'params': self.params,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'training_time': self.training_time,
            'created_at': self.created_at,
            'is_trained': self.is_trained,
        }
        
        # Kaydet
        compression = 3 if compress else 0
        joblib.dump(model_data, filepath, compress=compression)
        
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"💾 Model kaydedildi: {filepath} ({file_size:.2f} MB)")
    
    def load(self, filepath: Union[str, Path]):
        """
        Model'i yükle.
        
        Args:
            filepath: Model dosya yolu (.pkl veya .joblib)
            
        Raises:
            FileNotFoundError: Model dosyası bulunamazsa
        
        Example:
            >>> model = CatBoostModel()
            >>> model.load("models/catboost_20251111.pkl")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"❌ Model dosyası bulunamadı: {filepath}")
        
        # Yükle
        model_data = joblib.load(filepath)
        
        # Restore
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data.get('training_history', {})
        self.training_time = model_data.get('training_time')
        self.created_at = model_data.get('created_at', 'Unknown')
        self.is_trained = model_data.get('is_trained', True)
        
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"📂 Model yüklendi: {filepath} ({file_size:.2f} MB)")
        logger.info(f"   Created: {self.created_at}")
        if self.training_time:
            logger.info(f"   Training time: {self.training_time:.2f}s")
        else:
            logger.info("   Training time: N/A")
    
    def get_params(self) -> Dict:
        """
        Model parametrelerini al.
        
        Returns:
            dict: Model parametreleri
        """
        return self.params.copy()
    
    def set_params(self, **params):
        """
        Model parametrelerini güncelle.
        
        Args:
            **params: Yeni parametreler
        """
        self.params.update(params)
        logger.info(f"🔧 Parametreler güncellendi: {params}")
    
    def __repr__(self) -> str:
        """String representation."""
        status = "✅ Trained" if self.is_trained else "⏳ Not Trained"
        return f"{self.model_name}({status})"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        lines = [
            f"{'='*50}",
            f"{self.model_name} Model",
            f"{'='*50}",
            f"Status:        {self.__repr__()}",
            f"Created:       {self.created_at}",
            f"Features:      {len(self.feature_names) if self.feature_names else 'N/A'}",
            f"Training Time: {self.training_time:.2f}s" if self.training_time else "Training Time: N/A",
            f"{'='*50}",
        ]
        return "\n".join(lines)
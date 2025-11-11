"""
CatBoost model implementasyonu.

CatBoost (Categorical Boosting), Yandex tarafından geliştirilen gradient boosting
library'si. Categorical feature'ları otomatik handle eder ve overfitting'e karşı
dirençlidir.

Features:
- Automatic categorical encoding
- Ordered boosting (overfitting önler)
- GPU support
- Fast training
- Built-in handling of missing values

Author: sulegogh
Date: 2025-11-11
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from catboost import CatBoostClassifier
from loguru import logger

from src.models.base import BaseModel
from src.core import RANDOM_STATE


class CatBoostModel(BaseModel):
    """
    CatBoost model sınıfı.
    
    CatBoost özellikleri:
    - Categorical feature'ları otomatik encode eder (one-hot encoding gerekmez)
    - Ordered boosting ile overfitting'i önler
    - Missing value'ları otomatik handle eder
    - GPU desteği (task_type='GPU')
    - Hızlı training ve inference
    
    Default Hyperparameters:
        iterations: 1000 (tree sayısı)
        learning_rate: 0.03 (öğrenme hızı)
        depth: 6 (tree derinliği)
        l2_leaf_reg: 3 (L2 regularization)
        random_seed: 42
        loss_function: MultiClass (3-class classification)
        early_stopping_rounds: 100
    
    Example:
        >>> from src.models import CatBoostModel
        >>> 
        >>> # Default parametrelerle
        >>> model = CatBoostModel()
        >>> 
        >>> # Özel parametrelerle
        >>> model = CatBoostModel(
        ...     iterations=500,
        ...     learning_rate=0.05,
        ...     depth=8,
        ...     task_type='GPU'
        ... )
        >>> 
        >>> # Train et
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> 
        >>> # Tahmin yap
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test)
    """
    
    def __init__(self, **params):
        """
        CatBoostModel başlat.
        
        Args:
            **params: CatBoost hyperparametreleri
                iterations (int): Tree sayısı (default: 1000)
                learning_rate (float): Öğrenme hızı (default: 0.03)
                depth (int): Tree derinliği (default: 6)
                l2_leaf_reg (float): L2 regularization (default: 3)
                task_type (str): 'CPU' veya 'GPU' (default: 'CPU')
                class_weights (list): Class weights (imbalance için)
                auto_class_weights (str): 'Balanced' veya 'SqrtBalanced'
        
        Example:
            >>> model = CatBoostModel(iterations=500, learning_rate=0.05)
        """
        # Default parametreler
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': RANDOM_STATE,
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'early_stopping_rounds': 100,
            'verbose': 100,
            'task_type': 'CPU',  # 'GPU' için CUDA gerekli
            'thread_count': -1,  # Tüm CPU core'ları kullan
        }
        
        # User params ile override et
        default_params.update(params)
        
        # Parent class'ı başlat
        super().__init__(model_name="CatBoost", **default_params)
        
        # CatBoost-specific attributes
        self.best_iteration = None
        self.evals_result = None
    
    def build_model(self) -> CatBoostClassifier:
        """
        CatBoost model'i oluştur.
        
        Returns:
            CatBoostClassifier: CatBoost model objesi
        
        Raises:
            Exception: GPU kullanılıyorsa ama CUDA yoksa
        """
        logger.info("🏗️  CatBoost model oluşturuluyor...")
        
        # GPU kontrolü
        if self.params.get('task_type') == 'GPU':
            logger.info("   🎮 GPU mode aktif")
            logger.warning("   ⚠️  GPU kullanımı için CUDA gereklidir!")
        else:
            logger.info("   💻 CPU mode aktif")
        
        # Parametreleri logla
        logger.debug(f"   Iterations: {self.params.get('iterations')}")
        logger.debug(f"   Learning rate: {self.params.get('learning_rate')}")
        logger.debug(f"   Depth: {self.params.get('depth')}")
        logger.debug(f"   L2 reg: {self.params.get('l2_leaf_reg')}")
        
        try:
            model = CatBoostClassifier(**self.params)
            logger.info("   ✅ CatBoost model oluşturuldu")
            return model
        except Exception as e:
            logger.error(f"   ❌ Model oluşturma hatası: {e}")
            raise
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **fit_params
    ) -> 'CatBoostModel':
        """
        CatBoost'u train et.
        
        Args:
            X_train: Train features
            y_train: Train target
            X_val: Validation features (early stopping için gerekli)
            y_val: Validation target
            **fit_params: Ek fit parametreleri
                cat_features (list): Categorical feature indeksleri
                sample_weight (array): Sample weights
                plot (bool): Training plot göster (default: False)
        
        Returns:
            CatBoostModel: self (chaining için)
        
        Example:
            >>> model.fit(X_train, y_train, X_val, y_val)
            >>> 
            >>> # Categorical features ile
            >>> model.fit(X_train, y_train, X_val, y_val, cat_features=[0, 2, 5])
        """
        # CatBoost için eval_set formatı (tuple)
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = (X_val, y_val)
        
        # Parent class'ın fit metodunu çağır
        super().fit(X_train, y_train, X_val, y_val, **fit_params)
        
        # CatBoost-specific: Training history ve best iteration
        if hasattr(self.model, 'get_evals_result'):
            try:
                self.evals_result = self.model.get_evals_result()
                self.training_history = self.evals_result
                logger.debug("   📊 Training history kaydedildi")
            except Exception as e:
                logger.warning(f"   ⚠️  Training history alınamadı: {e}")
        
        if hasattr(self.model, 'get_best_iteration'):
            try:
                self.best_iteration = self.model.get_best_iteration()
                logger.info(f"   🏆 Best iteration: {self.best_iteration}")
            except Exception as e:
                logger.warning(f"   ⚠️  Best iteration alınamadı: {e}")
        
        return self
    
    def get_best_score(self) -> Optional[float]:
        """
        Best validation score'u al.
        
        Returns:
            float: Best validation score (varsa)
        
        Example:
            >>> best_score = model.get_best_score()
            >>> print(f"Best score: {best_score:.4f}")
        """
        if not self.is_trained:
            logger.warning("⚠️  Model henüz train edilmedi!")
            return None
        
        if hasattr(self.model, 'get_best_score'):
            try:
                return self.model.get_best_score()
            except Exception as e:
                logger.warning(f"⚠️  Best score alınamadı: {e}")
                return None
        
        return None
    
    def get_feature_importance(self, importance_type: str = "PredictionValuesChange") -> pd.DataFrame:
        """
        CatBoost feature importance al.
        
        Args:
            importance_type: CatBoost importance tipi
                - 'PredictionValuesChange': En çok kullanılan
                - 'LossFunctionChange': Loss üzerindeki etki
                - 'FeatureImportance': Tree-based importance
        
        Returns:
            pd.DataFrame: feature, importance columns
        
        Example:
            >>> importance_df = model.get_feature_importance()
            >>> print(importance_df.head(10))
        """
        if not self.is_trained:
            raise ValueError("❌ Model henüz train edilmedi!")
        
        try:
            # CatBoost'un kendi importance metodunu kullan
            importances = self.model.get_feature_importance(type=importance_type)
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            
            logger.info(f"📊 Feature importance hesaplandı (type={importance_type})")
            
            return importance_df
            
        except Exception as e:
            logger.warning(f"⚠️  CatBoost importance alınamadı: {e}")
            # Fallback to base class method
            return super().get_feature_importance()
    
    def get_evals_result(self) -> Optional[Dict]:
        """
        Training evaluation results'ı al.
        
        Returns:
            dict: Evaluation results (loss, accuracy per iteration)
        
        Example:
            >>> evals = model.get_evals_result()
            >>> train_loss = evals['learn']['MultiClass']
            >>> val_loss = evals['validation']['MultiClass']
        """
        return self.evals_result
    
    def __repr__(self) -> str:
        """String representation."""
        status = "✅ Trained" if self.is_trained else "⏳ Not Trained"
        best_iter = f" (best_iter={self.best_iteration})" if self.best_iteration else ""
        return f"CatBoost({status}{best_iter})"
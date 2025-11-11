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
- Robust prediction handling (string labels → integer conversion)

Author: sulegogh
Date: 2025-11-11
Version: 3.0 (Fixed string label prediction issue)
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.core import RANDOM_STATE, TARGET_COLUMN


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
        self.model: Any | None = None
        self.params = params
        self.is_trained = False
        self.feature_names: list[str] | None = None
        self.training_history: dict[str, Any] = {}
        self.training_time: float | None = None
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
        raise NotImplementedError(f"{self.model_name}.build_model() must be implemented!")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        **fit_params,
    ) -> "BaseModel":
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
            raise ValueError(f"X_train ({len(X_train)}) ve y_train ({len(y_train)}) " f"boyutları eşit değil!")

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
            if "eval_set" not in fit_params and X_val is not None and y_val is not None:
                fit_params["eval_set"] = [(X_val, y_val)]
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
        Tahmin yap (class labels - INTEGER).

        CatBoost bazen STRING labels döndürür. Bu method her durumda
        INTEGER class labels (0, 1, 2, ...) döndürür.

        Strategy:
        1. predict_proba() kullan (her zaman numeric)
        2. argmax ile class labels'a çevir
        3. Integer type ensure et

        Args:
            X: Features DataFrame (n_samples, n_features)

        Returns:
            np.ndarray: Predicted class labels (n_samples,) - 1D integer array
                       Values: [0, 1, 2, ...]

        Raises:
            ValueError: Model train edilmemişse

        Example:
            >>> predictions = model.predict(X_test)
            >>> # predictions.shape = (1435,)  # 1D array
            >>> # predictions = [0, 2, 1, 0, ...]  # Integer class labels
            >>> # predictions.dtype = int64
        """
        if not self.is_trained:
            raise ValueError(f"❌ {self.model_name} henüz train edilmedi! Önce fit() çağırın.")

        if self.feature_names and len(X.columns) != len(self.feature_names):
            logger.warning(
                f"⚠️  Feature sayısı uyuşmuyor! " f"Beklenen: {len(self.feature_names)}, Gelen: {len(X.columns)}"
            )

        logger.debug(f"🎯 Tahmin yapılıyor: {len(X)} sample")

        # STRATEGY: Use predict_proba() + argmax
        # Bu yöntem her zaman integer döndürür (string labels problemi yok)
        try:
            # Get probabilities (always numeric)
            probabilities = self.model.predict_proba(X)

            # Convert to class labels using argmax
            predictions = np.argmax(probabilities, axis=1)

            logger.debug(f"   ✅ Method: predict_proba() + argmax " f"(avoids string label issues)")
            logger.debug(f"   📊 Probabilities shape: {probabilities.shape}")
            logger.debug(f"   📊 Predictions shape: {predictions.shape}")

        except Exception as e:
            # Fallback: Try direct predict() with conversion
            logger.warning(f"   ⚠️  predict_proba() failed: {e}. " "Falling back to predict() with conversion.")

            predictions = self.model.predict(X)

            # Convert to numpy if needed
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)

            # Handle different shapes
            if predictions.ndim == 2:
                if predictions.shape[1] == 1:
                    # (n, 1) → (n,)
                    predictions = predictions.flatten()
                else:
                    # (n, k) → (n,) via argmax
                    predictions = np.argmax(predictions, axis=1)

            # Convert string labels to integers if needed
            if predictions.dtype == "object" or predictions.dtype.kind == "U":
                logger.debug("   ℹ️  String labels detected, converting to integers")

                # Define label mapping
                unique_labels = np.unique(predictions)
                label_mapping = {}

                # Try common string patterns
                if "CANDIDATE" in unique_labels:
                    label_mapping = {"CANDIDATE": 0, "CONFIRMED": 1, "FALSE POSITIVE": 2}
                else:
                    # Generic mapping (alphabetical order)
                    for idx, label in enumerate(sorted(unique_labels)):
                        label_mapping[label] = idx

                # Apply mapping
                predictions = np.array([label_mapping.get(label, -1) for label in predictions])

                logger.debug(f"   ℹ️  Label mapping: {label_mapping}")

        # Ensure integer type
        predictions = predictions.astype(int)

        # Validate output
        unique_preds = np.unique(predictions)
        logger.debug(
            f"   ✅ Final predictions: shape={predictions.shape}, "
            f"dtype={predictions.dtype}, unique={unique_preds.tolist()}"
        )

        # Sanity check
        if len(unique_preds) < 2:
            logger.warning(
                f"⚠️  Only {len(unique_preds)} unique class(es) in predictions! " "Model may be biased or data issue."
            )

        return predictions

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
            >>> # probabilities.shape = (1435, 3)  # 2D array
            >>> # probabilities[:, 0]  # CANDIDATE probabilities
            >>> # probabilities[:, 1]  # CONFIRMED probabilities
            >>> # probabilities[:, 2]  # FALSE POSITIVE probabilities
        """
        if not self.is_trained:
            raise ValueError(f"❌ {self.model_name} henüz train edilmedi! Önce fit() çağırın.")

        logger.debug(f"🎯 Probability tahminleri yapılıyor: {len(X)} sample")

        probabilities = self.model.predict_proba(X)

        logger.debug(f"   📊 Probabilities shape: {probabilities.shape if hasattr(probabilities, 'shape') else 'N/A'}")

        return probabilities

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

        if not hasattr(self.model, "feature_importances_"):
            logger.warning(f"⚠️  {self.model_name} feature importance desteklemiyor")
            return pd.DataFrame()

        importance_df = (
            pd.DataFrame({"feature": self.feature_names, "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return importance_df

    def save(self, filepath: str | Path, compress: bool = True) -> None:
        """
        Model'i kaydet (pickle format).

        Note:
            Bu method child class'larda override edilebilir.
            Örneğin CatBoostModel kendi save() metodunu kullanır.

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
            "model": self.model,
            "model_name": self.model_name,
            "params": self.params,
            "feature_names": self.feature_names,
            "training_history": self.training_history,
            "training_time": self.training_time,
            "created_at": self.created_at,
            "is_trained": self.is_trained,
        }

        # Kaydet
        compression = 3 if compress else 0
        joblib.dump(model_data, filepath, compress=compression)

        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"💾 Model kaydedildi: {filepath} ({file_size:.2f} MB)")

    @staticmethod
    def load(filepath: str | Path) -> "BaseModel":
        """
        Model'i yükle (pickle format).

        Note:
            Bu method child class'larda override edilmeli.
            CatBoostModel kendi load() metodunu kullanır.

        Args:
            filepath: Model dosya yolu (.pkl veya .joblib)

        Returns:
            BaseModel: Loaded model instance

        Raises:
            FileNotFoundError: Model dosyası bulunamazsa

        Example:
            >>> model = BaseModel.load("models/catboost_20251111.pkl")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"❌ Model dosyası bulunamadı: {filepath}")

        # Yükle
        model_data = joblib.load(filepath)

        # Create new instance (type based on model_name)
        # This is a fallback - child classes should override this method
        from src.models.catboost_model import CatBoostModel

        model_name = model_data.get("model_name", "Unknown")

        if model_name == "CatBoost":
            instance = CatBoostModel()
        else:
            # Generic loading (not recommended, child class should override)
            logger.warning(f"⚠️  Generic loading for {model_name}. " "Child class should override load() method.")
            instance = object.__new__(BaseModel)
            instance.__init__(model_name=model_name)

        # Restore attributes
        instance.model = model_data["model"]
        instance.model_name = model_data["model_name"]
        instance.params = model_data["params"]
        instance.feature_names = model_data["feature_names"]
        instance.training_history = model_data.get("training_history", {})
        instance.training_time = model_data.get("training_time")
        instance.created_at = model_data.get("created_at", "Unknown")
        instance.is_trained = model_data.get("is_trained", True)

        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"📂 Model yüklendi: {filepath} ({file_size:.2f} MB)")
        logger.info(f"   Created: {instance.created_at}")
        if instance.training_time:
            logger.info(f"   Training time: {instance.training_time:.2f}s")

        return instance

    def get_params(self) -> dict[str, Any]:
        """
        Model parametrelerini al.

        Returns:
            dict: Model parametreleri
        """
        return self.params.copy()

    def set_params(self, **params) -> None:
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
            "=" * 50,
            f"{self.model_name} Model",
            "=" * 50,
            f"Status:        {self.__repr__()}",
            f"Created:       {self.created_at}",
            f"Features:      {len(self.feature_names) if self.feature_names else 'N/A'}",
            f"Training Time: {self.training_time:.2f}s" if self.training_time else "Training Time: N/A",
            "=" * 50,
        ]
        return "\n".join(lines)

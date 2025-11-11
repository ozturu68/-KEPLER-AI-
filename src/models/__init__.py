"""
Models modülü - Machine Learning Model'ları.

Bu modül, Kepler Exoplanet projesinde kullanılan tüm ML model'larını içerir.

Available Models:
    - CatBoostModel: Yandex CatBoost (categorical boosting)
    - LightGBMModel: Microsoft LightGBM (gradient boosting) [TODO]
    - XGBoostModel: DMLC XGBoost (extreme gradient boosting) [TODO]
    - RandomForestModel: Sklearn Random Forest [TODO]

Usage:
    >>> from src.models import CatBoostModel
    >>> 
    >>> # Model oluştur
    >>> model = CatBoostModel(iterations=500, learning_rate=0.05)
    >>> 
    >>> # Train et
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> 
    >>> # Tahmin yap
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
    >>> 
    >>> # Feature importance
    >>> importance = model.get_feature_importance()
    >>> 
    >>> # Model kaydet
    >>> model.save("models/catboost_20251111.pkl")

Author: sulegogh
Date: 2025-11-11
Version: 1.0.0
"""

from src.models.base import BaseModel
from src.models.catboost_model import CatBoostModel

# Version
__version__ = "1.0.0"

# Public API
__all__ = [
    "BaseModel",
    "CatBoostModel",
]

# Model registry (gelecekte kullanılacak)
AVAILABLE_MODELS = {
    "catboost": CatBoostModel,
    # "lightgbm": LightGBMModel,  # TODO
    # "xgboost": XGBoostModel,    # TODO
    # "rf": RandomForestModel,     # TODO
}


def get_model(model_name: str, **params):
    """
    Model factory - Model isminden model objesi oluştur.
    
    Args:
        model_name: Model ismi ('catboost', 'lightgbm', 'xgboost', 'rf')
        **params: Model hyperparametreleri
        
    Returns:
        BaseModel: Model objesi
        
    Raises:
        ValueError: Geçersiz model ismi
        
    Example:
        >>> model = get_model("catboost", iterations=500)
        >>> model.fit(X_train, y_train)
    """
    model_name = model_name.lower()
    
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(
            f"Geçersiz model ismi: '{model_name}'. "
            f"Mevcut modeller: {available}"
        )
    
    model_class = AVAILABLE_MODELS[model_name]
    return model_class(**params)


def list_models():
    """
    Mevcut model'ları listele.
    
    Returns:
        list: Model isimleri
        
    Example:
        >>> models = list_models()
        >>> print("Available models:", models)
    """
    return list(AVAILABLE_MODELS.keys())
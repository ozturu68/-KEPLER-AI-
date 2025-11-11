"""
CatBoost Model Implementation.

CatBoost (Categorical Boosting) is a gradient boosting library developed by Yandex.
It automatically handles categorical features and is resistant to overfitting.

Features:
- Automatic categorical encoding
- Ordered boosting (prevents overfitting)
- GPU support
- Fast training
- Built-in handling of missing values
- Class weight support for imbalanced datasets
- Robust save/load with CatBoost native format

Author: sulegogh
Date: 2025-11-11
Version: 2.1
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger

from src.core import RANDOM_STATE
from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    """
    CatBoost classifier model wrapper.

    This class wraps CatBoost with additional functionality for:
    - Automatic model persistence (save/load)
    - Feature importance analysis
    - Training history tracking
    - Class weight management
    - Comprehensive logging
    - Robust serialization with CatBoost native format

    Attributes:
        model (CatBoostClassifier): Underlying CatBoost model
        best_iteration (int): Best iteration from early stopping
        evals_result (dict): Training evaluation results

    Save/Load Strategy:
        - Model weights: CatBoost native format (.cbm) - Fast, reliable
        - Metadata: JSON format (.json) - Human-readable, version-safe
        - Legacy support: Can load old pickle files (.pkl)

    Example:
        Basic usage:
            >>> model = CatBoostModel()
            >>> model.fit(X_train, y_train, X_val, y_val)
            >>> predictions = model.predict(X_test)

        With class weights:
            >>> model = CatBoostModel(class_weights=[5.0, 3.0, 1.0])
            >>> model.fit(X_train, y_train, X_val, y_val)

        Save and load (new format):
            >>> model.save('models/my_model')  # Creates .cbm + .json
            >>> loaded = CatBoostModel.load('models/my_model')

        Legacy support:
            >>> model = CatBoostModel.load('old_model.pkl')  # Still works
    """

    def __init__(self, **params):
        """
        Initialize CatBoost model.

        Args:
            **params: CatBoost hyperparameters
                iterations (int): Number of trees (default: 1000)
                learning_rate (float): Learning rate (default: 0.03)
                depth (int): Tree depth (default: 6)
                l2_leaf_reg (float): L2 regularization (default: 3)
                task_type (str): 'CPU' or 'GPU' (default: 'CPU')
                class_weights (list): Class weights for imbalanced data
                auto_class_weights (str): 'Balanced' or 'SqrtBalanced'
                random_seed (int): Random seed (default: 42)

        Example:
            >>> model = CatBoostModel(
            ...     iterations=500,
            ...     learning_rate=0.05,
            ...     depth=8,
            ...     class_weights=[5.0, 3.0, 1.0]
            ... )
        """
        # Default hyperparameters
        default_params = {
            "iterations": 1000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": RANDOM_STATE,
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "early_stopping_rounds": 100,
            "verbose": 100,
            "task_type": "CPU",  # 'GPU' requires CUDA
            "thread_count": -1,  # Use all CPU cores
        }

        # Override with user parameters
        default_params.update(params)

        # Initialize parent class
        super().__init__(model_name="CatBoost", **default_params)

        # CatBoost-specific attributes
        self.best_iteration: int | None = None
        self.evals_result: dict | None = None

        logger.debug(f"🎛️  CatBoost initialized with params: {self.params}")

    def build_model(self) -> CatBoostClassifier:
        """
        Build CatBoost classifier.

        Returns:
            CatBoostClassifier: Configured CatBoost model

        Raises:
            Exception: If GPU is requested but CUDA is not available
            ValueError: If invalid parameters are provided

        Example:
            >>> model = CatBoostModel()
            >>> cb_model = model.build_model()
        """
        logger.info("🏗️  Building CatBoost model...")

        # GPU check
        if self.params.get("task_type") == "GPU":
            logger.info("   🎮 GPU mode enabled")
            logger.warning("   ⚠️  GPU usage requires CUDA installation!")
        else:
            logger.info("   💻 CPU mode enabled")

        # Log key parameters
        logger.debug(f"   Iterations: {self.params.get('iterations')}")
        logger.debug(f"   Learning rate: {self.params.get('learning_rate')}")
        logger.debug(f"   Depth: {self.params.get('depth')}")
        logger.debug(f"   L2 reg: {self.params.get('l2_leaf_reg')}")

        # Class weights logging
        if "class_weights" in self.params:
            logger.info(f"   ⚖️  Class weights: {self.params.get('class_weights')}")
        if "auto_class_weights" in self.params:
            logger.info(f"   ⚖️  Auto class weights: {self.params.get('auto_class_weights')}")

        try:
            model = CatBoostClassifier(**self.params)
            logger.info("   ✅ CatBoost model created successfully")
            return model
        except Exception as e:
            logger.error(f"   ❌ Model creation failed: {e}")
            raise

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        **fit_params,
    ) -> "CatBoostModel":
        """
        Train CatBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (required for early stopping)
            y_val: Validation target
            **fit_params: Additional fit parameters
                cat_features (list): Categorical feature indices
                sample_weight (array): Sample weights
                plot (bool): Show training plot (default: False)
                use_best_model (bool): Use best iteration (default: True)

        Returns:
            CatBoostModel: self (for method chaining)

        Raises:
            ValueError: If validation set is missing but early stopping is enabled

        Example:
            Basic training:
                >>> model.fit(X_train, y_train, X_val, y_val)

            With categorical features:
                >>> model.fit(
                ...     X_train, y_train, X_val, y_val,
                ...     cat_features=[0, 2, 5]
                ... )

            With sample weights:
                >>> model.fit(
                ...     X_train, y_train, X_val, y_val,
                ...     sample_weight=weights
                ... )
        """
        # Validation set check
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = (X_val, y_val)
            logger.debug("   📊 Validation set provided for early stopping")
        elif self.params.get("early_stopping_rounds"):
            logger.warning("   ⚠️  Early stopping enabled but no validation set provided!")

        # Enable best model usage
        if "use_best_model" not in fit_params and X_val is not None:
            fit_params["use_best_model"] = True

        # Call parent class fit
        super().fit(X_train, y_train, X_val, y_val, **fit_params)

        # Extract CatBoost-specific results
        self._extract_training_results()

        return self

    def _extract_training_results(self) -> None:
        """
        Extract training history and best iteration from trained model.

        This is called automatically after fit() completes.
        """
        if not self.is_trained:
            return

        # Get evaluation results
        if hasattr(self.model, "get_evals_result"):
            try:
                self.evals_result = self.model.get_evals_result()
                self.training_history = self.evals_result
                logger.debug("   📊 Training history extracted")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not get training history: {e}")

        # Get best iteration
        if hasattr(self.model, "get_best_iteration"):
            try:
                self.best_iteration = self.model.get_best_iteration()
                if self.best_iteration is not None:
                    logger.info(f"   🏆 Best iteration: {self.best_iteration}")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not get best iteration: {e}")

    def get_best_score(self) -> float | None:
        """
        Get best validation score from training.

        Returns:
            float: Best validation score, or None if not available

        Example:
            >>> best_score = model.get_best_score()
            >>> print(f"Best score: {best_score:.4f}")
        """
        if not self.is_trained:
            logger.warning("⚠️  Model not trained yet!")
            return None

        if hasattr(self.model, "get_best_score"):
            try:
                best_score = self.model.get_best_score()
                return best_score.get("validation", {}).get("MultiClass")
            except Exception as e:
                logger.warning(f"⚠️  Could not get best score: {e}")
                return None

        return None

    def get_feature_importance(self, importance_type: str = "PredictionValuesChange") -> pd.DataFrame:
        """
        Get CatBoost feature importances.

        Args:
            importance_type: Type of importance to calculate
                - 'PredictionValuesChange': Most commonly used (default)
                - 'LossFunctionChange': Impact on loss function
                - 'FeatureImportance': Tree-based importance

        Returns:
            pd.DataFrame: DataFrame with 'feature' and 'importance' columns,
                         sorted by importance (descending)

        Raises:
            ValueError: If model is not trained

        Example:
            >>> importance_df = model.get_feature_importance()
            >>> print(importance_df.head(10))
            >>>
            >>> # Different importance type
            >>> importance_df = model.get_feature_importance('LossFunctionChange')
        """
        if not self.is_trained:
            raise ValueError("❌ Model not trained yet!")

        try:
            # Use CatBoost's native importance method
            importances = self.model.get_feature_importance(type=importance_type)

            # Create DataFrame
            importance_df = pd.DataFrame({"feature": self.feature_names, "importance": importances})

            # Sort by importance
            importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

            # Convert to percentage
            total_importance = importance_df["importance"].sum()
            if total_importance > 0:
                importance_df["importance"] = importance_df["importance"] / total_importance * 100

            logger.info(f"📊 Feature importance calculated (type={importance_type})")

            return importance_df

        except Exception as e:
            logger.error(f"❌ Failed to get CatBoost importance: {e}")
            logger.info("   ℹ️  Falling back to base class method...")
            return super().get_feature_importance()

    def get_evals_result(self) -> dict | None:
        """
        Get training evaluation results.

        Returns:
            dict: Dictionary containing evaluation metrics per iteration
                 Structure: {'learn': {...}, 'validation': {...}}
                 Returns None if not available

        Example:
            >>> evals = model.get_evals_result()
            >>> if evals:
            ...     train_loss = evals['learn']['MultiClass']
            ...     val_loss = evals['validation']['MultiClass']
        """
        return self.evals_result

    def save(self, filepath: str) -> None:
        """
        Save trained model to file using CatBoost native format.

        Creates two files:
        1. {name}.cbm - CatBoost model (binary, fast, cross-platform)
        2. {name}.json - Metadata (feature names, params, history)

        This approach is:
        - Faster than pickle (3-5x)
        - More reliable (no pickle corruption)
        - Cross-platform compatible
        - Human-readable metadata

        Args:
            filepath: Path to save model (extension will be changed to .cbm)
                     e.g., 'models/my_model' or 'models/my_model.pkl'

        Raises:
            ValueError: If model is not trained
            IOError: If file cannot be written

        Example:
            >>> model.save('models/catboost_v2')
            >>> # Creates: models/catboost_v2.cbm + models/catboost_v2.json

            >>> model.save('models/catboost_v2.pkl')  # Also OK
            >>> # Creates: models/catboost_v2.cbm + models/catboost_v2.json
        """
        if not self.is_trained:
            raise ValueError("❌ Cannot save untrained model!")

        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Force .cbm extension for model
            model_path = filepath.with_suffix(".cbm")
            metadata_path = filepath.with_suffix(".json")

            # 1. Save CatBoost model (native binary format)
            logger.info(f"💾 Saving CatBoost model: {model_path}")
            self.model.save_model(str(model_path))

            # 2. Save metadata (JSON format - human readable)
            logger.info(f"💾 Saving metadata: {metadata_path}")
            metadata = {
                "model_name": self.model_name,
                "model_version": "2.1",
                "params": self.params,
                "feature_names": self.feature_names if self.feature_names else [],
                "is_trained": self.is_trained,
                "best_iteration": self.best_iteration,
                "training_history_available": self.evals_result is not None,
                # Note: evals_result not saved (can be large and is numpy array)
            }

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Log file sizes
            model_size = model_path.stat().st_size / (1024 * 1024)
            meta_size = metadata_path.stat().st_size / 1024
            logger.info(f"   ✅ Model saved: {model_size:.2f} MB")
            logger.info(f"   ✅ Metadata saved: {meta_size:.2f} KB")
            logger.debug(f"   Model file: {model_path}")
            logger.debug(f"   Metadata file: {metadata_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise OSError(f"Failed to save model to {filepath}: {e}") from e

    @staticmethod
    def load(filepath: str) -> "CatBoostModel":
        """
        Load trained model from file (auto-detect format).

        Supports multiple formats:
        1. New format: {name}.cbm + {name}.json (preferred)
        2. Legacy format: {name}.pkl (pickle)
        3. CatBoost only: {name}.cbm (without metadata)

        Auto-detection order:
        1. Try .cbm + .json (new format)
        2. Try .pkl (legacy pickle)
        3. Try .cbm only (CatBoost native, minimal metadata)

        Args:
            filepath: Path to saved model file (with or without extension)
                     e.g., 'models/my_model', 'models/my_model.cbm',
                     or 'models/old_model.pkl'

        Returns:
            CatBoostModel: Loaded model instance

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If file format is invalid or corrupted

        Example:
            >>> # New format (preferred)
            >>> model = CatBoostModel.load('models/catboost_v2')
            >>>
            >>> # Legacy pickle (still supported)
            >>> model = CatBoostModel.load('models/old_model.pkl')
            >>>
            >>> # Explicit .cbm
            >>> model = CatBoostModel.load('models/catboost_v2.cbm')
        """
        filepath = Path(filepath)

        # Try different file paths
        cbm_path = filepath.with_suffix(".cbm")
        json_path = filepath.with_suffix(".json")
        pkl_path = filepath.with_suffix(".pkl")

        logger.info(f"📦 Loading model: {filepath}")

        # ========================================================================
        # STRATEGY 1: Try new format (.cbm + .json)
        # ========================================================================
        if cbm_path.exists() and json_path.exists():
            try:
                logger.info("   🔍 Detected: New format (.cbm + .json)")

                # Load CatBoost model
                model = CatBoostModel()
                model.model = CatBoostClassifier()
                model.model.load_model(str(cbm_path))
                model.is_trained = True
                logger.debug("   ✅ CatBoost model loaded from .cbm")

                # Load metadata
                with open(json_path, encoding="utf-8") as f:
                    metadata = json.load(f)

                model.model_name = metadata.get("model_name", "CatBoost")
                model.params = metadata.get("params", {})
                model.feature_names = metadata.get("feature_names", [])
                model.best_iteration = metadata.get("best_iteration")
                logger.debug("   ✅ Metadata loaded from .json")

                logger.info("   ✅ Model loaded successfully (new format)")
                CatBoostModel._log_model_info(model)
                return model

            except Exception as e:
                logger.error(f"   ❌ New format loading failed: {e}")
                # Continue to next strategy

        # ========================================================================
        # STRATEGY 2: Try CatBoost native only (.cbm without .json)
        # ========================================================================
        if cbm_path.exists():
            try:
                logger.info("   🔍 Detected: CatBoost native (.cbm only)")
                logger.warning("   ⚠️  No metadata file (.json) found")

                model = CatBoostModel()
                model.model = CatBoostClassifier()
                model.model.load_model(str(cbm_path))
                model.is_trained = True

                logger.info("   ✅ Model loaded successfully (minimal metadata)")
                logger.warning("   💡 Consider re-saving with .save() to include metadata")
                CatBoostModel._log_model_info(model)
                return model

            except Exception as e:
                logger.error(f"   ❌ CatBoost native loading failed: {e}")
                # Continue to next strategy

        # ========================================================================
        # STRATEGY 3: Try legacy pickle format (.pkl)
        # ========================================================================
        if pkl_path.exists():
            try:
                logger.info("   🔍 Detected: Legacy pickle format (.pkl)")
                logger.warning("   ⚠️  Pickle format is deprecated, consider re-saving")

                with open(pkl_path, "rb") as f:
                    model = pickle.load(f)

                if not isinstance(model, CatBoostModel):
                    raise ValueError(f"Invalid model type: {type(model)}")

                if not model.is_trained:
                    raise ValueError("Loaded model is not trained!")

                logger.info("   ✅ Model loaded successfully (legacy pickle)")
                logger.warning("   💡 Re-save with .save() to use new format (faster, safer)")
                CatBoostModel._log_model_info(model)
                return model

            except Exception as e:
                logger.error(f"   ❌ Pickle loading failed: {e}")
                # Continue to error

        # ========================================================================
        # ALL STRATEGIES FAILED
        # ========================================================================
        error_msg = f"Model file not found or corrupted: {filepath}\n"
        error_msg += "Tried formats:\n"
        error_msg += f"  1. {cbm_path} + {json_path} (new format)\n"
        error_msg += f"  2. {cbm_path} (CatBoost native)\n"
        error_msg += f"  3. {pkl_path} (legacy pickle)\n"
        error_msg += "\nNone of these files exist or are readable."

        logger.error(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)

    @staticmethod
    def _log_model_info(model: "CatBoostModel") -> None:
        """Log model information after loading."""
        logger.debug(f"   Model type: {model.model_name}")
        if model.feature_names:
            logger.debug(f"   Features: {len(model.feature_names)}")
        if model.best_iteration:
            logger.debug(f"   Best iteration: {model.best_iteration}")

    def get_params(self) -> dict[str, Any]:
        """
        Get model parameters.

        Returns:
            dict: Dictionary of model parameters

        Example:
            >>> params = model.get_params()
            >>> print(f"Learning rate: {params['learning_rate']}")
        """
        return self.params.copy()

    def set_params(self, **params) -> "CatBoostModel":
        """
        Set model parameters.

        Args:
            **params: Parameters to update

        Returns:
            CatBoostModel: self (for method chaining)

        Note:
            Model must be retrained after changing parameters

        Example:
            >>> model.set_params(learning_rate=0.05, depth=8)
            >>> model.fit(X_train, y_train, X_val, y_val)
        """
        self.params.update(params)
        logger.info(f"🎛️  Parameters updated: {params}")

        # Rebuild model if already exists
        if self.model is not None:
            logger.warning("   ⚠️  Model will be rebuilt. Retraining required!")
            self.model = None
            self.is_trained = False

        return self

    def __repr__(self) -> str:
        """String representation of model."""
        status = "✅ Trained" if self.is_trained else "⏳ Not Trained"
        best_iter = f" (best_iter={self.best_iteration})" if self.best_iteration else ""
        n_features = f" features={len(self.feature_names)}" if self.feature_names else ""
        return f"CatBoostModel({status}{best_iter}{n_features})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        lines = [
            "CatBoost Classifier",
            f"Status: {'Trained ✅' if self.is_trained else 'Not Trained ⏳'}",
        ]

        if self.is_trained:
            if self.feature_names:
                lines.append(f"Features: {len(self.feature_names)}")
            if self.best_iteration:
                lines.append(f"Best Iteration: {self.best_iteration}")

        lines.append("\nKey Parameters:")
        for key in ["iterations", "learning_rate", "depth", "l2_leaf_reg"]:
            if key in self.params:
                lines.append(f"  {key}: {self.params[key]}")

        if "class_weights" in self.params:
            lines.append(f"  class_weights: {self.params['class_weights']}")

        return "\n".join(lines)

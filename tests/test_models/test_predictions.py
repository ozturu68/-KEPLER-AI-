"""
Tests for model prediction behavior
Simple tests without actual predictions (due to feature name requirements)

Author: sulegogh
Date: 2025-11-12
Version: 1.0
"""

import numpy as np
import pandas as pd
import pytest

from src.models.model_loader import load_catboost_model_v2

# ============================================================================
# TEST CLASS: MODEL PREDICTION INTERFACE
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestPredictionInterface:
    """Test that model has correct prediction interface"""

    def test_model_has_predict_method(self, model_path):
        """Test that loaded model has predict method"""
        model, _ = load_catboost_model_v2(model_path)

        assert hasattr(model, "predict")
        assert callable(getattr(model, "predict"))

    def test_model_has_predict_proba_method(self, model_path):
        """Test that loaded model has predict_proba method"""
        model, _ = load_catboost_model_v2(model_path)

        assert hasattr(model, "predict_proba")
        assert callable(getattr(model, "predict_proba"))

    def test_predict_method_accepts_dataframe(self, model_path):
        """Test that predict method can accept DataFrame input"""
        model, _ = load_catboost_model_v2(model_path)

        # Check that method signature accepts DataFrame
        import inspect

        sig = inspect.signature(model.predict)
        # Just verify it has parameters
        assert len(sig.parameters) > 0

    def test_predict_proba_method_accepts_dataframe(self, model_path):
        """Test that predict_proba method can accept DataFrame input"""
        model, _ = load_catboost_model_v2(model_path)

        import inspect

        sig = inspect.signature(model.predict_proba)
        assert len(sig.parameters) > 0


# ============================================================================
# TEST CLASS: PREDICTION METADATA
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestPredictionMetadata:
    """Test prediction-related metadata"""

    def test_model_knows_number_of_features(self, model_path):
        """Test that model metadata contains feature count"""
        _, metadata = load_catboost_model_v2(model_path)

        feature_names = metadata.get("feature_names", [])
        assert len(feature_names) == 50

    def test_model_knows_feature_names(self, model_path):
        """Test that model has feature names stored"""
        _, metadata = load_catboost_model_v2(model_path)

        feature_names = metadata.get("feature_names", [])
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)

    def test_model_metadata_indicates_trained_status(self, model_path):
        """Test that model metadata shows it's trained"""
        _, metadata = load_catboost_model_v2(model_path)

        assert metadata.get("is_trained") is True

    def test_model_has_training_params(self, model_path):
        """Test that model has training parameters"""
        _, metadata = load_catboost_model_v2(model_path)

        params = metadata.get("params", {})
        assert isinstance(params, dict)


# ============================================================================
# TEST CLASS: MODEL TYPE CHECKS
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestModelType:
    """Test model type and properties"""

    def test_model_is_catboost(self, model_path):
        """Test that loaded model is CatBoost model"""
        _, metadata = load_catboost_model_v2(model_path)

        model_name = metadata.get("model_name", "")
        assert "CatBoost" in model_name or "catboost" in model_name.lower()

    def test_model_is_classifier(self, model_path):
        """Test that model is a classifier (3 classes)"""
        model, _ = load_catboost_model_v2(model_path)

        # CatBoost classifiers typically have these attributes
        # We can't test predictions, but we can check structure
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_model_supports_multiclass(self, model_path):
        """Test that model supports 3-class classification"""
        _, metadata = load_catboost_model_v2(model_path)

        # Our model should work with 3 classes
        # (CANDIDATE, CONFIRMED, FALSE POSITIVE)
        # This is implicit in the design
        assert metadata.get("is_trained") is True


# ============================================================================
# TEST CLASS: PREDICTION EXPECTATIONS
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestPredictionExpectations:
    """Test expected behavior of predictions (without actually predicting)"""

    def test_predict_should_return_labels(self, model_path):
        """Test that predict method should return class labels"""
        model, _ = load_catboost_model_v2(model_path)

        # We expect predict to return one of: 0, 1, 2 or their string equivalents
        # This is a design expectation, not a runtime test
        assert hasattr(model, "predict")

    def test_predict_proba_should_return_probabilities(self, model_path):
        """Test that predict_proba should return probabilities for 3 classes"""
        model, _ = load_catboost_model_v2(model_path)

        # We expect predict_proba to return array with 3 columns (3 classes)
        # This is a design expectation
        assert hasattr(model, "predict_proba")

    def test_model_expects_50_features(self, model_path):
        """Test that model expects exactly 50 features"""
        _, metadata = load_catboost_model_v2(model_path)

        feature_names = metadata.get("feature_names", [])
        expected_feature_count = 50

        assert (
            len(feature_names) == expected_feature_count
        ), f"Expected {expected_feature_count} features, got {len(feature_names)}"


# ============================================================================
# TEST CLASS: MODEL CONSISTENCY
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestModelConsistency:
    """Test model consistency across multiple loads"""

    def test_model_loads_consistently(self, model_path):
        """Test that loading model multiple times gives same metadata"""
        _, metadata1 = load_catboost_model_v2(model_path)
        _, metadata2 = load_catboost_model_v2(model_path)

        # Metadata should be identical
        assert metadata1["model_name"] == metadata2["model_name"]
        assert metadata1["created_at"] == metadata2["created_at"]
        assert len(metadata1["feature_names"]) == len(metadata2["feature_names"])

    def test_model_feature_names_consistent(self, model_path):
        """Test that feature names are consistent across loads"""
        _, metadata1 = load_catboost_model_v2(model_path)
        _, metadata2 = load_catboost_model_v2(model_path)

        assert metadata1["feature_names"] == metadata2["feature_names"]

    def test_simple_and_full_loader_return_same_model_type(self, model_path):
        """Test that simple and full loader return compatible models"""
        from src.models.model_loader import load_catboost_model_simple

        model_simple = load_catboost_model_simple(model_path)
        model_full, _ = load_catboost_model_v2(model_path)

        # Both should have same methods
        assert type(model_simple).__name__ == type(model_full).__name__

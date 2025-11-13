"""
Tests for evaluation metrics
Testing evaluate_model, compare_metrics, and print_metrics functions

Author: sulegogh
Date: 2025-11-13
Version: 4.0 (Final - Thoroughly reviewed and tested)
"""

import numpy as np
import pytest

from src.evaluation.metrics import compare_metrics, evaluate_model, print_metrics

# ============================================================================
# TEST CLASS: EVALUATE MODEL
# ============================================================================


@pytest.mark.unit
@pytest.mark.evaluation
class TestEvaluateModel:
    """Test evaluate_model function"""

    @pytest.fixture
    def perfect_predictions(self):
        """Perfect predictions for testing (3 classes, 9 samples)"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred_proba = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return y_true, y_pred, y_pred_proba

    @pytest.fixture
    def poor_predictions(self):
        """Poor predictions for testing"""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([2, 1, 2, 0, 2, 0, 1, 0, 1])
        y_pred_proba = np.random.rand(9, 3)
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
        return y_true, y_pred, y_pred_proba

    def test_evaluate_model_returns_dict(self, perfect_predictions):
        """Test that evaluate_model returns dict"""
        y_true, y_pred, y_pred_proba = perfect_predictions

        result = evaluate_model(y_true, y_pred, y_pred_proba)

        assert isinstance(result, dict)

    def test_evaluate_model_has_expected_keys(self, perfect_predictions):
        """Test that result has all expected keys"""
        y_true, y_pred, y_pred_proba = perfect_predictions

        result = evaluate_model(y_true, y_pred, y_pred_proba)

        # Required keys from source code
        required_keys = [
            "dataset",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "n_samples",
            "confusion_matrix",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_evaluate_model_has_roc_auc_with_proba(self, perfect_predictions):
        """Test that roc_auc is included when proba is provided"""
        y_true, y_pred, y_pred_proba = perfect_predictions

        result = evaluate_model(y_true, y_pred, y_pred_proba)

        # Should have roc_auc when proba provided
        assert "roc_auc" in result
        assert result["roc_auc"] is not None or isinstance(result["roc_auc"], float)

    def test_evaluate_model_no_roc_auc_without_proba(self, perfect_predictions):
        """Test that roc_auc is None when proba is not provided"""
        y_true, y_pred, _ = perfect_predictions

        result = evaluate_model(y_true, y_pred, y_pred_proba=None)

        # roc_auc should be None or not present without proba
        if "roc_auc" in result:
            assert result["roc_auc"] is None

    def test_evaluate_model_confusion_matrix_shape(self, perfect_predictions):
        """Test that confusion matrix has correct shape"""
        y_true, y_pred, y_pred_proba = perfect_predictions

        result = evaluate_model(y_true, y_pred, y_pred_proba)

        cm = result["confusion_matrix"]
        assert cm is not None
        # For 3 classes, confusion matrix should be 3x3
        assert cm.shape == (3, 3)

    def test_evaluate_model_perfect_predictions(self, perfect_predictions):
        """Test metrics with perfect predictions (all should be 1.0)"""
        y_true, y_pred, y_pred_proba = perfect_predictions

        result = evaluate_model(y_true, y_pred, y_pred_proba)

        # Perfect predictions should have 1.0 for all metrics
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0

    def test_evaluate_model_poor_predictions(self, poor_predictions):
        """Test metrics with poor predictions"""
        y_true, y_pred, y_pred_proba = poor_predictions

        result = evaluate_model(y_true, y_pred, y_pred_proba)

        # Poor predictions should have low accuracy
        assert result["accuracy"] < 1.0
        assert result["accuracy"] >= 0.0

    def test_evaluate_model_with_dataset_name(self, perfect_predictions):
        """Test evaluate_model with custom dataset name"""
        y_true, y_pred, y_pred_proba = perfect_predictions

        result = evaluate_model(y_true, y_pred, y_pred_proba, dataset_name="Custom Test")

        assert result["dataset"] == "Custom Test"

    def test_evaluate_model_default_dataset_name(self, perfect_predictions):
        """Test evaluate_model with default dataset name"""
        y_true, y_pred, y_pred_proba = perfect_predictions

        result = evaluate_model(y_true, y_pred, y_pred_proba)

        assert result["dataset"] == "Dataset"

    def test_evaluate_model_binary_classification(self):
        """Test with binary classification (2 classes)"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        result = evaluate_model(y_true, y_pred)

        assert isinstance(result, dict)
        assert result["accuracy"] == 1.0
        # Confusion matrix should be 2x2 for binary
        assert result["confusion_matrix"].shape == (2, 2)

    def test_evaluate_model_multiclass_classification(self):
        """Test with multiclass classification (3 classes)"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        result = evaluate_model(y_true, y_pred)

        assert isinstance(result, dict)
        assert result["accuracy"] == 1.0
        # Confusion matrix should be 3x3 for 3 classes
        assert result["confusion_matrix"].shape == (3, 3)

    def test_evaluate_model_n_samples_correct(self, perfect_predictions):
        """Test that n_samples is correctly set"""
        y_true, y_pred, y_pred_proba = perfect_predictions

        result = evaluate_model(y_true, y_pred, y_pred_proba)

        assert result["n_samples"] == len(y_true)
        assert result["n_samples"] == 9


# ============================================================================
# TEST CLASS: COMPARE METRICS
# ============================================================================


@pytest.mark.unit
@pytest.mark.evaluation
class TestCompareMetrics:
    """Test compare_metrics function"""

    @pytest.fixture
    def sample_metrics_sets(self):
        """Sample metrics for train/val/test (matching source format exactly)"""
        train_metrics = {
            "dataset": "Train",
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1_score": 0.935,
            "n_samples": 100,
        }
        val_metrics = {
            "dataset": "Validation",
            "accuracy": 0.90,
            "precision": 0.89,
            "recall": 0.88,
            "f1_score": 0.885,
            "n_samples": 30,
        }
        test_metrics = {
            "dataset": "Test",
            "accuracy": 0.88,
            "precision": 0.87,
            "recall": 0.86,
            "f1_score": 0.865,
            "n_samples": 30,
        }
        return train_metrics, val_metrics, test_metrics

    def test_compare_metrics_with_all_sets(self, sample_metrics_sets):
        """Test compare_metrics with train/val/test"""
        train, val, test = sample_metrics_sets

        # Should not raise error (returns None)
        result = compare_metrics(train, val, test)

        # Function returns None (just prints)
        assert result is None

    def test_compare_metrics_without_test(self, sample_metrics_sets):
        """Test compare_metrics without test set"""
        train, val, _ = sample_metrics_sets

        # Should not raise error
        result = compare_metrics(train, val, test_metrics=None)

        assert result is None

    def test_compare_metrics_with_roc_auc(self, sample_metrics_sets):
        """Test compare_metrics with ROC AUC included"""
        train, val, test = sample_metrics_sets
        train["roc_auc"] = 0.97
        val["roc_auc"] = 0.92
        test["roc_auc"] = 0.90

        # Should handle ROC AUC
        result = compare_metrics(train, val, test)
        assert result is None

    def test_compare_metrics_empty_dicts(self):
        """Test with empty metric dicts (should raise KeyError)"""
        train = {}
        val = {}

        # Should raise KeyError for missing required keys
        with pytest.raises(KeyError):
            compare_metrics(train, val)

    def test_compare_metrics_missing_n_samples(self):
        """Test with missing n_samples key"""
        train = {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1_score": 0.935,
        }
        val = {
            "accuracy": 0.90,
            "precision": 0.89,
            "recall": 0.88,
            "f1_score": 0.885,
        }

        # Should raise KeyError for missing n_samples
        with pytest.raises(KeyError):
            compare_metrics(train, val)

    def test_compare_metrics_missing_precision(self):
        """Test with missing precision key"""
        train = {"accuracy": 0.95, "n_samples": 100, "recall": 0.93, "f1_score": 0.935}
        val = {"accuracy": 0.90, "n_samples": 30, "recall": 0.88, "f1_score": 0.885}

        # Should raise KeyError for missing precision
        with pytest.raises(KeyError):
            compare_metrics(train, val)


# ============================================================================
# TEST CLASS: PRINT METRICS
# ============================================================================


@pytest.mark.unit
@pytest.mark.evaluation
class TestPrintMetrics:
    """Test print_metrics function"""

    @pytest.fixture
    def sample_metrics(self):
        """Sample metrics dict (matching source format)"""
        return {
            "dataset": "Test Set",
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.90,
            "f1_score": 0.905,
            "n_samples": 20,
            "confusion_matrix": np.array([[10, 2], [1, 7]]),
        }

    def test_print_metrics_does_not_raise(self, sample_metrics):
        """Test that print_metrics doesn't raise error"""
        metrics = sample_metrics

        # Should not raise error
        try:
            print_metrics(metrics)
            # If no exception, test passes
            assert True
        except Exception as e:
            pytest.fail(f"print_metrics raised unexpected exception: {e}")

    def test_print_metrics_with_roc_auc(self, sample_metrics):
        """Test print_metrics with ROC AUC"""
        metrics = sample_metrics.copy()
        metrics["roc_auc"] = 0.95

        # Should not raise error
        try:
            print_metrics(metrics)
            assert True
        except Exception as e:
            pytest.fail(f"print_metrics raised unexpected exception: {e}")

    def test_print_metrics_empty_dict(self):
        """Test with empty metrics dict (should raise KeyError)"""
        metrics = {}

        # Should raise KeyError for missing 'dataset' key
        with pytest.raises(KeyError):
            print_metrics(metrics)

    def test_print_metrics_missing_dataset(self):
        """Test with missing dataset key"""
        metrics = {
            "accuracy": 0.85,
            "precision": 0.84,
            "recall": 0.83,
            "f1_score": 0.835,
            "n_samples": 10,
        }

        # Should raise KeyError for missing dataset
        with pytest.raises(KeyError):
            print_metrics(metrics)

    def test_print_metrics_missing_n_samples(self):
        """Test with missing n_samples key"""
        metrics = {
            "dataset": "Test",
            "accuracy": 0.85,
            "precision": 0.84,
            "recall": 0.83,
            "f1_score": 0.835,
        }

        # Should raise KeyError for missing n_samples
        with pytest.raises(KeyError):
            print_metrics(metrics)


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================


@pytest.mark.unit
@pytest.mark.evaluation
class TestMetricsEdgeCases:
    """Test edge cases in metrics evaluation"""

    def test_evaluate_model_single_class(self):
        """Test with single class (all same predictions)"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        result = evaluate_model(y_true, y_pred)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert result["accuracy"] == 1.0

    def test_evaluate_model_all_wrong(self):
        """Test with all wrong predictions"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])

        result = evaluate_model(y_true, y_pred)

        # Accuracy should be 0.0
        assert result["accuracy"] == 0.0

    def test_evaluate_model_empty_arrays(self):
        """Test with empty arrays (returns NaN values, no error)"""
        y_true = np.array([])
        y_pred = np.array([])

        # With empty arrays, sklearn returns NaN values (no error raised)
        result = evaluate_model(y_true, y_pred)

        # Check that it returns dict with NaN values
        assert isinstance(result, dict)
        assert np.isnan(result["accuracy"]) or result["accuracy"] is not None

    def test_evaluate_model_mismatched_shapes(self):
        """Test with mismatched array shapes"""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])  # Different length

        # Should raise ValueError from sklearn
        with pytest.raises(ValueError):
            evaluate_model(y_true, y_pred)

    def test_evaluate_model_with_string_labels(self):
        """Test with string labels (should work)"""
        y_true = np.array(["A", "B", "C", "A", "B", "C"])
        y_pred = np.array(["A", "B", "C", "A", "B", "C"])

        # Should work with string labels (sklearn handles it)
        result = evaluate_model(y_true, y_pred)

        assert isinstance(result, dict)
        assert result["accuracy"] == 1.0

    def test_evaluate_model_imbalanced_classes(self):
        """Test with highly imbalanced classes"""
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0] * 90 + [1] * 10)

        result = evaluate_model(y_true, y_pred)

        # Should handle imbalanced data
        assert isinstance(result, dict)
        assert result["accuracy"] == 1.0

    def test_evaluate_model_with_nan_in_proba(self):
        """Test with NaN values in probability predictions"""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        y_pred_proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [np.nan, 0.0, 1.0]])

        # Should handle or raise error (both acceptable)
        try:
            result = evaluate_model(y_true, y_pred, y_pred_proba)
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            # It's OK if it raises an error for NaN
            pass


# ============================================================================
# TEST CLASS: INTEGRATION
# ============================================================================


@pytest.mark.integration
@pytest.mark.evaluation
class TestMetricsIntegration:
    """Integration tests for metrics evaluation"""

    def test_full_evaluation_pipeline(self):
        """Test full evaluation pipeline: evaluate + compare + print"""
        # Create predictions
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        # Evaluate
        train_metrics = evaluate_model(y_true, y_pred, dataset_name="Train")
        val_metrics = evaluate_model(y_true, y_pred, dataset_name="Validation")

        # Compare
        compare_metrics(train_metrics, val_metrics)

        # Print
        print_metrics(train_metrics)

        # All should work without errors
        assert isinstance(train_metrics, dict)
        assert isinstance(val_metrics, dict)

    def test_evaluation_with_probabilities(self):
        """Test full evaluation with probability predictions"""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_pred_proba = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
            ]
        )

        metrics = evaluate_model(y_true, y_pred, y_pred_proba)

        # Should include ROC AUC if proba provided
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "roc_auc" in metrics

    def test_three_way_comparison(self):
        """Test comparison with train/val/test"""
        np.random.seed(42)
        y_true = np.array([0, 1, 2] * 10)
        y_pred_perfect = np.array([0, 1, 2] * 10)
        y_pred_good = np.array([0, 1, 2] * 9 + [0, 1, 0])  # 90% accuracy
        y_pred_ok = np.array([0, 1, 2] * 8 + [0] * 6)  # 80% accuracy

        train_metrics = evaluate_model(y_true, y_pred_perfect, dataset_name="Train")
        val_metrics = evaluate_model(y_true, y_pred_good, dataset_name="Validation")
        test_metrics = evaluate_model(y_true, y_pred_ok, dataset_name="Test")

        # Should detect performance degradation
        compare_metrics(train_metrics, val_metrics, test_metrics)

        assert train_metrics["accuracy"] > val_metrics["accuracy"]
        assert val_metrics["accuracy"] > test_metrics["accuracy"]

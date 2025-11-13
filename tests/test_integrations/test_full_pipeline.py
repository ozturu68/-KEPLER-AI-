"""
Integration tests for full ML pipeline
Testing end-to-end workflows with correct API usage

Author: sulegogh
Date: 2025-11-13
Version: 4.0 (Final - correct API signatures verified)
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessors import MissingValueHandler, SimpleImputer
from src.evaluation.metrics import compare_metrics, evaluate_model
from src.features.engineering import ExoplanetFeatureEngineer
from src.features.scalers import FeatureScaler, scale_train_val_test
from src.features.selection import FeatureSelector, select_features_train_val_test

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_data_with_missing():
    """Sample data with missing values for testing"""
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "feature4": np.random.randn(n_samples),
            "koi_disposition": np.random.choice(["A", "B", "C"], n_samples),
        }
    )

    # Add some missing values
    df.loc[0:10, "feature1"] = np.nan
    df.loc[20:25, "feature2"] = np.nan

    return df


@pytest.fixture
def sample_exoplanet_data():
    """Sample exoplanet-like data for feature engineering"""
    np.random.seed(42)
    n_samples = 50

    return pd.DataFrame(
        {
            "koi_period": np.random.uniform(1, 100, n_samples),
            "koi_prad": np.random.uniform(1, 20, n_samples),
            "koi_teq": np.random.uniform(200, 2000, n_samples),
            "koi_insol": np.random.uniform(0.1, 10, n_samples),
            "koi_steff": np.random.uniform(4000, 7000, n_samples),
            "koi_slogg": np.random.uniform(3.5, 5.0, n_samples),
            "koi_srad": np.random.uniform(0.5, 2.0, n_samples),
            "koi_disposition": np.random.choice(["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"], n_samples),
        }
    )


@pytest.fixture
def sample_train_val_test():
    """Sample train/val/test splits"""
    np.random.seed(42)

    def create_split(n_samples):
        return pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
                "feature4": np.random.randn(n_samples),
                "koi_disposition": np.random.choice(["A", "B", "C"], n_samples),
            }
        )

    train = create_split(60)
    val = create_split(20)
    test = create_split(20)

    return train, val, test


# ============================================================================
# TEST CLASS: DATA PREPROCESSING PIPELINE
# ============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestDataPreprocessingPipeline:
    """Test data preprocessing pipeline"""

    def test_missing_value_handler_pipeline(self, sample_data_with_missing):
        """Test MissingValueHandler in pipeline (correct API: fit_transform)"""
        df = sample_data_with_missing.copy()

        # Handle missing values (correct API: fit_transform, not handle_missing_values)
        handler = MissingValueHandler(numerical_strategy="mean", threshold=0.9)
        filled_df = handler.fit_transform(df)

        # Should have no missing values in numeric columns
        numeric_cols = filled_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            assert filled_df[numeric_cols].isnull().sum().sum() == 0

    def test_simple_imputer_pipeline(self, sample_data_with_missing):
        """Test SimpleImputer in pipeline"""
        df = sample_data_with_missing.copy()

        # Impute missing values
        imputer = SimpleImputer(strategy="mean")
        X = df.drop(columns=["koi_disposition"])

        imputer.fit(X)
        X_imputed_array = imputer.transform(X)  # Returns numpy array

        # Convert to DataFrame for assertions
        X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)

        # Should have no missing values
        assert X_imputed.isnull().sum().sum() == 0

    def test_preprocessing_preserves_row_count(self, sample_data_with_missing):
        """Test that preprocessing preserves row count"""
        df = sample_data_with_missing.copy()
        original_rows = df.shape[0]

        # Preprocess (correct API: fit_transform)
        handler = MissingValueHandler(numerical_strategy="median", threshold=0.9)
        filled_df = handler.fit_transform(df)

        # Row count should be preserved
        assert len(filled_df) == original_rows


# ============================================================================
# TEST CLASS: FEATURE ENGINEERING PIPELINE
# ============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestFeatureEngineeringPipeline:
    """Test feature engineering pipeline"""

    def test_feature_engineering_pipeline(self, sample_exoplanet_data):
        """Test feature engineering with ExoplanetFeatureEngineer"""
        df = sample_exoplanet_data.copy()

        # Engineer features
        engineer = ExoplanetFeatureEngineer()
        engineered_df = engineer.fit_transform(df)

        # Should add new features
        assert len(engineered_df.columns) >= len(df.columns)

    def test_engineering_scaling_pipeline(self, sample_exoplanet_data):
        """Test feature engineering + scaling pipeline"""
        df = sample_exoplanet_data.copy()

        # Step 1: Engineer
        engineer = ExoplanetFeatureEngineer()
        engineered_df = engineer.fit_transform(df)

        # Step 2: Scale (correct API: transform() returns DataFrame but doesn't have keep_target)
        scaler = FeatureScaler(method="standard")

        # Separate target column
        target = engineered_df["koi_disposition"].copy()
        X = engineered_df.drop(columns=["koi_disposition"], errors="ignore")

        if len(X.columns) > 0:
            scaler.fit(X)
            scaled_X = scaler.transform(X)

            # Add target back
            scaled_df = scaled_X.copy()
            scaled_df["koi_disposition"] = target

            # Should be scaled
            assert isinstance(scaled_df, pd.DataFrame)
            assert len(scaled_df) == len(engineered_df)
            assert "koi_disposition" in scaled_df.columns

    def test_engineering_scaling_selection_pipeline(self, sample_exoplanet_data):
        """Test full feature pipeline: engineer → scale → select"""
        df = sample_exoplanet_data.copy()

        # Step 1: Engineer
        engineer = ExoplanetFeatureEngineer()
        engineered_df = engineer.fit_transform(df)

        # Step 2: Scale (separate target, then add back)
        scaler = FeatureScaler(method="minmax")

        # Separate target
        target = engineered_df["koi_disposition"].copy()
        X = engineered_df.drop(columns=["koi_disposition"], errors="ignore")

        if len(X.columns) > 0:
            scaler.fit(X)
            scaled_X = scaler.transform(X)

            # Add target back
            scaled_df = scaled_X.copy()
            scaled_df["koi_disposition"] = target

            # Step 3: Select
            selector = FeatureSelector()
            selected_features, info = selector.select_features(scaled_df, target_col="koi_disposition", n_features=5)

            # Should have selected features
            assert isinstance(selected_features, list)
            assert len(selected_features) > 0


# ============================================================================
# TEST CLASS: SCALING AND SELECTION PIPELINE
# ============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestScalingSelectionPipeline:
    """Test scaling and feature selection pipeline"""

    def test_scale_select_train_val_test(self, sample_train_val_test):
        """Test scale + select with train/val/test splits"""
        train, val, test = sample_train_val_test

        # Step 1: Scale
        train_s, val_s, test_s, scaler = scale_train_val_test(train, val, test, method="standard")

        # Step 2: Select features
        train_f, val_f, test_f, selector, info = select_features_train_val_test(train_s, val_s, test_s, n_features=3)

        # Assertions
        assert len(train_f) == len(train)
        assert len(val_f) == len(val)
        assert len(test_f) == len(test)

        # All splits should have same columns
        assert list(train_f.columns) == list(val_f.columns)
        assert list(val_f.columns) == list(test_f.columns)

    def test_pipeline_consistency_across_splits(self, sample_train_val_test):
        """Test that pipeline maintains consistency across train/val/test"""
        train, val, test = sample_train_val_test

        # Run pipeline
        train_s, val_s, test_s, scaler = scale_train_val_test(train, val, test, method="robust")

        train_f, val_f, test_f, selector, info = select_features_train_val_test(train_s, val_s, test_s, n_features=2)

        # Selected features should be same across splits
        assert list(train_f.columns) == list(val_f.columns) == list(test_f.columns)

    def test_pipeline_with_different_scalers(self, sample_train_val_test):
        """Test pipeline with different scaling methods"""
        train, val, test = sample_train_val_test

        for method in ["standard", "minmax", "robust"]:
            train_s, val_s, test_s, scaler = scale_train_val_test(train, val, test, method=method)

            # Should work with all scalers
            assert isinstance(train_s, pd.DataFrame)
            assert len(train_s) == len(train)


# ============================================================================
# TEST CLASS: MODEL EVALUATION PIPELINE
# ============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestModelEvaluationPipeline:
    """Test model training and evaluation pipeline"""

    def test_predict_evaluate_pipeline(self):
        """Test prediction + evaluation pipeline"""
        np.random.seed(42)

        # Create synthetic predictions
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        # Evaluate
        metrics = evaluate_model(y_true, y_pred, dataset_name="Test")

        # Assertions
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0

    def test_train_val_test_evaluation_pipeline(self):
        """Test evaluation pipeline with train/val/test"""
        np.random.seed(42)

        # Create predictions
        y_true = np.array([0, 1, 2, 0, 1, 2])

        y_pred_train = np.array([0, 1, 2, 0, 1, 2])  # Perfect
        y_pred_val = np.array([0, 1, 2, 0, 1, 0])  # 1 error
        y_pred_test = np.array([0, 1, 0, 0, 2, 0])  # 2 errors

        # Evaluate all
        train_metrics = evaluate_model(y_true, y_pred_train, dataset_name="Train")
        val_metrics = evaluate_model(y_true, y_pred_val, dataset_name="Validation")
        test_metrics = evaluate_model(y_true, y_pred_test, dataset_name="Test")

        # Compare
        compare_metrics(train_metrics, val_metrics, test_metrics)

        # Assertions
        assert train_metrics["accuracy"] > val_metrics["accuracy"]
        assert val_metrics["accuracy"] > test_metrics["accuracy"]


# ============================================================================
# TEST CLASS: FULL END-TO-END PIPELINE
# ============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
@pytest.mark.slow
class TestFullE2EPipeline:
    """Test complete end-to-end ML pipeline"""

    def test_complete_pipeline_simple(self):
        """Test complete pipeline with simple synthetic data"""
        np.random.seed(42)

        # Step 1: Create data
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "feature4": np.random.randn(100),
                "koi_disposition": np.random.choice(["A", "B", "C"], 100),
            }
        )

        # Step 2: Split data
        train_df = df.iloc[:60]
        val_df = df.iloc[60:80]
        test_df = df.iloc[80:]

        # Step 3: Scale features
        train_s, val_s, test_s, scaler = scale_train_val_test(train_df, val_df, test_df, method="standard")

        # Step 4: Select features
        train_f, val_f, test_f, selector, info = select_features_train_val_test(train_s, val_s, test_s, n_features=3)

        # Step 5: Prepare for model
        y_test = test_f["koi_disposition"].values

        # Step 6: Simple baseline prediction (mode)
        y_train = train_f["koi_disposition"].values
        mode_class = pd.Series(y_train).mode()[0]
        y_pred_test = np.full(len(y_test), mode_class)

        # Step 7: Evaluate
        metrics = evaluate_model(y_test, y_pred_test, dataset_name="Test")

        # Assertions
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics

    def test_pipeline_with_feature_engineering(self, sample_exoplanet_data):
        """Test pipeline with feature engineering included"""
        df = sample_exoplanet_data.copy()

        # Step 1: Engineer features
        engineer = ExoplanetFeatureEngineer()
        engineered_df = engineer.fit_transform(df)

        # Step 2: Scale (correct API: separate target, transform, then add back)
        scaler = FeatureScaler(method="standard")

        # Separate target
        target = engineered_df["koi_disposition"].copy()
        X = engineered_df.drop(columns=["koi_disposition"], errors="ignore")

        if len(X.columns) > 0:
            scaler.fit(X)
            scaled_X = scaler.transform(X)

            # Add target back
            scaled_df = scaled_X.copy()
            scaled_df["koi_disposition"] = target

            # Step 3: Select
            selector = FeatureSelector()
            selected_features, info = selector.select_features(scaled_df, target_col="koi_disposition", n_features=5)

            # Assertions
            assert isinstance(selected_features, list)
            assert len(selected_features) > 0

    def test_pipeline_reproducibility(self):
        """Test that pipeline produces reproducible results"""
        np.random.seed(42)

        # Create data
        df = pd.DataFrame(
            {
                "f1": np.random.randn(50),
                "f2": np.random.randn(50),
                "koi_disposition": np.random.choice(["A", "B"], 50),
            }
        )

        # Run pipeline twice
        results = []
        for _ in range(2):
            scaler = FeatureScaler(method="standard")
            X = df.drop(columns=["koi_disposition"])
            scaler.fit(X)
            scaled_X = scaler.transform(X)
            results.append(scaled_X)

        # Results should be identical
        pd.testing.assert_frame_equal(results[0], results[1])


# ============================================================================
# TEST CLASS: PIPELINE EDGE CASES
# ============================================================================


@pytest.mark.integration
@pytest.mark.pipeline
class TestPipelineEdgeCases:
    """Test edge cases in pipeline"""

    def test_pipeline_with_single_sample(self):
        """Test pipeline with single sample"""
        df = pd.DataFrame({"f1": [1.0], "f2": [2.0], "koi_disposition": ["A"]})

        scaler = FeatureScaler()
        X = df.drop(columns=["koi_disposition"])

        try:
            scaler.fit(X)
            scaled_X = scaler.transform(X)
            assert isinstance(scaled_X, pd.DataFrame)
        except ValueError:
            # May raise error for single sample (expected)
            pass

    def test_pipeline_with_constant_features(self):
        """Test pipeline with constant features"""
        df = pd.DataFrame({"f1": [1.0] * 10, "f2": [2.0] * 10, "koi_disposition": ["A"] * 10})

        scaler = FeatureScaler()
        X = df.drop(columns=["koi_disposition"])
        scaler.fit(X)
        scaled_X = scaler.transform(X)

        assert isinstance(scaled_X, pd.DataFrame)

    def test_pipeline_with_missing_values(self):
        """Test pipeline handles missing values (correct API: numpy array to DataFrame)"""
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, np.nan, 4.0],
                "f2": [10.0, np.nan, 30.0, 40.0],
                "koi_disposition": ["A", "B", "A", "B"],
            }
        )

        # Handle missing first
        imputer = SimpleImputer(strategy="mean")
        X = df.drop(columns=["koi_disposition"])

        imputer.fit(X)
        X_imputed_array = imputer.transform(X)  # Returns numpy array

        # Convert back to DataFrame (IMPORTANT!)
        X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)

        # Then scale
        scaler = FeatureScaler()
        scaler.fit(X_imputed)
        scaled = scaler.transform(X_imputed)

        assert not np.isnan(scaled.values).any()

    def test_pipeline_with_mixed_operations(self):
        """Test pipeline with multiple operations in sequence"""
        np.random.seed(42)

        # Create data with missing values
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, np.nan, 4.0, 5.0],
                "f2": [10.0, np.nan, 30.0, 40.0, 50.0],
                "f3": [100.0, 200.0, 300.0, 400.0, 500.0],
                "koi_disposition": ["A", "B", "C", "A", "B"],
            }
        )

        # Step 1: Impute (correct API: array to DataFrame conversion)
        imputer = SimpleImputer(strategy="mean")
        X = df.drop(columns=["koi_disposition"])
        imputer.fit(X)
        X_imputed_array = imputer.transform(X)  # Returns numpy array

        # Convert back to DataFrame (IMPORTANT!)
        X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)

        # Step 2: Scale
        scaler = FeatureScaler(method="minmax")
        scaler.fit(X_imputed)
        X_scaled = scaler.transform(X_imputed)

        # Add target back for selection
        scaled_df = X_scaled.copy()
        scaled_df["koi_disposition"] = df["koi_disposition"].values

        # Step 3: Select features
        selector = FeatureSelector()
        selected_features, info = selector.select_features(scaled_df, target_col="koi_disposition", n_features=2)

        # Assertions
        assert isinstance(selected_features, list)
        assert len(selected_features) > 0
        assert len(selected_features) <= 2

    def test_pipeline_with_missing_value_handler(self):
        """Test pipeline with MissingValueHandler (correct API)"""
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, np.nan, 4.0, 5.0] * 10,
                "f2": [10.0, np.nan, 30.0, 40.0, 50.0] * 10,
                "koi_disposition": ["A", "B", "C"] * 16 + ["A", "B"],
            }
        )

        # Step 1: Handle missing (correct API: fit_transform)
        handler = MissingValueHandler(numerical_strategy="median", threshold=0.8)
        filled_df = handler.fit_transform(df)

        # Step 2: Scale
        scaler = FeatureScaler(method="standard")
        target = filled_df["koi_disposition"].copy()
        X = filled_df.drop(columns=["koi_disposition"])

        scaler.fit(X)
        scaled_X = scaler.transform(X)

        # Add target back
        scaled_df = scaled_X.copy()
        scaled_df["koi_disposition"] = target

        # Assertions
        assert isinstance(scaled_df, pd.DataFrame)
        assert scaled_df["koi_disposition"].equals(target)
        assert not scaled_X.isnull().any().any()

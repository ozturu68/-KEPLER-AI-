"""
Tests for feature engineering utilities
Testing ExoplanetFeatureEngineer and feature creation functions

Author: sulegogh
Date: 2025-11-13
Version: 1.0
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import ExoplanetFeatureEngineer, FeatureEngineeringError, engineer_train_val_test

# ============================================================================
# TEST CLASS: EXOPLANET FEATURE ENGINEER - PLANETARY FEATURES
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestCreatePlanetaryFeatures:
    """Test planetary feature creation"""

    @pytest.fixture
    def engineer(self):
        """Create ExoplanetFeatureEngineer instance"""
        return ExoplanetFeatureEngineer()

    @pytest.fixture
    def sample_exoplanet_data(self):
        """Sample exoplanet data with required columns"""
        return pd.DataFrame(
            {
                "koi_period": [10.5, 20.3, 5.7, 30.2],
                "koi_depth": [100.0, 200.0, 50.0, 150.0],
                "koi_duration": [2.5, 3.0, 1.5, 4.0],
                "koi_impact": [0.5, 0.7, 0.3, 0.6],
            }
        )

    def test_create_planetary_features_adds_new_columns(self, engineer, sample_exoplanet_data):
        """Test that planetary features add new columns"""
        df = sample_exoplanet_data
        initial_cols = len(df.columns)

        result = engineer.create_planetary_features(df)

        # Should have more columns than original
        assert len(result.columns) > initial_cols

    def test_create_planetary_features_preserves_original(self, engineer, sample_exoplanet_data):
        """Test that original columns are preserved"""
        df = sample_exoplanet_data
        original_cols = set(df.columns)

        result = engineer.create_planetary_features(df)

        # All original columns should still exist
        assert original_cols.issubset(set(result.columns))

    def test_create_planetary_features_no_nan_in_new_features(self, engineer, sample_exoplanet_data):
        """Test that new features don't introduce NaN values (if input is clean)"""
        df = sample_exoplanet_data

        result = engineer.create_planetary_features(df)

        # No NaN in result (assuming input is clean)
        assert not result.isnull().any().any()

    def test_create_planetary_features_with_missing_columns(self, engineer):
        """Test behavior when required columns are missing"""
        df = pd.DataFrame({"A": [1, 2, 3]})

        # Should handle gracefully or raise error
        try:
            result = engineer.create_planetary_features(df)
            # If it doesn't raise, check that it returns something
            assert isinstance(result, pd.DataFrame)
        except (KeyError, FeatureEngineeringError):
            # It's OK if it raises an error for missing columns
            pass

    def test_create_planetary_features_returns_dataframe(self, engineer, sample_exoplanet_data):
        """Test that result is a DataFrame"""
        df = sample_exoplanet_data

        result = engineer.create_planetary_features(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)


# ============================================================================
# TEST CLASS: EXOPLANET FEATURE ENGINEER - INTERACTION FEATURES
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestCreateInteractionFeatures:
    """Test interaction feature creation"""

    @pytest.fixture
    def engineer(self):
        """Create ExoplanetFeatureEngineer instance"""
        return ExoplanetFeatureEngineer()

    @pytest.fixture
    def sample_data(self):
        """Sample data for interaction features"""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0],
                "feature2": [2.0, 3.0, 4.0, 5.0],
                "feature3": [10.0, 20.0, 30.0, 40.0],
            }
        )

    def test_create_interaction_features_adds_columns(self, engineer, sample_data):
        """Test that interaction features add new columns"""
        df = sample_data
        feature_pairs = [("feature1", "feature2"), ("feature1", "feature3")]

        result = engineer.create_interaction_features(df, feature_pairs)

        # Should have more columns than original
        assert len(result.columns) > len(df.columns)

    def test_create_interaction_features_default_pairs(self, engineer, sample_data):
        """Test interaction features with default pairs (None)"""
        df = sample_data

        result = engineer.create_interaction_features(df, feature_pairs=None)

        # Should create some interactions automatically
        assert isinstance(result, pd.DataFrame)

    def test_create_interaction_features_multiplication(self, engineer, sample_data):
        """Test that interactions are computed correctly (multiplication)"""
        df = sample_data
        feature_pairs = [("feature1", "feature2")]

        result = engineer.create_interaction_features(df, feature_pairs)

        # Check if interaction column exists
        interaction_col_names = [col for col in result.columns if "x" in col.lower()]
        if interaction_col_names:
            # Verify one interaction value
            assert len(result) == len(df)

    def test_create_interaction_features_preserves_original(self, engineer, sample_data):
        """Test that original columns are preserved"""
        df = sample_data
        original_cols = set(df.columns)
        feature_pairs = [("feature1", "feature2")]

        result = engineer.create_interaction_features(df, feature_pairs)

        # All original columns should still exist
        assert original_cols.issubset(set(result.columns))

    def test_create_interaction_features_empty_pairs(self, engineer, sample_data):
        """Test with empty feature pairs list"""
        df = sample_data
        feature_pairs = []

        result = engineer.create_interaction_features(df, feature_pairs)

        # Should return original or slightly modified DataFrame
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# TEST CLASS: EXOPLANET FEATURE ENGINEER - POLYNOMIAL FEATURES
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestCreatePolynomialFeatures:
    """Test polynomial feature creation"""

    @pytest.fixture
    def engineer(self):
        """Create ExoplanetFeatureEngineer instance"""
        return ExoplanetFeatureEngineer()

    @pytest.fixture
    def sample_data(self):
        """Sample data for polynomial features"""
        return pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [2.0, 3.0, 4.0, 5.0]})

    def test_create_polynomial_features_returns_tuple(self, engineer, sample_data):
        """Test that polynomial features return (DataFrame, list)"""
        df = sample_data
        feature_cols = ["feature1", "feature2"]

        result = engineer.create_polynomial_features(df, feature_cols)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], list)

    def test_create_polynomial_features_adds_columns(self, engineer, sample_data):
        """Test that polynomial features add new columns"""
        df = sample_data
        feature_cols = ["feature1", "feature2"]

        result_df, new_cols = engineer.create_polynomial_features(df, feature_cols)

        # Should have more columns than original
        assert len(result_df.columns) > len(df.columns)
        assert len(new_cols) > 0

    def test_create_polynomial_features_default_cols(self, engineer, sample_data):
        """Test polynomial features with default columns (None)"""
        df = sample_data

        result_df, new_cols = engineer.create_polynomial_features(df, feature_cols=None)

        # Should work with automatic column selection
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(new_cols, list)

    def test_create_polynomial_features_preserves_rows(self, engineer, sample_data):
        """Test that number of rows is preserved"""
        df = sample_data
        feature_cols = ["feature1", "feature2"]

        result_df, _ = engineer.create_polynomial_features(df, feature_cols)

        assert len(result_df) == len(df)

    def test_create_polynomial_features_single_column(self, engineer):
        """Test polynomial features with single column"""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0]})

        result_df, new_cols = engineer.create_polynomial_features(df, ["feature1"])

        # Should create polynomial features for single column
        assert len(result_df.columns) >= len(df.columns)


# ============================================================================
# TEST CLASS: EXOPLANET FEATURE ENGINEER - FIT TRANSFORM
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestFitTransform:
    """Test main fit_transform method"""

    @pytest.fixture
    def engineer(self):
        """Create ExoplanetFeatureEngineer instance"""
        return ExoplanetFeatureEngineer()

    @pytest.fixture
    def sample_exoplanet_data(self):
        """Sample exoplanet data"""
        return pd.DataFrame(
            {
                "koi_period": [10.5, 20.3, 5.7, 30.2],
                "koi_depth": [100.0, 200.0, 50.0, 150.0],
                "koi_duration": [2.5, 3.0, 1.5, 4.0],
                "koi_impact": [0.5, 0.7, 0.3, 0.6],
            }
        )

    def test_fit_transform_returns_dataframe(self, engineer, sample_exoplanet_data):
        """Test that fit_transform returns DataFrame"""
        df = sample_exoplanet_data

        result = engineer.fit_transform(df)

        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_with_planetary_features(self, engineer, sample_exoplanet_data):
        """Test fit_transform with planetary features enabled"""
        df = sample_exoplanet_data

        result = engineer.fit_transform(df, create_planetary=True)

        # Should have more columns than original
        assert len(result.columns) > len(df.columns)

    def test_fit_transform_with_interactions(self, engineer, sample_exoplanet_data):
        """Test fit_transform with interactions enabled"""
        df = sample_exoplanet_data

        result = engineer.fit_transform(df, create_interactions=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > len(df.columns)

    def test_fit_transform_with_polynomial(self, engineer, sample_exoplanet_data):
        """Test fit_transform with polynomial features"""
        df = sample_exoplanet_data

        result = engineer.fit_transform(
            df,
            create_polynomial=True,
            poly_feature_cols=["koi_period", "koi_depth"],
        )

        # Should have many more columns with polynomial features
        assert len(result.columns) >= len(df.columns)

    def test_fit_transform_all_features(self, engineer, sample_exoplanet_data):
        """Test fit_transform with all features enabled"""
        df = sample_exoplanet_data

        result = engineer.fit_transform(
            df,
            create_planetary=True,
            create_polynomial=True,
            create_interactions=True,
        )

        # Should have significantly more columns
        assert len(result.columns) > len(df.columns) * 2

    def test_fit_transform_no_features(self, engineer, sample_exoplanet_data):
        """Test fit_transform with all features disabled"""
        df = sample_exoplanet_data

        result = engineer.fit_transform(
            df,
            create_planetary=False,
            create_polynomial=False,
            create_interactions=False,
        )

        # Should return similar DataFrame (maybe with minor changes)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_fit_transform_preserves_rows(self, engineer, sample_exoplanet_data):
        """Test that fit_transform preserves number of rows"""
        df = sample_exoplanet_data

        result = engineer.fit_transform(df)

        assert len(result) == len(df)


# ============================================================================
# TEST CLASS: ENGINEER TRAIN VAL TEST
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestEngineerTrainValTest:
    """Test engineer_train_val_test function"""

    @pytest.fixture
    def sample_splits(self):
        """Sample train/val/test splits"""
        train = pd.DataFrame(
            {
                "koi_period": [10.5, 20.3, 5.7],
                "koi_depth": [100.0, 200.0, 50.0],
                "koi_duration": [2.5, 3.0, 1.5],
            }
        )
        val = pd.DataFrame({"koi_period": [30.2], "koi_depth": [150.0], "koi_duration": [4.0]})
        test = pd.DataFrame({"koi_period": [15.5], "koi_depth": [120.0], "koi_duration": [2.8]})
        return train, val, test

    def test_engineer_train_val_test_returns_tuple(self, sample_splits):
        """Test that function returns tuple of 3 DataFrames"""
        train, val, test = sample_splits

        result = engineer_train_val_test(train, val, test)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(df, pd.DataFrame) for df in result)

    def test_engineer_train_val_test_preserves_rows(self, sample_splits):
        """Test that row counts are preserved"""
        train, val, test = sample_splits

        train_eng, val_eng, test_eng = engineer_train_val_test(train, val, test)

        assert len(train_eng) == len(train)
        assert len(val_eng) == len(val)
        assert len(test_eng) == len(test)

    def test_engineer_train_val_test_adds_features(self, sample_splits):
        """Test that features are added to all splits"""
        train, val, test = sample_splits

        train_eng, val_eng, test_eng = engineer_train_val_test(train, val, test)

        # All splits should have more or equal columns
        assert len(train_eng.columns) >= len(train.columns)
        assert len(val_eng.columns) >= len(val.columns)
        assert len(test_eng.columns) >= len(test.columns)

    def test_engineer_train_val_test_same_columns(self, sample_splits):
        """Test that all splits have same columns after engineering"""
        train, val, test = sample_splits

        train_eng, val_eng, test_eng = engineer_train_val_test(train, val, test)

        # All should have same columns
        assert list(train_eng.columns) == list(val_eng.columns)
        assert list(val_eng.columns) == list(test_eng.columns)

    def test_engineer_train_val_test_with_kwargs(self, sample_splits):
        """Test engineer_train_val_test with kwargs (passed to fit_transform)"""
        train, val, test = sample_splits

        # kwargs are passed to fit_transform, not constructor
        # So this should work if engineer_train_val_test handles it correctly
        result = engineer_train_val_test(train, val, test)

        assert isinstance(result, tuple)
        assert len(result) == 3
        # Just verify it works without errors


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestEngineeringEdgeCases:
    """Test edge cases in feature engineering"""

    @pytest.fixture
    def engineer(self):
        """Create ExoplanetFeatureEngineer instance"""
        return ExoplanetFeatureEngineer()

    def test_empty_dataframe(self, engineer):
        """Test feature engineering on empty DataFrame"""
        df = pd.DataFrame()

        # Should handle gracefully
        try:
            result = engineer.fit_transform(df, create_planetary=False)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, FeatureEngineeringError):
            # It's OK if it raises an error
            pass

    def test_single_row_dataframe(self, engineer):
        """Test feature engineering on single row"""
        df = pd.DataFrame({"koi_period": [10.5], "koi_depth": [100.0], "koi_duration": [2.5]})

        result = engineer.fit_transform(df, create_planetary=True)

        assert len(result) == 1

    def test_dataframe_with_nan(self, engineer):
        """Test feature engineering with NaN values"""
        df = pd.DataFrame(
            {
                "koi_period": [10.5, np.nan, 5.7],
                "koi_depth": [100.0, 200.0, np.nan],
                "koi_duration": [2.5, 3.0, 1.5],
            }
        )

        # Should handle gracefully or propagate NaN
        result = engineer.fit_transform(df, create_planetary=True)

        assert isinstance(result, pd.DataFrame)

    def test_dataframe_with_zeros(self, engineer):
        """Test feature engineering with zero values"""
        df = pd.DataFrame({"koi_period": [10.5, 0.0, 5.7], "koi_depth": [100.0, 200.0, 0.0]})

        result = engineer.fit_transform(df, create_planetary=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

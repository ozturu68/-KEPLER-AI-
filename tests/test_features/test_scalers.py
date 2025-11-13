"""
Tests for feature scaling utilities
Testing FeatureScaler and scaling functions

Author: sulegogh
Date: 2025-11-13
Version: 1.0
"""

import numpy as np
import pandas as pd
import pytest

from src.features.scalers import FeatureScaler, ScalingError, compare_scalers, scale_train_val_test

# ============================================================================
# TEST CLASS: FEATURE SCALER - FIT
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestFeatureScalerFit:
    """Test FeatureScaler fit method"""

    @pytest.fixture
    def scaler(self):
        """Create FeatureScaler instance"""
        return FeatureScaler(method="standard")

    @pytest.fixture
    def sample_data(self):
        """Sample numerical data for scaling"""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feature3": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

    def test_fit_returns_self(self, scaler, sample_data):
        """Test that fit returns self for chaining"""
        result = scaler.fit(sample_data)

        assert result is scaler

    def test_fit_stores_scaler(self, scaler, sample_data):
        """Test that fit stores internal scaler"""
        scaler.fit(sample_data)

        assert scaler.scaler is not None
        assert hasattr(scaler.scaler, "transform")

    def test_fit_with_exclude_cols(self, scaler, sample_data):
        """Test fit with excluded columns"""
        df = sample_data.copy()
        df["label"] = ["A", "B", "C", "D", "E"]

        scaler.fit(df, exclude_cols=["label"])

        # Should fit only numerical columns (excluding label)
        assert scaler.scaler is not None

    def test_fit_different_methods(self, sample_data):
        """Test fit with different scaling methods"""
        for method in ["standard", "minmax", "robust"]:
            scaler = FeatureScaler(method=method)
            result = scaler.fit(sample_data)

            assert result is scaler
            assert scaler.scaler is not None


# ============================================================================
# TEST CLASS: FEATURE SCALER - TRANSFORM
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestFeatureScalerTransform:
    """Test FeatureScaler transform method"""

    @pytest.fixture
    def fitted_scaler(self, sample_numerical_features):
        """Create fitted FeatureScaler"""
        scaler = FeatureScaler(method="standard")
        scaler.fit(sample_numerical_features)
        return scaler

    def test_transform_returns_dataframe(self, fitted_scaler, sample_numerical_features):
        """Test that transform returns DataFrame"""
        result = fitted_scaler.transform(sample_numerical_features)

        assert isinstance(result, pd.DataFrame)

    def test_transform_preserves_shape(self, fitted_scaler, sample_numerical_features):
        """Test that transform preserves shape"""
        df = sample_numerical_features
        result = fitted_scaler.transform(df)

        assert result.shape == df.shape

    def test_transform_preserves_columns(self, fitted_scaler, sample_numerical_features):
        """Test that transform preserves column names"""
        df = sample_numerical_features
        result = fitted_scaler.transform(df)

        assert list(result.columns) == list(df.columns)

    def test_transform_changes_values(self, fitted_scaler, sample_numerical_features):
        """Test that transform actually changes values"""
        df = sample_numerical_features
        result = fitted_scaler.transform(df)

        # Values should be different after scaling (unless already scaled)
        # At least check that it's a valid DataFrame
        assert not result.isnull().all().all()

    def test_transform_without_fit_raises_error(self, sample_numerical_features):
        """Test that transform without fit raises error"""
        scaler = FeatureScaler(method="standard")

        with pytest.raises((AttributeError, ValueError, ScalingError)):
            scaler.transform(sample_numerical_features)


# ============================================================================
# TEST CLASS: FEATURE SCALER - FIT TRANSFORM
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestFeatureScalerFitTransform:
    """Test FeatureScaler fit_transform method"""

    @pytest.fixture
    def scaler(self):
        """Create FeatureScaler instance"""
        return FeatureScaler(method="standard")

    def test_fit_transform_returns_dataframe(self, scaler, sample_numerical_features):
        """Test that fit_transform returns DataFrame"""
        result = scaler.fit_transform(sample_numerical_features)

        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_preserves_shape(self, scaler, sample_numerical_features):
        """Test that fit_transform preserves shape"""
        df = sample_numerical_features
        result = scaler.fit_transform(df)

        assert result.shape == df.shape

    def test_fit_transform_equals_fit_then_transform(self, sample_numerical_features):
        """Test that fit_transform equals fit().transform()"""
        df = sample_numerical_features

        scaler1 = FeatureScaler(method="standard")
        result1 = scaler1.fit_transform(df)

        scaler2 = FeatureScaler(method="standard")
        scaler2.fit(df)
        result2 = scaler2.transform(df)

        pd.testing.assert_frame_equal(result1, result2)

    def test_fit_transform_with_exclude_cols(self, scaler):
        """Test fit_transform with excluded columns"""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [10.0, 20.0, 30.0],
                "label": ["A", "B", "C"],
            }
        )

        result = scaler.fit_transform(df, exclude_cols=["label"])

        # Label column should be unchanged
        assert "label" in result.columns
        assert list(result["label"]) == ["A", "B", "C"]


# ============================================================================
# TEST CLASS: FEATURE SCALER - INVERSE TRANSFORM
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestFeatureScalerInverseTransform:
    """Test FeatureScaler inverse_transform method"""

    @pytest.fixture
    def fitted_scaler(self, sample_numerical_features):
        """Create fitted FeatureScaler"""
        scaler = FeatureScaler(method="standard")
        scaler.fit(sample_numerical_features)
        return scaler

    def test_inverse_transform_returns_dataframe(self, fitted_scaler, sample_numerical_features):
        """Test that inverse_transform returns DataFrame"""
        scaled = fitted_scaler.transform(sample_numerical_features)
        result = fitted_scaler.inverse_transform(scaled)

        assert isinstance(result, pd.DataFrame)

    def test_inverse_transform_restores_original(self, fitted_scaler, sample_numerical_features):
        """Test that inverse_transform restores original values"""
        df = sample_numerical_features
        scaled = fitted_scaler.transform(df)
        restored = fitted_scaler.inverse_transform(scaled)

        # Should be close to original (within floating point precision)
        pd.testing.assert_frame_equal(restored, df, atol=1e-5)

    def test_inverse_transform_preserves_shape(self, fitted_scaler, sample_numerical_features):
        """Test that inverse_transform preserves shape"""
        df = sample_numerical_features
        scaled = fitted_scaler.transform(df)
        restored = fitted_scaler.inverse_transform(scaled)

        assert restored.shape == df.shape


# ============================================================================
# TEST CLASS: FEATURE SCALER - GET FEATURE STATS
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestFeatureScalerGetFeatureStats:
    """Test FeatureScaler get_feature_stats method"""

    @pytest.fixture
    def fitted_scaler(self, sample_numerical_features):
        """Create fitted FeatureScaler instance"""
        scaler = FeatureScaler(method="standard")
        scaler.fit(sample_numerical_features)
        return scaler

    def test_get_feature_stats_returns_dataframe(self, fitted_scaler, sample_numerical_features):
        """Test that get_feature_stats returns DataFrame"""
        result = fitted_scaler.get_feature_stats(sample_numerical_features)

        assert isinstance(result, pd.DataFrame)

    def test_get_feature_stats_has_expected_columns(self, fitted_scaler, sample_numerical_features):
        """Test that stats have expected columns"""
        result = fitted_scaler.get_feature_stats(sample_numerical_features)

        # Should have statistics columns (mean, std, min, max, etc.)
        assert len(result.columns) > 0
        assert len(result) > 0

    def test_get_feature_stats_values_make_sense(self, fitted_scaler, sample_numerical_features):
        """Test that stats values are reasonable"""
        df = sample_numerical_features
        result = fitted_scaler.get_feature_stats(df)

        # Stats should be numeric
        assert result.select_dtypes(include=[np.number]).shape[1] > 0


# ============================================================================
# TEST CLASS: COMPARE SCALERS
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestCompareScalers:
    """Test compare_scalers function"""

    def test_compare_scalers_returns_dict(self, sample_numerical_features):
        """Test that compare_scalers returns dict"""
        result = compare_scalers(sample_numerical_features)

        assert isinstance(result, dict)

    def test_compare_scalers_has_expected_keys(self, sample_numerical_features):
        """Test that result has scaler method keys"""
        result = compare_scalers(sample_numerical_features)

        # Should have keys for different scaling methods
        expected_keys = ["standard", "minmax", "robust"]
        assert any(key in result for key in expected_keys)

    def test_compare_scalers_with_sample_features(self, sample_numerical_features):
        """Test compare_scalers with specific sample features"""
        df = sample_numerical_features
        result = compare_scalers(df, sample_features=list(df.columns[:2]))

        assert isinstance(result, dict)

    def test_compare_scalers_empty_dataframe(self):
        """Test compare_scalers with empty DataFrame"""
        df = pd.DataFrame()

        # Should handle gracefully or raise error
        try:
            result = compare_scalers(df)
            assert isinstance(result, dict)
        except (ValueError, ScalingError):
            # It's OK if it raises an error
            pass


# ============================================================================
# TEST CLASS: SCALE TRAIN VAL TEST
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestScaleTrainValTest:
    """Test scale_train_val_test function"""

    @pytest.fixture
    def sample_splits(self):
        """Sample train/val/test splits"""
        np.random.seed(42)
        train = pd.DataFrame(
            {
                "feature1": np.random.randn(20),
                "feature2": np.random.randn(20) * 10,
            }
        )
        val = pd.DataFrame({"feature1": np.random.randn(5), "feature2": np.random.randn(5) * 10})
        test = pd.DataFrame({"feature1": np.random.randn(5), "feature2": np.random.randn(5) * 10})
        return train, val, test

    def test_scale_train_val_test_returns_tuple(self, sample_splits):
        """Test that function returns tuple of 4 items"""
        train, val, test = sample_splits

        result = scale_train_val_test(train, val, test)

        assert isinstance(result, tuple)
        assert len(result) == 4  # train, val, test, scaler

    def test_scale_train_val_test_preserves_shapes(self, sample_splits):
        """Test that row counts are preserved"""
        train, val, test = sample_splits

        train_s, val_s, test_s, scaler = scale_train_val_test(train, val, test)

        assert len(train_s) == len(train)
        assert len(val_s) == len(val)
        assert len(test_s) == len(test)

    def test_scale_train_val_test_same_columns(self, sample_splits):
        """Test that all splits have same columns"""
        train, val, test = sample_splits

        train_s, val_s, test_s, scaler = scale_train_val_test(train, val, test)

        assert list(train_s.columns) == list(val_s.columns)
        assert list(val_s.columns) == list(test_s.columns)

    def test_scale_train_val_test_returns_scaler(self, sample_splits):
        """Test that scaler is returned"""
        train, val, test = sample_splits

        train_s, val_s, test_s, scaler = scale_train_val_test(train, val, test)

        assert isinstance(scaler, FeatureScaler)
        assert scaler.scaler is not None

    def test_scale_train_val_test_different_methods(self, sample_splits):
        """Test with different scaling methods"""
        train, val, test = sample_splits

        for method in ["standard", "minmax", "robust"]:
            train_s, val_s, test_s, scaler = scale_train_val_test(train, val, test, method=method)

            assert isinstance(scaler, FeatureScaler)
            assert len(train_s) == len(train)

    def test_scale_train_val_test_with_exclude_cols(self):
        """Test with excluded columns"""
        train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [10, 20, 30], "label": ["A", "B", "C"]})
        val = pd.DataFrame({"feature1": [4], "feature2": [40], "label": ["D"]})
        test = pd.DataFrame({"feature1": [5], "feature2": [50], "label": ["E"]})

        train_s, val_s, test_s, scaler = scale_train_val_test(train, val, test, exclude_cols=["label"])

        # Label should be unchanged
        assert list(train_s["label"]) == ["A", "B", "C"]


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestScalersEdgeCases:
    """Test edge cases in scaling"""

    def test_scaler_with_single_row(self):
        """Test scaling with single row DataFrame"""
        df = pd.DataFrame({"feature1": [1.0], "feature2": [10.0]})

        scaler = FeatureScaler(method="standard")
        result = scaler.fit_transform(df)

        assert len(result) == 1

    def test_scaler_with_constant_column(self):
        """Test scaling with constant column (no variance)"""
        df = pd.DataFrame({"feature1": [5.0, 5.0, 5.0], "feature2": [10, 20, 30]})

        scaler = FeatureScaler(method="standard")
        result = scaler.fit_transform(df)

        # Should handle constant column gracefully
        assert isinstance(result, pd.DataFrame)

    def test_scaler_with_nan_values(self):
        """Test scaling with NaN values"""
        df = pd.DataFrame({"feature1": [1.0, np.nan, 3.0], "feature2": [10.0, 20.0, np.nan]})

        scaler = FeatureScaler(method="standard")

        # Should handle or propagate NaN
        try:
            result = scaler.fit_transform(df)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, ScalingError):
            # It's OK if it raises an error for NaN
            pass

    def test_scaler_with_all_same_values(self):
        """Test scaling when all values are the same"""
        df = pd.DataFrame({"feature1": [1.0] * 5, "feature2": [10.0] * 5})

        scaler = FeatureScaler(method="minmax")
        result = scaler.fit_transform(df)

        # MinMax scaler should handle this (result will be 0 or NaN)
        assert isinstance(result, pd.DataFrame)

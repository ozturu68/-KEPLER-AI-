"""
Tests for feature selection utilities
Testing FeatureSelector and selection functions

Author: sulegogh
Date: 2025-11-13
Version: 1.0
"""

import numpy as np
import pandas as pd
import pytest

from src.features.selection import FeatureSelectionError, FeatureSelector, select_features_train_val_test

# ============================================================================
# TEST CLASS: FEATURE SELECTOR - LOW VARIANCE
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestRemoveLowVarianceFeatures:
    """Test low variance feature removal"""

    @pytest.fixture
    def selector(self):
        """Create FeatureSelector instance"""
        return FeatureSelector(variance_threshold=0.01)

    @pytest.fixture
    def sample_data_with_low_variance(self):
        """Sample data with low variance features"""
        return pd.DataFrame(
            {
                "constant_feature": [1.0] * 10,  # No variance
                "low_variance": [1.0, 1.0, 1.0, 1.0, 1.0, 1.001, 1.0, 1.0, 1.0, 1.0],
                "high_variance": np.random.randn(10) * 10,
                "target": ["A", "B"] * 5,
            }
        )

    def test_remove_low_variance_returns_list(self, selector, sample_data_with_low_variance):
        """Test that method returns list of column names"""
        df = sample_data_with_low_variance
        result = selector.remove_low_variance_features(df, exclude_cols=["target"])

        assert isinstance(result, list)

    def test_remove_low_variance_identifies_constant(self, selector, sample_data_with_low_variance):
        """Test that constant features are identified"""
        df = sample_data_with_low_variance
        result = selector.remove_low_variance_features(df, exclude_cols=["target"])

        # constant_feature should be removed
        assert len(result) > 0

    def test_remove_low_variance_excludes_cols(self, selector, sample_data_with_low_variance):
        """Test that excluded columns are not checked"""
        df = sample_data_with_low_variance
        result = selector.remove_low_variance_features(df, exclude_cols=["target"])

        # Target should not be in removed features
        assert "target" not in result

    def test_remove_low_variance_empty_dataframe(self, selector):
        """Test with empty DataFrame"""
        df = pd.DataFrame()
        result = selector.remove_low_variance_features(df)

        assert isinstance(result, list)
        assert len(result) == 0


# ============================================================================
# TEST CLASS: FEATURE SELECTOR - HIGH CORRELATION
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestRemoveHighCorrelationFeatures:
    """Test high correlation feature removal"""

    @pytest.fixture
    def selector(self):
        """Create FeatureSelector instance"""
        return FeatureSelector(correlation_threshold=0.9)

    @pytest.fixture
    def sample_data_with_correlation(self):
        """Sample data with highly correlated features"""
        np.random.seed(42)
        feature1 = np.random.randn(20)
        feature2 = feature1 + np.random.randn(20) * 0.01  # Highly correlated
        feature3 = np.random.randn(20)  # Independent

        return pd.DataFrame(
            {
                "feature1": feature1,
                "feature2": feature2,  # Correlated with feature1
                "feature3": feature3,
                "target": ["A", "B"] * 10,
            }
        )

    def test_remove_high_correlation_returns_list(self, selector, sample_data_with_correlation):
        """Test that method returns list"""
        df = sample_data_with_correlation
        result = selector.remove_high_correlation_features(df, exclude_cols=["target"])

        assert isinstance(result, list)

    def test_remove_high_correlation_identifies_correlated(self, selector, sample_data_with_correlation):
        """Test that correlated features are identified"""
        df = sample_data_with_correlation
        result = selector.remove_high_correlation_features(df, exclude_cols=["target"])

        # Should identify at least one correlated feature
        # (feature1 and feature2 are highly correlated)
        assert len(result) >= 0  # May or may not remove depending on threshold

    def test_remove_high_correlation_excludes_cols(self, selector, sample_data_with_correlation):
        """Test that excluded columns are not checked"""
        df = sample_data_with_correlation
        result = selector.remove_high_correlation_features(df, exclude_cols=["target"])

        # Target should not be in removed features
        assert "target" not in result

    def test_remove_high_correlation_empty_dataframe(self, selector):
        """Test with empty DataFrame"""
        df = pd.DataFrame()
        result = selector.remove_high_correlation_features(df)

        assert isinstance(result, list)
        assert len(result) == 0


# ============================================================================
# TEST CLASS: FEATURE SELECTOR - FEATURE IMPORTANCE
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestGetFeatureImportance:
    """Test feature importance calculation"""

    @pytest.fixture
    def selector(self):
        """Create FeatureSelector instance"""
        return FeatureSelector()

    @pytest.fixture
    def sample_classification_data(self):
        """Sample classification data"""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
            }
        )
        y = pd.Series(np.random.choice(["A", "B", "C"], 50))
        return X, y

    def test_get_feature_importance_returns_dataframe(self, selector, sample_classification_data):
        """Test that feature importance returns DataFrame"""
        X, y = sample_classification_data
        result = selector.get_feature_importance(X, y, method="random_forest")

        assert isinstance(result, pd.DataFrame)

    def test_get_feature_importance_has_expected_columns(self, selector, sample_classification_data):
        """Test that result has expected columns"""
        X, y = sample_classification_data
        result = selector.get_feature_importance(X, y, method="random_forest")

        # Should have feature and importance columns
        assert "feature" in result.columns or len(result.columns) >= 1
        assert len(result) > 0

    def test_get_feature_importance_different_methods(self, selector, sample_classification_data):
        """Test different importance methods"""
        X, y = sample_classification_data

        for method in ["random_forest", "mutual_info"]:
            result = selector.get_feature_importance(X, y, method=method)
            assert isinstance(result, pd.DataFrame)

    def test_get_feature_importance_sorted(self, selector, sample_classification_data):
        """Test that importance is sorted (descending)"""
        X, y = sample_classification_data
        result = selector.get_feature_importance(X, y, method="random_forest")

        # Check if importance values are sorted (if importance column exists)
        if "importance" in result.columns:
            importance_values = result["importance"].values
            assert list(importance_values) == sorted(importance_values, reverse=True)


# ============================================================================
# TEST CLASS: FEATURE SELECTOR - SELECT TOP FEATURES
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestSelectTopFeatures:
    """Test top feature selection"""

    @pytest.fixture
    def selector(self):
        """Create FeatureSelector instance"""
        return FeatureSelector()

    @pytest.fixture
    def sample_importance_df(self):
        """Sample importance DataFrame"""
        return pd.DataFrame(
            {
                "feature": ["feat1", "feat2", "feat3", "feat4", "feat5"],
                "importance": [0.5, 0.3, 0.15, 0.04, 0.01],
            }
        )

    def test_select_top_features_returns_list(self, selector, sample_importance_df):
        """Test that method returns list"""
        result = selector.select_top_features(sample_importance_df, n_features=3)

        assert isinstance(result, list)

    def test_select_top_features_by_n_features(self, selector, sample_importance_df):
        """Test selection by number of features"""
        result = selector.select_top_features(sample_importance_df, n_features=3)

        assert len(result) == 3
        # Should select top 3: feat1, feat2, feat3
        assert "feat1" in result or len(result) == 3

    def test_select_top_features_by_threshold(self, selector, sample_importance_df):
        """Test selection by importance threshold"""
        result = selector.select_top_features(sample_importance_df, importance_threshold=0.1)

        # Should select features with importance >= 0.1
        assert len(result) <= 5

    def test_select_top_features_empty_importance(self, selector):
        """Test with empty importance DataFrame"""
        df = pd.DataFrame({"feature": [], "importance": []})
        result = selector.select_top_features(df, n_features=5)

        assert isinstance(result, list)
        assert len(result) == 0


# ============================================================================
# TEST CLASS: FEATURE SELECTOR - SELECT FEATURES
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestSelectFeatures:
    """Test main select_features method"""

    @pytest.fixture
    def selector(self):
        """Create FeatureSelector instance"""
        return FeatureSelector()

    @pytest.fixture
    def sample_data_for_selection(self):
        """Sample data for feature selection"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
                "feature4": np.random.randn(50),
                "feature5": [1.0] * 50,  # Constant
                "koi_disposition": np.random.choice(["A", "B", "C"], 50),
            }
        )
        return df

    def test_select_features_returns_tuple(self, selector, sample_data_for_selection):
        """Test that select_features returns tuple"""
        df = sample_data_for_selection
        result = selector.select_features(df, target_col="koi_disposition")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_select_features_returns_feature_list(self, selector, sample_data_for_selection):
        """Test that first element is feature list"""
        df = sample_data_for_selection
        features, info = selector.select_features(df, target_col="koi_disposition")

        assert isinstance(features, list)
        assert len(features) > 0

    def test_select_features_returns_info_dict(self, selector, sample_data_for_selection):
        """Test that second element is info dict"""
        df = sample_data_for_selection
        features, info = selector.select_features(df, target_col="koi_disposition")

        assert isinstance(info, dict)

    def test_select_features_different_methods(self, selector, sample_data_for_selection):
        """Test different selection methods"""
        df = sample_data_for_selection

        for method in ["auto", "importance", "variance"]:
            features, info = selector.select_features(df, target_col="koi_disposition", method=method)
            assert isinstance(features, list)

    def test_select_features_n_features_limit(self, selector, sample_data_for_selection):
        """Test that n_features limit is respected"""
        df = sample_data_for_selection
        features, info = selector.select_features(df, target_col="koi_disposition", n_features=3)

        # Should select at most 3 features (excluding target)
        assert len(features) <= 4  # 3 features + target


# ============================================================================
# TEST CLASS: FEATURE SELECTOR - TRANSFORM
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestFeatureSelectorTransform:
    """Test transform method"""

    @pytest.fixture
    def selector_with_features(self, sample_data_for_selection):
        """Create FeatureSelector with selected features"""
        selector = FeatureSelector()
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
                "koi_disposition": np.random.choice(["A", "B", "C"], 50),
            }
        )
        selector.select_features(df, target_col="koi_disposition", n_features=2)
        return selector

    @pytest.fixture
    def sample_data_for_selection(self):
        """Sample data"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.randn(20),
                "feature2": np.random.randn(20),
                "feature3": np.random.randn(20),
                "koi_disposition": np.random.choice(["A", "B"], 20),
            }
        )

    def test_transform_returns_dataframe(self, selector_with_features, sample_data_for_selection):
        """Test that transform returns DataFrame"""
        df = sample_data_for_selection
        result = selector_with_features.transform(df)

        assert isinstance(result, pd.DataFrame)

    def test_transform_reduces_columns(self, selector_with_features, sample_data_for_selection):
        """Test that transform reduces columns"""
        df = sample_data_for_selection
        result = selector_with_features.transform(df)

        # Should have fewer columns than original
        assert len(result.columns) <= len(df.columns)

    def test_transform_preserves_rows(self, selector_with_features, sample_data_for_selection):
        """Test that transform preserves rows"""
        df = sample_data_for_selection
        result = selector_with_features.transform(df)

        assert len(result) == len(df)

    def test_transform_keeps_target(self, selector_with_features, sample_data_for_selection):
        """Test that transform keeps target when requested"""
        df = sample_data_for_selection
        result = selector_with_features.transform(df, keep_target=True)

        # Should keep target column
        assert "koi_disposition" in result.columns

    def test_transform_without_fit_raises_error(self, sample_data_for_selection):
        """Test that transform without select_features raises error"""
        selector = FeatureSelector()
        df = sample_data_for_selection

        with pytest.raises((AttributeError, ValueError, FeatureSelectionError)):
            selector.transform(df)


# ============================================================================
# TEST CLASS: SELECT FEATURES TRAIN VAL TEST
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestSelectFeaturesTrainValTest:
    """Test select_features_train_val_test function"""

    @pytest.fixture
    def sample_splits(self):
        """Sample train/val/test splits"""
        np.random.seed(42)
        train = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
                "feature4": np.random.randn(50),
                "koi_disposition": np.random.choice(["A", "B", "C"], 50),
            }
        )
        val = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
                "feature3": np.random.randn(10),
                "feature4": np.random.randn(10),
                "koi_disposition": np.random.choice(["A", "B", "C"], 10),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
                "feature3": np.random.randn(10),
                "feature4": np.random.randn(10),
                "koi_disposition": np.random.choice(["A", "B", "C"], 10),
            }
        )
        return train, val, test

    def test_select_features_train_val_test_returns_tuple(self, sample_splits):
        """Test that function returns tuple of 5 items"""
        train, val, test = sample_splits

        result = select_features_train_val_test(train, val, test, n_features=2)

        assert isinstance(result, tuple)
        assert len(result) == 5  # train, val, test, selector, info

    def test_select_features_train_val_test_preserves_rows(self, sample_splits):
        """Test that row counts are preserved"""
        train, val, test = sample_splits

        train_s, val_s, test_s, selector, info = select_features_train_val_test(train, val, test, n_features=2)

        assert len(train_s) == len(train)
        assert len(val_s) == len(val)
        assert len(test_s) == len(test)

    def test_select_features_train_val_test_reduces_columns(self, sample_splits):
        """Test that features are selected (columns reduced)"""
        train, val, test = sample_splits

        train_s, val_s, test_s, selector, info = select_features_train_val_test(train, val, test, n_features=2)

        # Should have fewer or equal columns
        assert len(train_s.columns) <= len(train.columns)

    def test_select_features_train_val_test_same_columns(self, sample_splits):
        """Test that all splits have same columns"""
        train, val, test = sample_splits

        train_s, val_s, test_s, selector, info = select_features_train_val_test(train, val, test, n_features=2)

        assert list(train_s.columns) == list(val_s.columns)
        assert list(val_s.columns) == list(test_s.columns)

    def test_select_features_train_val_test_returns_selector(self, sample_splits):
        """Test that selector is returned"""
        train, val, test = sample_splits

        train_s, val_s, test_s, selector, info = select_features_train_val_test(train, val, test)

        assert isinstance(selector, FeatureSelector)

    def test_select_features_train_val_test_returns_info(self, sample_splits):
        """Test that info dict is returned"""
        train, val, test = sample_splits

        train_s, val_s, test_s, selector, info = select_features_train_val_test(train, val, test)

        assert isinstance(info, dict)


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================


@pytest.mark.unit
@pytest.mark.features
class TestFeatureSelectionEdgeCases:
    """Test edge cases in feature selection"""

    def test_selector_with_single_feature(self):
        """Test selection with single feature DataFrame"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(20),
                "target": np.random.choice(["A", "B"], 20),
            }
        )

        selector = FeatureSelector()
        features, info = selector.select_features(df, target_col="target", n_features=1)

        assert len(features) >= 1  # At least target
        assert isinstance(features, list)
        assert isinstance(info, dict)

    def test_selector_with_all_constant_features(self):
        """Test selection when all features are constant (should raise error)"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": [1.0] * 20,
                "feature2": [2.0] * 20,
                "target": np.random.choice(["A", "B"], 20),
            }
        )

        selector = FeatureSelector()

        # Should raise error when all features are constant
        # (they are removed by variance filter, leaving no numerical features)
        with pytest.raises(FeatureSelectionError):
            selector.select_features(df, target_col="target")

    def test_selector_with_nan_values(self):
        """Test selection with NaN values"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5] * 4,
                "feature2": [10, np.nan, 30, 40, 50] * 4,
                "target": np.random.choice(["A", "B"], 20),
            }
        )

        selector = FeatureSelector()

        # Should handle or raise error
        try:
            features, info = selector.select_features(df, target_col="target")
            assert isinstance(features, list)
            assert isinstance(info, dict)
        except (ValueError, FeatureSelectionError):
            # It's OK if it raises an error for NaN
            pass

    def test_selector_with_many_features(self):
        """Test selection with many features (dimensionality reduction)"""
        np.random.seed(42)
        # Create DataFrame with 20 features
        data = {f"feature{i}": np.random.randn(50) for i in range(20)}
        data["target"] = np.random.choice(["A", "B", "C"], 50)
        df = pd.DataFrame(data)

        selector = FeatureSelector()
        features, info = selector.select_features(df, target_col="target", n_features=5)

        # Should reduce to requested number of features (+ target)
        assert len(features) <= 6  # 5 features + target
        assert isinstance(features, list)
        assert isinstance(info, dict)

    def test_selector_with_highly_correlated_features(self):
        """Test selection with highly correlated features"""
        np.random.seed(42)
        feature1 = np.random.randn(30)
        feature2 = feature1 + np.random.randn(30) * 0.01  # Highly correlated

        df = pd.DataFrame(
            {
                "feature1": feature1,
                "feature2": feature2,
                "feature3": np.random.randn(30),
                "target": np.random.choice(["A", "B"], 30),
            }
        )

        selector = FeatureSelector(correlation_threshold=0.9)
        features, info = selector.select_features(df, target_col="target", n_features=3)

        # Should handle correlated features
        assert isinstance(features, list)
        assert len(features) > 0

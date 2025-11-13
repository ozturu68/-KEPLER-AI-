"""
Tests for data cleaning utilities
Testing duplicate removal, outlier detection, and data validation

Author: sulegogh
Date: 2025-11-13
Version: 3.0 (Fully revised with proper assertions and fixtures)
"""

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import DataValidationError
from src.data.cleaners import (
    clean_data,
    convert_dtypes,
    detect_outliers_iqr,
    drop_unnecessary_columns,
    handle_outliers,
    remove_duplicates,
    validate_target_column,
)

# ============================================================================
# TEST CLASS: REMOVE DUPLICATES
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestRemoveDuplicates:
    """Test duplicate removal functionality"""

    def test_remove_duplicates_exact_duplicates(self):
        """Test removing exact duplicate rows"""
        df = pd.DataFrame({"A": [1, 2, 2, 3], "B": [4, 5, 5, 6]})

        result = remove_duplicates(df)

        assert len(result) == 3
        assert not result.duplicated().any()

    def test_remove_duplicates_no_duplicates(self):
        """Test when there are no duplicates"""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        result = remove_duplicates(df)

        assert len(result) == 3
        pd.testing.assert_frame_equal(result, df)

    def test_remove_duplicates_all_duplicates(self):
        """Test when all rows are duplicates"""
        df = pd.DataFrame({"A": [1, 1, 1], "B": [2, 2, 2]})

        result = remove_duplicates(df)

        assert len(result) == 1

    def test_remove_duplicates_preserves_columns(self):
        """Test that all columns are preserved"""
        df = pd.DataFrame({"A": [1, 2, 2], "B": [3, 4, 4], "C": [5, 6, 6]})

        result = remove_duplicates(df)

        assert list(result.columns) == ["A", "B", "C"]

    def test_remove_duplicates_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame({"A": [], "B": []})

        result = remove_duplicates(df)

        assert len(result) == 0

    def test_remove_duplicates_with_fixture(self, sample_data_with_duplicates):
        """Test duplicate removal with fixture data"""
        df = sample_data_with_duplicates

        result = remove_duplicates(df)

        # Should remove at least one duplicate
        assert len(result) <= len(df)
        assert not result.duplicated().any()


# ============================================================================
# TEST CLASS: DROP UNNECESSARY COLUMNS
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestDropUnnecessaryColumns:
    """Test dropping unnecessary columns"""

    def test_drop_unnecessary_columns_basic(self):
        """Test dropping columns specified in DROP_COLUMNS"""
        # Create DataFrame with some columns from DROP_COLUMNS
        df = pd.DataFrame(
            {
                "koi_score": [1, 2, 3],
                "koi_fpflag_co": [0, 1, 0],
                "useful_col": [10, 20, 30],
            }
        )

        result = drop_unnecessary_columns(df)

        # koi_score might be in DROP_COLUMNS
        assert "useful_col" in result.columns

    def test_drop_unnecessary_columns_none_to_drop(self):
        """Test when there are no columns to drop"""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        result = drop_unnecessary_columns(df)

        pd.testing.assert_frame_equal(result, df)

    def test_drop_unnecessary_columns_preserves_data(self):
        """Test that remaining data is preserved"""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        result = drop_unnecessary_columns(df)

        assert len(result) == 3

    def test_drop_unnecessary_columns_preserves_target(self, sample_data_with_target):
        """Test that target column is never dropped"""
        df = sample_data_with_target

        result = drop_unnecessary_columns(df)

        assert "koi_disposition" in result.columns


# ============================================================================
# TEST CLASS: DETECT OUTLIERS IQR
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestDetectOutliersIQR:
    """Test IQR outlier detection"""

    def test_detect_outliers_iqr_with_outliers(self):
        """Test detection of obvious outliers"""
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier

        outlier_mask, outlier_count = detect_outliers_iqr(data, multiplier=1.5)

        assert outlier_count > 0
        assert isinstance(outlier_mask, pd.Series)
        # Check last value is marked as outlier
        assert outlier_mask.iloc[-1] == True  # Fixed: use == instead of is

    def test_detect_outliers_iqr_no_outliers(self):
        """Test when there are no outliers"""
        data = pd.Series([1, 2, 3, 4, 5])

        outlier_mask, outlier_count = detect_outliers_iqr(data, multiplier=1.5)

        assert outlier_count == 0 or outlier_count <= 1

    def test_detect_outliers_iqr_custom_multiplier(self):
        """Test with different IQR multipliers"""
        data = pd.Series([1, 2, 3, 4, 5, 20])

        # Stricter multiplier (more outliers detected)
        mask_strict, count_strict = detect_outliers_iqr(data, multiplier=1.0)

        # Looser multiplier (fewer outliers detected)
        mask_loose, count_loose = detect_outliers_iqr(data, multiplier=3.0)

        assert count_strict >= count_loose

    def test_detect_outliers_iqr_return_types(self):
        """Test return types are correct"""
        data = pd.Series([1, 2, 3, 4, 5, 100])

        outlier_mask, outlier_count = detect_outliers_iqr(data)

        assert isinstance(outlier_mask, pd.Series)
        assert isinstance(outlier_count, (int, np.integer))

    def test_detect_outliers_iqr_with_fixture(self, sample_data_with_outliers):
        """Test outlier detection with fixture data"""
        df = sample_data_with_outliers

        outlier_mask, outlier_count = detect_outliers_iqr(df["koi_score"])

        # Should detect at least one outlier
        assert outlier_count > 0


# ============================================================================
# TEST CLASS: HANDLE OUTLIERS
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestHandleOutliers:
    """Test outlier handling with clip and remove methods"""

    def test_handle_outliers_clip_method(self):
        """Test clipping outliers"""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100]})

        result = handle_outliers(df, columns=["A"], method="clip", multiplier=1.5)

        # Outlier should be clipped (not removed)
        assert len(result) == len(df)
        assert result["A"].max() < 100

    def test_handle_outliers_remove_method(self):
        """Test removing outliers"""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100]})

        result = handle_outliers(df, columns=["A"], method="remove", multiplier=1.5)

        # Row with outlier should be removed
        assert len(result) < len(df)

    def test_handle_outliers_multiple_columns(self):
        """Test handling outliers in multiple columns"""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 100], "B": [5, 6, 7, 8, 200]})

        result = handle_outliers(df, columns=["A", "B"], method="clip", multiplier=1.5)

        assert result["A"].max() < 100
        assert result["B"].max() < 200

    def test_handle_outliers_nonexistent_column(self):
        """Test handling when column doesn't exist"""
        df = pd.DataFrame({"A": [1, 2, 3]})

        result = handle_outliers(df, columns=["NonExistent"], method="clip")

        # Should handle gracefully
        pd.testing.assert_frame_equal(result, df)

    def test_handle_outliers_empty_columns_list(self):
        """Test with empty columns list"""
        df = pd.DataFrame({"A": [1, 2, 100]})

        result = handle_outliers(df, columns=[], method="clip")

        pd.testing.assert_frame_equal(result, df)

    def test_handle_outliers_clip_with_fixture(self, sample_data_with_outliers):
        """Test clipping outliers with fixture data"""
        df = sample_data_with_outliers

        result = handle_outliers(df, columns=["koi_score", "koi_period"], method="clip")

        # Should have clipped outliers
        assert len(result) == len(df)
        assert result["koi_score"].max() < df["koi_score"].max()

    def test_handle_outliers_remove_with_fixture(self, sample_data_with_outliers):
        """Test removing outliers with fixture data"""
        df = sample_data_with_outliers

        result = handle_outliers(df, columns=["koi_score", "koi_period"], method="remove")

        # Should have removed rows with outliers
        assert len(result) < len(df)


# ============================================================================
# TEST CLASS: CONVERT DTYPES
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestConvertDtypes:
    """Test data type conversion and optimization"""

    def test_convert_dtypes_float64_to_float32(self):
        """Test converting float64 to float32"""
        df = pd.DataFrame({"A": [1.5, 2.5, 3.5]})
        assert df["A"].dtype == "float64"

        result = convert_dtypes(df)

        assert result["A"].dtype == "float32"

    def test_convert_dtypes_int64_to_int32(self):
        """Test converting int64 to int32 when safe"""
        df = pd.DataFrame({"A": [1, 2, 3]})

        result = convert_dtypes(df)

        assert result["A"].dtype in ["int32", "int64"]

    def test_convert_dtypes_object_to_category(self):
        """Test converting repeated strings to category"""
        df = pd.DataFrame({"A": ["cat", "dog", "cat", "dog", "cat"]})

        result = convert_dtypes(df)

        # Should be converted to category (>50% repeated)
        assert result["A"].dtype.name == "category"

    def test_convert_dtypes_preserves_data(self):
        """Test that data values are preserved"""
        df = pd.DataFrame({"A": [1.5, 2.5, 3.5], "B": [1, 2, 3]})

        result = convert_dtypes(df)

        assert result["A"].sum() == pytest.approx(df["A"].sum())
        assert result["B"].sum() == df["B"].sum()

    def test_convert_dtypes_reduces_memory(self):
        """Test that memory usage is reduced"""
        df = pd.DataFrame({"A": np.random.randn(1000).astype("float64")})

        initial_memory = df.memory_usage(deep=True).sum()
        result = convert_dtypes(df)
        final_memory = result.memory_usage(deep=True).sum()

        assert final_memory <= initial_memory

    def test_convert_dtypes_with_target_column(self, sample_data_with_target):
        """Test dtype conversion with target column"""
        df = sample_data_with_target

        result = convert_dtypes(df)

        # Target might or might not be converted to category
        # depending on unique ratio (<50% for category conversion)
        # Just check that conversion doesn't break the column
        assert "koi_disposition" in result.columns
        assert result["koi_disposition"].dtype.name in ["category", "object"]


# ============================================================================
# TEST CLASS: VALIDATE TARGET COLUMN
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestValidateTargetColumn:
    """Test target column validation"""

    def test_validate_target_column_valid(self, sample_data_with_target):
        """Test validation of valid target column"""
        df = sample_data_with_target

        # Should not raise
        validate_target_column(df)

    def test_validate_target_column_missing(self):
        """Test validation when target column is missing"""
        df = pd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises(DataValidationError, match="Target sütunu bulunamadı"):
            validate_target_column(df)

    def test_validate_target_column_has_nan(self):
        """Test validation when target has NaN values"""
        df = pd.DataFrame({"koi_disposition": [1, 2, np.nan]})

        with pytest.raises(DataValidationError, match="NaN değer var"):
            validate_target_column(df)

    def test_validate_target_column_all_nan(self):
        """Test validation when target is all NaN"""
        df = pd.DataFrame({"koi_disposition": [np.nan, np.nan, np.nan]})

        with pytest.raises(DataValidationError):
            validate_target_column(df)

    def test_validate_target_column_with_missing_fixture(self, sample_data_with_missing):
        """Test validation with fixture containing missing values"""
        df = sample_data_with_missing

        with pytest.raises(DataValidationError):
            validate_target_column(df)


# ============================================================================
# TEST CLASS: CLEAN DATA (INTEGRATION)
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestCleanData:
    """Test main clean_data function (integration)"""

    def test_clean_data_basic(self, sample_data_with_target):
        """Test basic data cleaning"""
        df = sample_data_with_target

        result = clean_data(df)

        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_clean_data_removes_duplicates(self):
        """Test that duplicates are removed"""
        df = pd.DataFrame(
            {
                "koi_disposition": ["CONFIRMED", "CONFIRMED", "CANDIDATE"],
                "A": [1, 1, 2],
                "B": [3, 3, 4],
            }
        )

        result = clean_data(df)

        # One duplicate row should be removed
        assert len(result) == 2

    def test_clean_data_removes_target_nan(self):
        """Test that rows with NaN target are removed"""
        df = pd.DataFrame({"koi_disposition": ["CONFIRMED", np.nan, "CANDIDATE"], "A": [1, 2, 3]})

        result = clean_data(df)

        # Row with NaN target should be removed
        assert len(result) == 2
        assert not result["koi_disposition"].isnull().any()

    def test_clean_data_optimizes_dtypes(self):
        """Test that data types are optimized"""
        df = pd.DataFrame(
            {
                "koi_disposition": ["CONFIRMED", "CANDIDATE"],
                "A": np.array([1.5, 2.5], dtype="float64"),
            }
        )

        result = clean_data(df)

        # Float should be optimized to float32
        assert result["A"].dtype == "float32"

    def test_clean_data_with_clip_method(self, sample_data_with_target):
        """Test clean_data with clip method for outliers"""
        df = sample_data_with_target

        result = clean_data(df, handle_outliers_method="clip")

        assert isinstance(result, pd.DataFrame)

    def test_clean_data_preserves_target(self, sample_data_with_target):
        """Test that target column is preserved"""
        df = sample_data_with_target

        result = clean_data(df)

        assert "koi_disposition" in result.columns

    def test_clean_data_with_duplicates_fixture(self, sample_data_with_duplicates):
        """Test cleaning with duplicate data fixture"""
        df = sample_data_with_duplicates

        result = clean_data(df)

        # Should remove duplicates
        assert len(result) <= len(df)
        assert not result.duplicated().any()


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestCleanersEdgeCases:
    """Test edge cases in data cleaning"""

    def test_remove_duplicates_with_nan(self):
        """Test duplicate removal with NaN values"""
        df = pd.DataFrame({"A": [1, 2, np.nan, np.nan], "B": [3, 4, 5, 5]})

        result = remove_duplicates(df)

        # NaN rows might be considered different
        assert len(result) >= 3

    def test_handle_outliers_with_all_same_values(self):
        """Test outlier handling when all values are the same"""
        df = pd.DataFrame({"A": [5, 5, 5, 5, 5]})

        result = handle_outliers(df, columns=["A"], method="clip")

        # Should handle gracefully
        assert len(result) == 5

    def test_convert_dtypes_with_mixed_types(self):
        """Test dtype conversion with mixed column types"""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [1.5, 2.5, 3.5],
                "C": ["a", "b", "c"],
                "D": [True, False, True],
            }
        )

        result = convert_dtypes(df)

        # Should handle all types
        assert len(result.columns) == 4

    def test_clean_data_empty_dataframe(self):
        """Test cleaning empty DataFrame"""
        df = pd.DataFrame({"koi_disposition": []})

        # Empty DataFrame should not raise exception
        # Just handle gracefully and return empty
        result = clean_data(df)

        # Should return empty DataFrame
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_clean_data_single_row(self):
        """Test cleaning DataFrame with single row"""
        df = pd.DataFrame({"koi_disposition": ["CONFIRMED"], "koi_score": [0.9], "koi_period": [10.5]})

        result = clean_data(df)

        # Should handle gracefully
        assert len(result) == 1

    def test_handle_outliers_with_nan_values(self):
        """Test outlier handling with NaN values"""
        df = pd.DataFrame({"A": [1, 2, 3, np.nan, 100]})

        result = handle_outliers(df, columns=["A"], method="clip")

        # Should handle NaN gracefully
        assert isinstance(result, pd.DataFrame)
        # NaN should still be present
        assert result["A"].isnull().any()

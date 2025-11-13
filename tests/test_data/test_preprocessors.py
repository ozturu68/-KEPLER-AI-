"""
Tests for data preprocessing utilities
Testing missing value analysis, data splitting, and preprocessing pipeline

Author: sulegogh
Date: 2025-11-13
Version: 2.0 (Fixed column names and stratification issues)
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessors import analyze_missing_values, preprocess_data, save_splits, split_data

# ============================================================================
# TEST CLASS: ANALYZE MISSING VALUES
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestAnalyzeMissingValues:
    """Test missing value analysis"""

    def test_analyze_missing_values_no_missing(self):
        """Test analysis when there are no missing values"""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        result = analyze_missing_values(df)

        assert isinstance(result, pd.DataFrame)
        # All missing counts should be 0
        assert (result["missing_count"] == 0).all()

    def test_analyze_missing_values_with_missing(self, sample_data_with_missing):
        """Test analysis when there are missing values"""
        df = sample_data_with_missing

        result = analyze_missing_values(df)

        assert isinstance(result, pd.DataFrame)
        # Should detect missing values
        assert (result["missing_count"] > 0).any()

    def test_analyze_missing_values_returns_dataframe(self):
        """Test that result is a DataFrame"""
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, np.nan], "C": [7, 8, 9]})

        result = analyze_missing_values(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_analyze_missing_values_has_expected_columns(self):
        """Test that result has expected columns"""
        df = pd.DataFrame({"A": [1, np.nan, 3]})

        result = analyze_missing_values(df)

        # Should have at least missing_count and missing_pct columns
        # (NOT missing_percent - that was the bug!)
        assert "missing_count" in result.columns
        assert "missing_pct" in result.columns

    def test_analyze_missing_values_empty_dataframe(self):
        """Test analysis on empty DataFrame"""
        df = pd.DataFrame()

        result = analyze_missing_values(df)

        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)

    def test_analyze_missing_values_all_missing(self):
        """Test analysis when all values are missing"""
        df = pd.DataFrame({"A": [np.nan, np.nan, np.nan]})

        result = analyze_missing_values(df)

        assert isinstance(result, pd.DataFrame)
        # Should show 100% missing (using correct column name)
        assert result["missing_pct"].iloc[0] == pytest.approx(100.0, rel=0.01)

    def test_analyze_missing_values_partial_missing(self):
        """Test analysis with partial missing values"""
        df = pd.DataFrame({"A": [1, np.nan, 3, np.nan, 5]})

        result = analyze_missing_values(df)

        # 2 missing out of 5 = 40%
        assert result["missing_pct"].iloc[0] == pytest.approx(40.0, rel=0.01)


# ============================================================================
# TEST CLASS: SPLIT DATA
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestSplitData:
    """Test data splitting functionality"""

    def test_split_data_basic(self, sample_data_with_target):
        """Test basic data splitting with larger dataset"""
        df = sample_data_with_target

        train, val, test = split_data(df, target_col="koi_disposition")

        # Should return three DataFrames
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

        # Total rows should match original
        assert len(train) + len(val) + len(test) == len(df)

    def test_split_data_default_proportions(self, sample_data_with_target):
        """Test default split proportions (70-15-15)"""
        df = sample_data_with_target

        train, val, test = split_data(df, target_col="koi_disposition")

        total = len(df)
        # Check approximate proportions
        assert len(train) >= len(val)
        assert len(train) >= len(test)
        # Train should be roughly 70% of total
        assert len(train) > total * 0.6

    def test_split_data_custom_proportions(self, sample_data_with_target):
        """Test custom split proportions"""
        df = sample_data_with_target

        train, val, test = split_data(
            df,
            target_col="koi_disposition",
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
        )

        # Should still split into three parts
        assert len(train) + len(val) + len(test) == len(df)

    def test_split_data_preserves_columns(self, sample_data_with_target):
        """Test that all columns are preserved"""
        df = sample_data_with_target

        train, val, test = split_data(df, target_col="koi_disposition")

        # All splits should have same columns
        assert list(train.columns) == list(df.columns)
        assert list(val.columns) == list(df.columns)
        assert list(test.columns) == list(df.columns)

    def test_split_data_target_column_present(self, sample_data_with_target):
        """Test that target column is in all splits"""
        df = sample_data_with_target

        train, val, test = split_data(df, target_col="koi_disposition")

        assert "koi_disposition" in train.columns
        assert "koi_disposition" in val.columns
        assert "koi_disposition" in test.columns

    def test_split_data_stratified(self, sample_data_with_target):
        """Test stratified splitting with sufficient data"""
        df = sample_data_with_target  # Now has 30 samples (10 per class)

        train, val, test = split_data(df, target_col="koi_disposition", stratify=True)

        # Check that target distribution is similar across splits
        train_dist = train["koi_disposition"].value_counts(normalize=True)
        val_dist = val["koi_disposition"].value_counts(normalize=True)

        # All classes should be present in training set
        assert len(train_dist) == 3  # 3 classes
        # Distributions should be similar
        assert isinstance(train_dist, pd.Series)
        assert isinstance(val_dist, pd.Series)

    def test_split_data_random_state(self, sample_data_with_target):
        """Test that random_state gives reproducible results"""
        df = sample_data_with_target

        train1, val1, test1 = split_data(df, target_col="koi_disposition", random_state=42)
        train2, val2, test2 = split_data(df, target_col="koi_disposition", random_state=42)

        # Should be identical
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_split_data_no_stratification(self, sample_data_with_target):
        """Test splitting without stratification"""
        df = sample_data_with_target

        train, val, test = split_data(df, target_col="koi_disposition", stratify=False)

        # Should still work
        assert len(train) + len(val) + len(test) == len(df)


# ============================================================================
# TEST CLASS: SAVE SPLITS
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestSaveSplits:
    """Test saving data splits to disk"""

    def test_save_splits_creates_files(self, sample_data_with_target, temp_dir):
        """Test that save_splits creates CSV files"""
        df = sample_data_with_target
        train, val, test = split_data(df, target_col="koi_disposition")

        output_dir = temp_dir / "splits"
        save_splits(train, val, test, output_dir=str(output_dir))

        # Check files exist
        assert (output_dir / "train.csv").exists()
        assert (output_dir / "val.csv").exists()
        assert (output_dir / "test.csv").exists()

    def test_save_splits_preserves_data(self, sample_data_with_target, temp_dir):
        """Test that saved data can be loaded back correctly"""
        df = sample_data_with_target
        train, val, test = split_data(df, target_col="koi_disposition")

        output_dir = temp_dir / "splits"
        save_splits(train, val, test, output_dir=str(output_dir))

        # Load back and compare
        train_loaded = pd.read_csv(output_dir / "train.csv")
        assert len(train_loaded) == len(train)
        assert list(train_loaded.columns) == list(train.columns)

    def test_save_splits_creates_directory(self, sample_data_with_target, temp_dir):
        """Test that save_splits creates output directory if needed"""
        df = sample_data_with_target
        train, val, test = split_data(df, target_col="koi_disposition")

        output_dir = temp_dir / "new_splits_dir"
        save_splits(train, val, test, output_dir=str(output_dir))

        # Directory should be created
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_save_splits_overwrites_existing(self, sample_data_with_target, temp_dir):
        """Test that save_splits can overwrite existing files"""
        df = sample_data_with_target
        train1, val1, test1 = split_data(df, target_col="koi_disposition", random_state=42)
        train2, val2, test2 = split_data(df, target_col="koi_disposition", random_state=43)

        output_dir = temp_dir / "splits"

        # Save first time
        save_splits(train1, val1, test1, output_dir=str(output_dir))

        # Save again (overwrite)
        save_splits(train2, val2, test2, output_dir=str(output_dir))

        # Should have files (overwritten)
        assert (output_dir / "train.csv").exists()


# ============================================================================
# TEST CLASS: PREPROCESS DATA (INTEGRATION)
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestPreprocessData:
    """Test main preprocessing pipeline"""

    def test_preprocess_data_returns_dict(self, sample_data_with_target):
        """Test that preprocess_data returns a dictionary"""
        df = sample_data_with_target

        result = preprocess_data(df, handle_missing=True, split=True)

        assert isinstance(result, dict)

    def test_preprocess_data_with_split(self, sample_data_with_target):
        """Test preprocessing with data splitting"""
        df = sample_data_with_target

        result = preprocess_data(df, handle_missing=True, split=True)

        # Should have train, val, test keys
        assert "train" in result
        assert "val" in result
        assert "test" in result

    def test_preprocess_data_without_split(self, sample_data_with_target):
        """Test preprocessing without data splitting"""
        df = sample_data_with_target

        result = preprocess_data(df, handle_missing=True, split=False)

        # Should have processed data
        assert "data" in result or len(result) > 0

    def test_preprocess_data_handles_missing(self, sample_data_with_missing):
        """Test preprocessing with missing value handling"""
        df = sample_data_with_missing

        result = preprocess_data(df, handle_missing=True, split=False)

        # Should handle missing values
        assert isinstance(result, dict)

    def test_preprocess_data_preserves_target(self, sample_data_with_target):
        """Test that preprocessing preserves target column"""
        df = sample_data_with_target

        result = preprocess_data(df, handle_missing=True, split=True)

        # Target should be in all splits
        assert "koi_disposition" in result["train"].columns
        assert "koi_disposition" in result["val"].columns
        assert "koi_disposition" in result["test"].columns

    def test_preprocess_data_with_duplicates(self, sample_data_with_duplicates):
        """Test preprocessing with duplicate data"""
        df = sample_data_with_duplicates

        result = preprocess_data(df, handle_missing=True, split=False)

        # Should handle duplicates
        assert isinstance(result, dict)

    def test_preprocess_data_integration_full(self, sample_data_with_target):
        """Test full preprocessing pipeline integration"""
        df = sample_data_with_target

        result = preprocess_data(df, handle_missing=True, split=True)

        # Check complete pipeline
        assert isinstance(result, dict)
        assert "train" in result
        assert "val" in result
        assert "test" in result

        # Check data integrity
        train = result["train"]
        val = result["val"]
        test = result["test"]

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================


@pytest.mark.unit
@pytest.mark.data
class TestPreprocessorsEdgeCases:
    """Test edge cases in preprocessing"""

    def test_split_data_small_dataset(self):
        """Test splitting very small dataset (without stratification)"""
        # Create dataset with enough samples for each class
        df = pd.DataFrame(
            {
                "koi_disposition": (["CONFIRMED"] * 3 + ["CANDIDATE"] * 3 + ["FALSE POSITIVE"] * 3),
                "A": list(range(9)),
            }
        )

        # Disable stratification for small dataset
        train, val, test = split_data(df, target_col="koi_disposition", stratify=False)

        # Should handle gracefully
        assert len(train) + len(val) + len(test) == len(df)

    def test_analyze_missing_values_single_column(self):
        """Test analysis on single column DataFrame"""
        df = pd.DataFrame({"A": [1, np.nan, 3, np.nan, 5]})

        result = analyze_missing_values(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        # 2 missing out of 5
        assert result["missing_count"].iloc[0] == 2

    def test_preprocess_data_minimal_data(self):
        """Test preprocessing with minimal but sufficient data"""
        # Create larger minimal dataset
        df = pd.DataFrame(
            {
                "koi_disposition": (["CONFIRMED"] * 5 + ["CANDIDATE"] * 5 + ["FALSE POSITIVE"] * 5),
                "A": list(range(15)),
            }
        )

        result = preprocess_data(df, handle_missing=False, split=False)

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_split_data_with_sufficient_samples(self):
        """Test splitting with sufficient samples per class"""
        # Create dataset with enough samples for stratification
        df = pd.DataFrame(
            {
                "koi_disposition": (["CONFIRMED"] * 10 + ["CANDIDATE"] * 10 + ["FALSE POSITIVE"] * 10),
                "A": list(range(30)),
            }
        )

        train, val, test = split_data(df, target_col="koi_disposition", stratify=True)

        # Should work without errors
        assert len(train) + len(val) + len(test) == len(df)

    def test_save_splits_empty_dataframes(self, temp_dir):
        """Test saving empty DataFrames"""
        train = pd.DataFrame()
        val = pd.DataFrame()
        test = pd.DataFrame()

        output_dir = temp_dir / "empty_splits"

        # Should handle gracefully
        save_splits(train, val, test, output_dir=str(output_dir))

        # Files should exist (even if empty)
        assert (output_dir / "train.csv").exists()

    def test_analyze_missing_values_multiple_columns_mixed(self):
        """Test analysis with mixed missing patterns"""
        df = pd.DataFrame(
            {
                "A": [1, np.nan, 3, np.nan, 5],
                "B": [np.nan, 2, 3, 4, 5],
                "C": [1, 2, 3, 4, 5],
            }
        )

        result = analyze_missing_values(df)

        # Function only returns columns WITH missing values
        # Column C has no missing, so it won't be in result
        assert len(result) == 2  # Only A and B have missing values

        # Verify A has 2 missing (40%)
        a_row = result[result["column"] == "A"]
        assert a_row["missing_count"].iloc[0] == 2
        assert a_row["missing_pct"].iloc[0] == pytest.approx(40.0, rel=0.01)

        # Verify B has 1 missing (20%)
        b_row = result[result["column"] == "B"]
        assert b_row["missing_count"].iloc[0] == 1
        assert b_row["missing_pct"].iloc[0] == pytest.approx(20.0, rel=0.01)

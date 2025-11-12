"""
Tests for constants module
Testing project constants

Author: sulegogh
Date: 2025-11-12
Version: 1.0
"""

from pathlib import Path

import pytest

from src.core.constants import RANDOM_STATE  # RANDOM_SEED yerine
from src.core.constants import TARGET_VALUES  # CLASS_NAMES yerine
from src.core.constants import (
    DATA_DIR,
    DATA_PROCESSED,
    DATA_RAW,
    LOGS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    TARGET_COLUMN,
    TEST_SIZE,
    VAL_SIZE,
)


@pytest.mark.unit
class TestPaths:
    """Test path constants"""

    def test_project_root_exists(self):
        """Test that project root is defined"""
        assert PROJECT_ROOT is not None
        assert isinstance(PROJECT_ROOT, (str, Path))

    def test_data_dir_exists(self):
        """Test that data directory constant is defined"""
        assert DATA_DIR is not None
        assert isinstance(DATA_DIR, (str, Path))

    def test_data_raw_exists(self):
        """Test that raw data path is defined"""
        assert DATA_RAW is not None
        assert isinstance(DATA_RAW, (str, Path))

    def test_data_processed_exists(self):
        """Test that processed data path is defined"""
        assert DATA_PROCESSED is not None
        assert isinstance(DATA_PROCESSED, (str, Path))

    def test_models_dir_exists(self):
        """Test that models directory constant is defined"""
        assert MODELS_DIR is not None
        assert isinstance(MODELS_DIR, (str, Path))

    def test_logs_dir_exists(self):
        """Test that logs directory constant is defined"""
        assert LOGS_DIR is not None
        assert isinstance(LOGS_DIR, (str, Path))


@pytest.mark.unit
class TestTargetColumn:
    """Test target column constant"""

    def test_target_column_defined(self):
        """Test that target column is defined"""
        assert TARGET_COLUMN is not None
        assert isinstance(TARGET_COLUMN, str)

    def test_target_column_value(self):
        """Test target column has expected value"""
        assert TARGET_COLUMN == 'koi_disposition'


@pytest.mark.unit
class TestTargetValues:
    """Test target values (class names) constant"""

    def test_target_values_defined(self):
        """Test that target values are defined"""
        assert TARGET_VALUES is not None
        assert isinstance(TARGET_VALUES, (list, tuple, dict))

    def test_target_values_count(self):
        """Test that there are 3 target values"""
        assert len(TARGET_VALUES) == 3

    def test_target_values_content(self):
        """Test that target values contain expected classes"""
        # Expected class names
        expected_classes = {'CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'}

        # Get actual class names based on structure
        if isinstance(TARGET_VALUES, dict):
            # If dict, check keys (class names)
            actual_classes = set(TARGET_VALUES.keys())
        else:
            # If list/tuple, convert to set
            actual_classes = set(TARGET_VALUES)

        # Check that all expected classes are present
        assert expected_classes.issubset(
            actual_classes
        ), f"Missing classes: {expected_classes - actual_classes}. Found: {actual_classes}"

    def test_target_values_mapping(self):
        """Test that target values have correct mappings"""
        if isinstance(TARGET_VALUES, dict):
            # Should map class names to numeric values
            assert all(isinstance(k, str) for k in TARGET_VALUES.keys()), "Keys should be string class names"
            assert all(isinstance(v, int) for v in TARGET_VALUES.values()), "Values should be integers"

            # Check specific mappings exist
            assert 'CANDIDATE' in TARGET_VALUES
            assert 'CONFIRMED' in TARGET_VALUES
            assert 'FALSE POSITIVE' in TARGET_VALUES

            # Values should be 0, 1, 2 (in any order)
            assert set(TARGET_VALUES.values()) == {0, 1, 2}


@pytest.mark.unit
class TestHyperparameters:
    """Test hyperparameter constants"""

    def test_random_state_defined(self):
        """Test that random state is defined"""
        assert RANDOM_STATE is not None
        assert isinstance(RANDOM_STATE, int)

    def test_random_state_value(self):
        """Test random state is 42 (for reproducibility)"""
        assert RANDOM_STATE == 42

    def test_test_size_defined(self):
        """Test that test size is defined"""
        assert TEST_SIZE is not None
        assert isinstance(TEST_SIZE, float)

    def test_test_size_range(self):
        """Test test size is between 0 and 1"""
        assert 0 < TEST_SIZE < 1

    def test_val_size_defined(self):
        """Test that validation size is defined"""
        assert VAL_SIZE is not None
        assert isinstance(VAL_SIZE, float)

    def test_val_size_range(self):
        """Test validation size is between 0 and 1"""
        assert 0 < VAL_SIZE < 1

    def test_split_sizes_sum_less_than_one(self):
        """Test that test + val < 1 (leaving room for train)"""
        assert TEST_SIZE + VAL_SIZE < 1.0

    def test_split_sizes_reasonable(self):
        """Test that split sizes are reasonable (not too small/large)"""
        # Test size should be at least 10% and at most 30%
        assert 0.1 <= TEST_SIZE <= 0.3
        # Val size should be at least 10% and at most 30%
        assert 0.1 <= VAL_SIZE <= 0.3


@pytest.mark.unit
class TestPathTypes:
    """Test that paths are proper Path objects or strings"""

    def test_all_path_constants_are_valid_types(self):
        """Test that all path constants have valid types"""
        path_constants = [
            ('PROJECT_ROOT', PROJECT_ROOT),
            ('DATA_DIR', DATA_DIR),
            ('DATA_RAW', DATA_RAW),
            ('DATA_PROCESSED', DATA_PROCESSED),
            ('MODELS_DIR', MODELS_DIR),
            ('LOGS_DIR', LOGS_DIR),
        ]

        for name, path in path_constants:
            assert isinstance(path, (str, Path)), f"{name} should be str or Path, got {type(path)}"

    def test_paths_can_be_converted_to_path_objects(self):
        """Test that all path constants can be converted to Path objects"""
        path_constants = [PROJECT_ROOT, DATA_DIR, DATA_RAW, DATA_PROCESSED, MODELS_DIR, LOGS_DIR]

        for path in path_constants:
            path_obj = Path(path)
            assert isinstance(path_obj, Path)

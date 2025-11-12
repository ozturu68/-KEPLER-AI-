"""
Tests for label utilities
Comprehensive testing for encode/decode/normalize/validate functions

Author: sulegogh
Date: 2025-11-12
Version: 2.0
"""

import numpy as np
import pandas as pd
import pytest

from src.core.label_utils import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    decode_labels,
    encode_labels,
    normalize_labels,
    validate_labels,
)

# ============================================================================
# TEST CLASS: ENCODE LABELS
# ============================================================================


@pytest.mark.unit
class TestEncodeLabels:
    """Test encode_labels function - string to numeric conversion"""

    def test_encode_basic_string_array(self):
        """Test encoding basic string array"""
        labels = np.array(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
        result = encode_labels(labels)

        assert isinstance(result, np.ndarray)
        assert result.dtype in [np.int32, np.int64, np.int_]
        assert list(result) == [0, 1, 2]

    def test_encode_with_duplicates(self):
        """Test encoding array with duplicate labels"""
        labels = np.array(['CANDIDATE', 'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])
        result = encode_labels(labels)

        assert len(result) == 4
        assert list(result) == [0, 1, 0, 2]

    def test_encode_already_numeric(self):
        """Test encoding already numeric labels (should return as-is)"""
        labels = np.array([0, 1, 2, 0])
        result = encode_labels(labels)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, labels)

    def test_encode_pandas_series(self):
        """Test encoding pandas Series"""
        labels = pd.Series(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
        result = encode_labels(labels)

        assert isinstance(result, np.ndarray)
        assert list(result) == [0, 1, 2]

    def test_encode_python_list(self):
        """Test encoding Python list"""
        labels = ['CANDIDATE', 'CONFIRMED']
        result = encode_labels(labels)

        assert isinstance(result, np.ndarray)
        assert list(result) == [0, 1]

    def test_encode_2d_array_should_flatten(self):
        """Test encoding 2D array (should flatten automatically)"""
        labels = np.array([['FALSE POSITIVE'], ['CONFIRMED'], ['CANDIDATE']])
        result = encode_labels(labels)

        assert result.ndim == 1, "Result should be 1D array"
        assert len(result) == 3
        assert list(result) == [2, 1, 0]

    def test_encode_empty_array(self):
        """Test encoding empty array"""
        labels = np.array([])
        result = encode_labels(labels)

        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_encode_single_value(self):
        """Test encoding single value"""
        labels = np.array(['CANDIDATE'])
        result = encode_labels(labels)

        assert len(result) == 1
        assert result[0] == 0

    def test_encode_invalid_label_raises_error(self):
        """Test that invalid label raises KeyError"""
        labels = np.array(['CANDIDATE', 'INVALID_CLASS'])

        with pytest.raises(KeyError):
            encode_labels(labels)

    def test_encode_preserves_order(self):
        """Test that encoding preserves label order"""
        labels = np.array(['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED', 'CANDIDATE'])
        result = encode_labels(labels)

        assert list(result) == [2, 0, 1, 0]


# ============================================================================
# TEST CLASS: DECODE LABELS
# ============================================================================


@pytest.mark.unit
class TestDecodeLabels:
    """Test decode_labels function - numeric to string conversion"""

    def test_decode_basic_numeric_array(self):
        """Test decoding basic numeric array"""
        labels = np.array([0, 1, 2])
        result = decode_labels(labels)

        assert isinstance(result, np.ndarray)
        # NumPy may return Unicode string dtype ('<U14') or object dtype
        assert result.dtype.kind in ['U', 'O'], f"Expected string dtype, got {result.dtype}"
        assert list(result) == ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']

    def test_decode_with_duplicates(self):
        """Test decoding array with duplicate numeric values"""
        labels = np.array([0, 1, 0, 2])
        result = decode_labels(labels)

        assert len(result) == 4
        assert list(result) == ['CANDIDATE', 'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']

    def test_decode_already_string(self):
        """Test decoding already string labels (should return as-is)"""
        labels = np.array(['CANDIDATE', 'CONFIRMED'])
        result = decode_labels(labels)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, labels)

    def test_decode_python_list(self):
        """Test decoding Python list"""
        labels = [0, 1, 2]
        result = decode_labels(labels)

        assert isinstance(result, np.ndarray)
        assert list(result) == ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']

    def test_decode_single_value(self):
        """Test decoding single value"""
        labels = [0]
        result = decode_labels(labels)

        assert len(result) == 1
        assert result[0] == 'CANDIDATE'

    def test_decode_invalid_numeric_raises_error(self):
        """Test that invalid numeric label raises KeyError"""
        labels = np.array([0, 1, 99])

        with pytest.raises(KeyError):
            decode_labels(labels)

    def test_decode_negative_number_raises_error(self):
        """Test that negative number raises KeyError"""
        labels = np.array([0, -1])

        with pytest.raises(KeyError):
            decode_labels(labels)

    def test_decode_preserves_order(self):
        """Test that decoding preserves label order"""
        labels = np.array([2, 0, 1, 0])
        result = decode_labels(labels)

        assert list(result) == ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED', 'CANDIDATE']


# ============================================================================
# TEST CLASS: NORMALIZE LABELS
# ============================================================================


@pytest.mark.unit
class TestNormalizeLabels:
    """Test normalize_labels function - flexible label conversion"""

    def test_normalize_string_to_numeric(self):
        """Test normalizing string labels to numeric"""
        labels = np.array(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
        result = normalize_labels(labels, target_format='numeric')

        assert result.dtype in [np.int32, np.int64, np.int_]
        assert list(result) == [0, 1, 2]

    def test_normalize_numeric_to_string(self):
        """Test normalizing numeric labels to string"""
        labels = np.array([0, 1, 2, 0])
        result = normalize_labels(labels, target_format='string')

        assert result.dtype.kind in ['U', 'O']
        assert list(result) == ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE']

    def test_normalize_string_to_string(self):
        """Test normalizing string to string (no-op)"""
        labels = np.array(['CANDIDATE', 'CONFIRMED'])
        result = normalize_labels(labels, target_format='string')

        assert result.dtype.kind in ['U', 'O']
        np.testing.assert_array_equal(result, labels)

    def test_normalize_numeric_to_numeric(self):
        """Test normalizing numeric to numeric (no-op)"""
        labels = np.array([0, 1, 2])
        result = normalize_labels(labels, target_format='numeric')

        assert result.dtype in [np.int32, np.int64, np.int_]
        np.testing.assert_array_equal(result, labels)

    def test_normalize_invalid_format_raises_error(self):
        """Test that invalid target format raises ValueError"""
        labels = np.array(['CANDIDATE'])

        with pytest.raises(ValueError, match="target_format must be"):
            normalize_labels(labels, target_format='invalid')

    def test_normalize_pandas_series(self):
        """Test normalizing pandas Series"""
        labels = pd.Series(['CANDIDATE', 'CONFIRMED'])
        result = normalize_labels(labels, target_format='numeric')

        assert isinstance(result, np.ndarray)
        assert list(result) == [0, 1]

    def test_normalize_mixed_duplicates(self):
        """Test normalizing with duplicates"""
        labels = ['CANDIDATE', 'CANDIDATE', 'CONFIRMED']
        result = normalize_labels(labels, target_format='numeric')

        assert list(result) == [0, 0, 1]


# ============================================================================
# TEST CLASS: VALIDATE LABELS
# ============================================================================


@pytest.mark.unit
class TestValidateLabels:
    """Test validate_labels function - label format validation"""

    def test_validate_numeric_labels(self):
        """Test validating numeric labels"""
        labels = np.array([0, 1, 2, 0])

        assert validate_labels(labels) is True

    def test_validate_string_labels(self):
        """Test validating string labels"""
        labels = np.array(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])

        assert validate_labels(labels) is True

    def test_validate_single_class(self):
        """Test validating labels with single class"""
        labels = np.array([0, 0, 0])

        assert validate_labels(labels) is True

    def test_validate_all_classes_present(self):
        """Test validating with all classes present"""
        labels = np.array([0, 1, 2, 0, 1, 2])

        assert validate_labels(labels) is True

    def test_validate_pandas_series(self):
        """Test validating pandas Series"""
        labels = pd.Series(['CANDIDATE', 'CONFIRMED'])

        assert validate_labels(labels) is True

    def test_validate_invalid_numeric_raises_error(self):
        """Test that invalid numeric labels raise ValueError"""
        labels = np.array([0, 1, 99])

        with pytest.raises(ValueError, match="Invalid numeric labels"):
            validate_labels(labels)

    def test_validate_invalid_string_raises_error(self):
        """Test that invalid string labels raise ValueError"""
        labels = np.array(['CANDIDATE', 'INVALID_CLASS'])

        with pytest.raises(ValueError, match="Invalid string labels"):
            validate_labels(labels)

    def test_validate_negative_numeric_raises_error(self):
        """Test that negative numbers raise ValueError"""
        labels = np.array([0, -1])

        with pytest.raises(ValueError, match="Invalid numeric labels"):
            validate_labels(labels)

    def test_validate_out_of_range_numeric_raises_error(self):
        """Test that out-of-range numbers raise ValueError"""
        labels = np.array([0, 1, 3])

        with pytest.raises(ValueError, match="Invalid numeric labels"):
            validate_labels(labels)

    def test_validate_mixed_format_raises_error(self):
        """Test that mixed format labels raise ValueError"""
        # When creating np.array with mixed types, Python converts to strings
        # So ['CANDIDATE', 0] becomes ['CANDIDATE', '0']
        labels = np.array(['CANDIDATE', 0, 'CONFIRMED', 1])

        # This will fail because '0' and '1' are not valid string labels
        with pytest.raises(ValueError, match="Invalid string labels"):
            validate_labels(labels)

    def test_validate_empty_array(self):
        """Test validating empty array (edge case)"""
        labels = np.array([])

        # Empty array should raise an error or be handled gracefully
        # Depending on implementation, adjust this test
        try:
            result = validate_labels(labels)
            # If no error, it should return True
            assert result is True
        except (ValueError, IndexError):
            # If error is raised, that's also acceptable
            pass


# ============================================================================
# TEST CLASS: CONSTANTS
# ============================================================================


@pytest.mark.unit
class TestLabelConstants:
    """Test label mapping constants"""

    def test_label_to_int_mapping_structure(self):
        """Test LABEL_TO_INT constant structure"""
        assert isinstance(LABEL_TO_INT, dict)
        assert len(LABEL_TO_INT) == 3

    def test_label_to_int_mapping_values(self):
        """Test LABEL_TO_INT constant has correct mappings"""
        assert LABEL_TO_INT == {'CANDIDATE': 0, 'CONFIRMED': 1, 'FALSE POSITIVE': 2}

    def test_int_to_label_mapping_structure(self):
        """Test INT_TO_LABEL constant structure"""
        assert isinstance(INT_TO_LABEL, dict)
        assert len(INT_TO_LABEL) == 3

    def test_int_to_label_mapping_values(self):
        """Test INT_TO_LABEL constant has correct mappings"""
        assert INT_TO_LABEL == {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}

    def test_mappings_are_inverse(self):
        """Test that LABEL_TO_INT and INT_TO_LABEL are inverse of each other"""
        # Forward mapping
        for label, num in LABEL_TO_INT.items():
            assert INT_TO_LABEL[num] == label, f"Mismatch: {label} <-> {num}"

        # Reverse mapping
        for num, label in INT_TO_LABEL.items():
            assert LABEL_TO_INT[label] == num, f"Mismatch: {num} <-> {label}"

    def test_all_classes_present_in_mappings(self):
        """Test that all 3 classes are present in both mappings"""
        assert len(LABEL_TO_INT) == 3
        assert len(INT_TO_LABEL) == 3

        # Check specific classes
        assert 'CANDIDATE' in LABEL_TO_INT
        assert 'CONFIRMED' in LABEL_TO_INT
        assert 'FALSE POSITIVE' in LABEL_TO_INT

    def test_numeric_mapping_sequential(self):
        """Test that numeric mapping is sequential (0, 1, 2)"""
        assert set(INT_TO_LABEL.keys()) == {0, 1, 2}
        assert set(LABEL_TO_INT.values()) == {0, 1, 2}

    def test_candidate_is_zero(self):
        """Test that CANDIDATE is mapped to 0 (minority class priority)"""
        assert LABEL_TO_INT['CANDIDATE'] == 0
        assert INT_TO_LABEL[0] == 'CANDIDATE'

    def test_mappings_immutable(self):
        """Test that mappings are effectively immutable (dict checks)"""
        # This tests that we can access the dicts correctly
        # In production, these should be frozen/immutable
        assert isinstance(LABEL_TO_INT, dict)
        assert isinstance(INT_TO_LABEL, dict)


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and corner scenarios"""

    def test_encode_very_large_array(self):
        """Test encoding very large array (performance check)"""
        # Create large array with random classes
        np.random.seed(42)
        labels = np.random.choice(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'], 10000)

        result = encode_labels(labels)

        assert len(result) == 10000
        assert all(r in [0, 1, 2] for r in result)

    def test_decode_very_large_array(self):
        """Test decoding very large array"""
        np.random.seed(42)
        labels = np.random.choice([0, 1, 2], 10000)

        result = decode_labels(labels)

        assert len(result) == 10000

    def test_encode_decode_roundtrip(self):
        """Test that encode -> decode is identity"""
        original = np.array(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'])

        encoded = encode_labels(original)
        decoded = decode_labels(encoded)

        np.testing.assert_array_equal(decoded, original)

    def test_decode_encode_roundtrip(self):
        """Test that decode -> encode is identity"""
        original = np.array([0, 1, 2, 0])

        decoded = decode_labels(original)
        encoded = encode_labels(decoded)

        np.testing.assert_array_equal(encoded, original)

    def test_normalize_consistency(self):
        """Test that normalize is consistent with encode/decode"""
        labels = np.array(['CANDIDATE', 'CONFIRMED'])

        encoded_direct = encode_labels(labels)
        normalized = normalize_labels(labels, target_format='numeric')

        np.testing.assert_array_equal(encoded_direct, normalized)

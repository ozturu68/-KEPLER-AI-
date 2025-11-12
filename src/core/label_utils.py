"""
Label Mapping Utilities
Standardize label encoding/decoding across the project

Author: sulegogh
Date: 2025-11-12
Version: 1.0
"""

from typing import List, Union

import numpy as np
import pandas as pd

# Standard label mappings
LABEL_TO_INT = {'CANDIDATE': 0, 'CONFIRMED': 1, 'FALSE POSITIVE': 2}

INT_TO_LABEL = {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}


def encode_labels(labels: list | np.ndarray | pd.Series) -> np.ndarray:
    """
    Convert string labels to numeric encoding

    Args:
        labels: String labels (CANDIDATE, CONFIRMED, FALSE POSITIVE)

    Returns:
        Numeric array (0, 1, 2)

    Example:
        >>> labels = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
        >>> encode_labels(labels)
        array([0, 1, 2])
    """
    if isinstance(labels, pd.Series):
        labels = labels.values

    # Flatten if multi-dimensional
    if isinstance(labels, np.ndarray) and len(labels.shape) > 1:
        labels = labels.flatten()

    # Convert to array if list
    if isinstance(labels, list):
        labels = np.array(labels)

    # Check if already numeric
    if len(labels) > 0:
        # Check first element type
        first_elem = labels.flat[0] if labels.ndim > 1 else labels[0]

        if not isinstance(first_elem, str):
            # Already numeric, just flatten and return
            return labels.flatten() if labels.ndim > 1 else labels

    # Encode string to numeric
    encoded = np.array([LABEL_TO_INT[label] for label in labels.flat if isinstance(label, str)])
    return encoded


def decode_labels(labels: list | np.ndarray) -> np.ndarray:
    """
    Convert numeric labels to string encoding

    Args:
        labels: Numeric labels (0, 1, 2)

    Returns:
        String array (CANDIDATE, CONFIRMED, FALSE POSITIVE)

    Example:
        >>> labels = [0, 1, 2]
        >>> decode_labels(labels)
        array(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
    """
    # Check if already string
    if isinstance(labels[0], str):
        return np.array(labels)

    # Decode
    decoded = np.array([INT_TO_LABEL[int(label)] for label in labels])
    return decoded


def normalize_labels(labels: list | np.ndarray | pd.Series, target_format: str = 'numeric') -> np.ndarray:
    """
    Normalize labels to target format (numeric or string)

    Args:
        labels: Input labels (any format)
        target_format: 'numeric' or 'string'

    Returns:
        Normalized labels

    Example:
        >>> labels = ['CANDIDATE', 0, 'CONFIRMED', 1]
        >>> normalize_labels(labels, 'numeric')
        array([0, 0, 1, 1])
    """
    if target_format == 'numeric':
        return encode_labels(labels)
    elif target_format == 'string':
        return decode_labels(labels)
    else:
        raise ValueError(f"target_format must be 'numeric' or 'string', got {target_format}")


def validate_labels(labels: list | np.ndarray | pd.Series) -> bool:
    """
    Validate that labels are in correct format

    Args:
        labels: Labels to validate

    Returns:
        True if valid, raises ValueError if not

    Example:
        >>> labels = [0, 1, 2]
        >>> validate_labels(labels)
        True
    """
    unique = np.unique(labels)

    # Check numeric format
    if all(isinstance(x, (int, np.integer)) for x in unique):
        valid_numeric = {0, 1, 2}
        if not set(unique).issubset(valid_numeric):
            raise ValueError(f"Invalid numeric labels: {unique}. Expected {valid_numeric}")
        return True

    # Check string format
    if all(isinstance(x, str) for x in unique):
        valid_string = set(LABEL_TO_INT.keys())
        if not set(unique).issubset(valid_string):
            raise ValueError(f"Invalid string labels: {unique}. Expected {valid_string}")
        return True

    # Mixed format
    raise ValueError(f"Mixed label formats detected: {unique}")


# CLI testing
if __name__ == "__main__":
    print("Testing label utilities...")

    # Test 1: Encode
    string_labels = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
    numeric = encode_labels(string_labels)
    print(f"\n✅ Encode: {string_labels} → {numeric}")

    # Test 2: Decode
    numeric_labels = [0, 1, 2]
    strings = decode_labels(numeric_labels)
    print(f"✅ Decode: {numeric_labels} → {strings}")

    # Test 3: Normalize
    mixed = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE']
    normalized = normalize_labels(mixed, 'numeric')
    print(f"✅ Normalize: {mixed} → {normalized}")

    # Test 4: Validate
    valid_numeric = [0, 1, 2, 0, 1]
    validate_labels(valid_numeric)
    print(f"✅ Validate: {valid_numeric} is valid")

    print("\n✅ All tests passed!")

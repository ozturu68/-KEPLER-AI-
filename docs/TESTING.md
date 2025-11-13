# 🧪 Testing Guide

**Last Updated:** 2025-11-13  
**Status:** ✅ Complete

## Overview

Kepler Exoplanet ML projesi kapsamlı test coverage'ı ile geliştirilmiştir. Bu döküman test stratejisi, yazma kuralları
ve best practice'leri içerir.

## Table of Contents

- [Test Statistics](#test-statistics)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Fixtures](#test-fixtures)
- [Test Markers](#test-markers)
- [Coverage](#coverage)
- [CI/CD Integration](#cicd-integration)

---

## Test Statistics

### Current Status

```
╔════════════════════════════════════════════════════╗
║           TEST SUITE STATISTICS                    ║
╠════════════════════════════════════════════════════╣
║  Total Tests:              359                     ║
║  Passed:                   359 (100%)              ║
║  Failed:                   0                       ║
║  Coverage:                 66.17%                  ║
║  Execution Time:           ~10 seconds             ║
║  Test Files:               11                      ║
╚════════════════════════════════════════════════════╝
```

### Test Distribution

```
Module                  Tests    Coverage    Status
─────────────────────────────────────────────────────
Core (constants)         23      100.00%     ✅
Core (label_utils)       49       96.97%     ✅
Data (cleaners)          35       97.10%     ✅
Data (preprocessors)     43       81.44%     ✅
Features (engineering)   29       84.47%     ✅
Features (scalers)       39       91.34%     ✅
Features (selection)     33       92.86%     ✅
Evaluation (metrics)     33       97.40%     ✅
Models (loader)          26       90.62%     ✅
Models (predictions)     30       ~95%       ✅
Integration (pipeline)   19      100.00%     ✅
─────────────────────────────────────────────────────
TOTAL                   359       66.17%     ✅
```

---

## Test Structure

### Directory Layout

```
tests/
├── conftest.py              # Shared fixtures
├── __init__.py
│
├── test_core/               # Core module tests (72 tests)
│   ├── __init__.py
│   ├── test_constants.py
│   └── test_label_utils.py
│
├── test_data/               # Data module tests (78 tests)
│   ├── __init__.py
│   ├── test_cleaners.py
│   └── test_preprocessors.py
│
├── test_evaluation/         # Evaluation tests (33 tests)
│   ├── __init__.py
│   └── test_metrics.py
│
├── test_features/           # Feature tests (101 tests)
│   ├── __init__.py
│   ├── test_engineering.py
│   ├── test_scalers.py
│   └── test_selection.py
│
├── test_integrations/       # Integration tests (19 tests)
│   ├── __init__.py
│   └── test_full_pipeline.py
│
└── test_models/             # Model tests (56 tests)
    ├── __init__.py
    ├── test_model_loader.py
    └── test_predictions.py
```

### Test File Naming

```python
# ✅ Correct
test_feature_name.py
test_module_name.py

# ❌ Incorrect
feature_test.py
module_tests.py
testing_module.py
```

### Test Class Naming

```python
# ✅ Correct
class TestFeatureName:
    """Test suite for specific feature."""
    pass

class TestModuleFunctionality:
    """Test suite for module functionality."""
    pass

# ❌ Incorrect
class FeatureTest:
    pass

class TestSuite:
    pass
```

### Test Method Naming

```python
# ✅ Correct - Descriptive and clear
def test_function_returns_correct_type(self):
    pass

def test_function_handles_empty_input(self):
    pass

def test_function_raises_error_on_invalid_input(self):
    pass

# ❌ Incorrect - Too vague
def test_function(self):
    pass

def test_1(self):
    pass

def test_basic(self):
    pass
```

---

## Running Tests

### Basic Commands

```bash
# All tests (verbose)
pytest tests/ -v

# Quick run (quiet mode)
pytest tests/ -q

# Very quick (show only summary)
pytest tests/ -q --tb=line

# With detailed output
pytest tests/ -vv
```

### Specific Tests

```bash
# Single module
pytest tests/test_features/ -v

# Single file
pytest tests/test_features/test_scalers.py -v

# Single class
pytest tests/test_features/test_scalers.py::TestFeatureScaler -v

# Single test
pytest tests/test_features/test_scalers.py::TestFeatureScaler::test_fit_returns_self -v
```

### Test Selection

```bash
# Run tests matching keyword
pytest tests/ -k "scaler"
pytest tests/ -k "test_fit"
pytest tests/ -k "not slow"

# Run by marker
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m "not slow"
```

### Parallel Execution

```bash
# Auto-detect CPU cores
pytest tests/ -n auto

# Specific number of workers
pytest tests/ -n 4

# With verbose output
pytest tests/ -n auto -v
```

### Failed Tests

```bash
# Run only failed tests from last run
pytest tests/ --lf

# Run failed first, then others
pytest tests/ --ff

# Stop at first failure
pytest tests/ -x

# Stop after N failures
pytest tests/ --maxfail=3
```

---

## Writing Tests

### Basic Test Structure

```python
"""
Test module for feature X.

This module tests the functionality of feature X including:
- Basic operations
- Edge cases
- Error handling
"""

import pytest
import pandas as pd
import numpy as np

from src.module import function_to_test


class TestFeatureName:
    """Test suite for feature name."""

    @pytest.fixture
    def sample_data(self):
        """
        Create sample data for tests.

        Returns:
            pd.DataFrame: Sample dataframe with test data
        """
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': ['A', 'B', 'A', 'B', 'A']
        })

    def test_basic_functionality(self, sample_data):
        """Test basic functionality with valid input."""
        # Arrange
        expected_type = pd.DataFrame

        # Act
        result = function_to_test(sample_data)

        # Assert
        assert isinstance(result, expected_type)
        assert len(result) == len(sample_data)
        assert list(result.columns) == list(sample_data.columns)

    def test_handles_empty_input(self):
        """Test behavior with empty input."""
        # Arrange
        empty_df = pd.DataFrame()

        # Act & Assert
        with pytest.raises(ValueError, match="Input cannot be empty"):
            function_to_test(empty_df)

    def test_preserves_data_integrity(self, sample_data):
        """Test that data integrity is maintained."""
        # Arrange
        original_sum = sample_data['feature1'].sum()

        # Act
        result = function_to_test(sample_data)

        # Assert
        assert result['feature1'].sum() == original_sum

    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
        (0, 0),
        (-1, -2)
    ])
    def test_parametrized_inputs(self, input_val, expected):
        """Test with multiple input-output pairs."""
        result = function_to_test(input_val)
        assert result == expected
```

### Test Template

```python
import pytest
import pandas as pd
import numpy as np

from src.module import FunctionOrClass


class TestFunctionOrClass:
    """Test suite for FunctionOrClass."""

    # ============== FIXTURES ==============

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

    # ============== BASIC TESTS ==============

    def test_returns_correct_type(self, sample_data):
        """Test return type is correct."""
        result = FunctionOrClass(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_shape(self, sample_data):
        """Test shape is preserved."""
        result = FunctionOrClass(sample_data)
        assert result.shape == sample_data.shape

    # ============== EDGE CASES ==============

    def test_handles_empty_input(self):
        """Test with empty DataFrame."""
        with pytest.raises(ValueError):
            FunctionOrClass(pd.DataFrame())

    def test_handles_single_row(self):
        """Test with single row."""
        df = pd.DataFrame({'col1': [1], 'col2': [2]})
        result = FunctionOrClass(df)
        assert len(result) == 1

    def test_handles_missing_values(self):
        """Test with NaN values."""
        df = pd.DataFrame({'col1': [1, np.nan, 3]})
        result = FunctionOrClass(df)
        assert not result.isnull().any().any()

    # ============== ERROR HANDLING ==============

    def test_raises_error_on_invalid_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError):
            FunctionOrClass("not a dataframe")

    def test_raises_error_on_missing_column(self):
        """Test error when required column is missing."""
        df = pd.DataFrame({'wrong_col': [1, 2, 3]})
        with pytest.raises(KeyError):
            FunctionOrClass(df)

    # ============== PARAMETRIZED TESTS ==============

    @pytest.mark.parametrize("input_data,expected_output", [
        ([1, 2, 3], [2, 4, 6]),
        ([0, 0, 0], [0, 0, 0]),
        ([-1, -2, -3], [-2, -4, -6])
    ])
    def test_parametrized(self, input_data, expected_output):
        """Test with multiple parameter combinations."""
        result = FunctionOrClass(input_data)
        assert result == expected_output
```

---

## Test Fixtures

### Shared Fixtures (conftest.py)

```python
# tests/conftest.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.choice(['A', 'B', 'C'], 100)
    })


@pytest.fixture
def sample_data_with_missing():
    """Create DataFrame with missing values."""
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10, np.nan, 30, 40, 50],
        'target': ['A', 'B', 'A', 'B', 'A']
    })
    return df


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_model(tmp_path):
    """Create a mock model for testing."""
    from src.models.base import BaseModel

    model = BaseModel()
    model.fit(X_train, y_train)

    # Save to temp path
    model_path = tmp_path / "model.pkl"
    model.save(model_path)

    return model_path
```

### Fixture Scopes

```python
# Function scope (default) - created for each test
@pytest.fixture
def function_scope_fixture():
    return "new instance for each test"

# Class scope - shared within test class
@pytest.fixture(scope="class")
def class_scope_fixture():
    return "shared within class"

# Module scope - shared within module
@pytest.fixture(scope="module")
def module_scope_fixture():
    return "shared within module"

# Session scope - shared across entire test session
@pytest.fixture(scope="session")
def session_scope_fixture():
    return "shared across all tests"
```

### Fixture Cleanup

```python
@pytest.fixture
def resource_with_cleanup(tmp_path):
    """Fixture with cleanup (teardown)."""
    # Setup
    file_path = tmp_path / "test_file.csv"
    df = pd.DataFrame({'col': [1, 2, 3]})
    df.to_csv(file_path)

    yield file_path  # Test runs here

    # Teardown (cleanup)
    if file_path.exists():
        file_path.unlink()
```

---

## Test Markers

### Built-in Markers

```python
# Skip test
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass

# Skip conditionally
@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_python310_feature():
    pass

# Expected to fail
@pytest.mark.xfail(reason="Known bug #123")
def test_known_bug():
    pass
```

### Custom Markers

```python
# pytest.ini
[pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests (>1s)
    fast: Fast tests (<0.1s)
    requires_data: Tests requiring external data
```

**Usage:**

```python
@pytest.mark.unit
def test_unit():
    """Fast unit test."""
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_integration():
    """Slow integration test."""
    pass

@pytest.mark.requires_data
def test_with_real_data():
    """Test with real dataset."""
    pass
```

**Running:**

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only fast tests
pytest tests/ -m fast

# Run unit but not slow
pytest tests/ -m "unit and not slow"

# Run integration or requires_data
pytest tests/ -m "integration or requires_data"
```

---

## Coverage

### Generate Coverage Report

```bash
# HTML report (recommended)
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Terminal report
pytest tests/ --cov=src --cov-report=term

# Terminal with missing lines
pytest tests/ --cov=src --cov-report=term-missing

# JSON report
pytest tests/ --cov=src --cov-report=json

# XML report (for CI/CD)
pytest tests/ --cov=src --cov-report=xml
```

### Coverage Configuration

```ini
# pytest.ini
[pytest]
addopts =
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=70
```

### Check Specific Module

```bash
# Only data module
pytest tests/test_data/ --cov=src.data --cov-report=term

# Only features module
pytest tests/test_features/ --cov=src.features --cov-report=html
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests with coverage
        run: |
          pytest tests/ --cov=src --cov-report=xml --cov-report=term

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: true
```

---

## Best Practices

### ✅ DO

```python
# 1. Use descriptive test names
def test_feature_scaler_preserves_dataframe_shape():
    pass

# 2. Use AAA pattern (Arrange-Act-Assert)
def test_function():
    # Arrange
    input_data = create_test_data()

    # Act
    result = function(input_data)

    # Assert
    assert result is not None

# 3. Test one thing per test
def test_returns_dataframe():
    result = function()
    assert isinstance(result, pd.DataFrame)

def test_preserves_row_count():
    result = function()
    assert len(result) == len(input_data)

# 4. Use fixtures for common setup
@pytest.fixture
def sample_data():
    return create_test_data()

def test_with_fixture(sample_data):
    result = function(sample_data)
    assert result is not None

# 5. Test edge cases
def test_handles_empty_input():
    with pytest.raises(ValueError):
        function(pd.DataFrame())
```

### ❌ DON'T

```python
# 1. Don't use vague test names
def test_1():  # ❌
    pass

def test_function():  # ❌ Too generic
    pass

# 2. Don't test multiple things in one test
def test_everything():  # ❌
    assert result.shape == (10, 5)
    assert result.dtype == 'float64'
    assert result.sum() > 0
    # Too many unrelated assertions

# 3. Don't use print statements (use logging if needed)
def test_debug():
    result = function()
    print(result)  # ❌ Use logging or pytest -s

# 4. Don't ignore warnings
def test_with_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ❌
        result = function()

# 5. Don't use sleep in tests
def test_async():
    start_process()
    time.sleep(5)  # ❌ Use proper async testing
    assert is_complete()
```

---

## Debugging Tests

### Using pdb

```bash
# Drop to debugger on failure
pytest tests/test_file.py --pdb

# Stop at first failure and debug
pytest tests/test_file.py -x --pdb
```

### Print Debugging

```bash
# Show print statements
pytest tests/test_file.py -s

# Show captured output even for passing tests
pytest tests/test_file.py -s --capture=no
```

### Verbose Output

```bash
# Very verbose
pytest tests/test_file.py -vv

# Show local variables on failure
pytest tests/test_file.py -l

# Show full diff
pytest tests/test_file.py -vv --tb=long
```

---

## Performance Testing

### Test Duration

```bash
# Show slowest 10 tests
pytest tests/ --durations=10

# Show all test durations
pytest tests/ --durations=0
```

### Profiling

```bash
# Profile specific test
py-spy record -o profile.svg -- pytest tests/test_slow.py

# Use pytest-benchmark
pytest tests/ --benchmark-only
```

---

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Test-Driven Development](https://www.amazon.com/Test-Driven-Development-Kent-Beck/dp/0321146530)
- [Python Testing with pytest](https://pragprog.com/titles/bopytest/python-testing-with-pytest/)

---

**Status:** ✅ Complete  
**Maintainer:** sulegogh  
**Last Updated:** 2025-11-13

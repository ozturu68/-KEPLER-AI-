# 🛠️ Development Guide

**Last Updated:** 2025-11-13  
**Maintainer:** sulegogh

## Table of Contents

- [Setup](#setup)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Git Workflow](#git-workflow)
- [Common Tasks](#common-tasks)

---

## Setup

### Prerequisites

```bash
# Required
- Python 3.10+
- Git
- pip

# Optional but recommended
- pyenv (Python version management)
- direnv (environment variable management)
- VS Code or PyCharm
```

### Initial Setup

```bash
# Clone repository
git clone https://github.com/sulegogh/kepler-new.git
cd kepler-new

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest tests/ -v
```

### IDE Setup

#### VS Code

```json
// .vscode/settings.json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "120"],
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "editor.formatOnSave": true,
  "editor.rulers": [120]
}
```

#### PyCharm

1. File → Settings → Python Interpreter
2. Select venv/bin/python
3. Enable pytest as test runner
4. Configure Black as formatter

---

## Development Workflow

### Branch Strategy

```
main           # Production-ready code
├── develop    # Integration branch (planned)
└── feature/*  # Feature branches
```

### Feature Development

```bash
# 1. Create feature branch
git checkout -b feature/my-new-feature

# 2. Make changes
# ... edit files ...

# 3. Run tests
pytest tests/ -v

# 4. Format code (pre-commit runs automatically)
black src/ tests/
isort src/ tests/

# 5. Commit
git add .
git commit -m "feat: Add my new feature"

# 6. Push
git push origin feature/my-new-feature

# 7. Create Pull Request on GitHub
```

---

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with modifications:

```python
# Line length: 120 characters (not 79)
# String quotes: Double quotes preferred
# Imports: Grouped and sorted by isort

# ✅ Good
def calculate_planetary_density(
    radius: float, mass: float, unit: str = "earth"
) -> float:
    """
    Calculate planetary density.

    Args:
        radius: Planet radius in specified unit
        mass: Planet mass in specified unit
        unit: Unit system ('earth', 'jupiter', 'solar')

    Returns:
        Density in g/cm³
    """
    if radius <= 0 or mass <= 0:
        raise ValueError("Radius and mass must be positive")

    return mass / (radius ** 3)

# ❌ Bad
def calc_dens(r,m,u='earth'):
    return m/(r**3)
```

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np

# ✅ Good
def process_features(
    df: pd.DataFrame,
    columns: List[str],
    scaler: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ...

# ❌ Bad
def process_features(df, columns, scaler=None):
    ...
```

### Docstrings

Use Google style docstrings:

```python
def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    **kwargs
) -> CatBoostClassifier:
    """
    Train CatBoost model on exoplanet data.

    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional CatBoost parameters

    Returns:
        Trained model instance

    Raises:
        ValueError: If data is invalid
        ModelTrainingError: If training fails

    Example:
        >>> model = train_model(X_train, y_train, iterations=1000)
        >>> predictions = model.predict(X_test)
    """
    ...
```

### Imports Organization

```python
# Standard library
import os
import sys
from pathlib import Path
from typing import List, Dict

# Third-party
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score

# Local
from src.core.constants import TARGET_COLUMN
from src.data.cleaners import clean_data
from src.features.engineering import ExoplanetFeatureEngineer
```

---

## Testing

### Test Structure

```python
import pytest
import pandas as pd
import numpy as np

from src.module import function_to_test


class TestFeatureName:
    """Test suite for specific feature."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

    def test_basic_functionality(self, sample_data):
        """Test basic use case."""
        result = function_to_test(sample_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_edge_case_empty_input(self):
        """Test with empty input."""
        with pytest.raises(ValueError):
            function_to_test(pd.DataFrame())

    def test_edge_case_invalid_input(self):
        """Test with invalid input."""
        with pytest.raises(TypeError):
            function_to_test("not a dataframe")
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_features/ -v

# Specific test
pytest tests/test_features/test_scalers.py::TestFeatureScaler -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Fast (no coverage)
pytest tests/ -q

# Parallel execution
pytest tests/ -n auto

# Only failed tests
pytest tests/ --lf

# Markers
pytest tests/ -m "unit"           # Only unit tests
pytest tests/ -m "integration"    # Only integration tests
pytest tests/ -m "not slow"       # Skip slow tests
```

### Test Markers

```python
@pytest.mark.unit
def test_unit():
    ...

@pytest.mark.integration
def test_integration():
    ...

@pytest.mark.slow
def test_slow_operation():
    ...

@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    ...

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6)
])
def test_parametrized(input, expected):
    assert input * 2 == expected
```

---

## Git Workflow

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format
<type>(<scope>): <subject>

<body>

<footer>

# Types
feat:     New feature
fix:      Bug fix
docs:     Documentation only
test:     Adding/updating tests
refactor: Code refactoring
chore:    Maintenance tasks
style:    Code style changes
perf:     Performance improvements

# Examples
feat(features): Add planetary density calculation
fix(data): Handle missing values in preprocessor
docs(api): Update API documentation
test(models): Add CatBoost training tests
refactor(core): Simplify label encoding logic
chore(deps): Update dependencies
```

### Pre-commit Hooks

Automatically run before each commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    hooks:
      - id: isort
```

Bypass hooks (use sparingly):

```bash
git commit -m "message" --no-verify
```

---

## Common Tasks

### Add New Feature Module

```bash
# 1. Create module file
touch src/features/new_feature.py

# 2. Create test file
touch tests/test_features/test_new_feature.py

# 3. Implement feature
# src/features/new_feature.py
"""
New feature implementation.
"""

def new_function():
    """Docstring."""
    pass

# 4. Write tests
# tests/test_features/test_new_feature.py
import pytest
from src.features.new_feature import new_function

def test_new_function():
    result = new_function()
    assert result is not None

# 5. Run tests
pytest tests/test_features/test_new_feature.py -v

# 6. Commit
git add src/features/new_feature.py tests/test_features/test_new_feature.py
git commit -m "feat(features): Add new feature"
```

### Update Dependencies

```bash
# Add new package
pip install new-package
pip freeze > requirements.txt

# Or use pip-tools
pip-compile requirements.in

# Update all packages
pip install --upgrade -r requirements.txt
```

### Generate Coverage Report

```bash
# HTML report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Terminal report
pytest tests/ --cov=src --cov-report=term-missing

# JSON report
pytest tests/ --cov=src --cov-report=json
```

### Debug Tests

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use pytest
pytest tests/test_file.py --pdb  # Drop to debugger on failure
pytest tests/test_file.py -x --pdb  # Stop at first failure

# Print debugging
pytest tests/test_file.py -s  # Show print statements
```

### Performance Profiling

```bash
# Profile test execution
pytest tests/ --durations=10

# Profile specific test
py-spy record -o profile.svg -- pytest tests/test_slow.py
```

---

## Troubleshooting

### Common Issues

**Issue:** Tests fail with import errors

```bash
# Solution: Ensure venv is activated and in project root
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue:** Pre-commit hooks fail

```bash
# Solution: Run manually and fix
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

**Issue:** Coverage too low

```bash
# Solution: Check which files are missing tests
pytest tests/ --cov=src --cov-report=term-missing
# Add tests for files with low coverage
```

---

## Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Type Hints (PEP 484)](https://peps.python.org/pep-0484/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## Getting Help

- **Issues:** Create GitHub issue
- **Discussions:** Use GitHub Discussions
- **Questions:** Open a discussion with `question` label

---

**Happy Coding! 🚀**

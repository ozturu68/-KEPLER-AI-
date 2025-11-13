# 🤝 Contributing Guide

**Last Updated:** 2025-11-13  
**Maintainer:** sulegogh

Kepler Exoplanet ML projesine katkıda bulunmak istediğiniz için teşekkürler! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Testing Requirements](#testing-requirements)

---

## Code of Conduct

### Our Pledge

Projemize katılan herkes için açık, hoş karşılayan, kapsayıcı ve taciz içermeyen bir ortam sağlamayı taahhüt ediyoruz.

### Our Standards

**✅ Olumlu Davranışlar:**

- Empati ve nezaket göstermek
- Farklı görüşlere ve deneyimlere saygılı olmak
- Yapıcı eleştiri vermek ve kabul etmek
- Topluluk için en iyisine odaklanmak
- Diğer topluluk üyelerine empati göstermek

**❌ Kabul Edilemez Davranışlar:**

- Cinsel içerikli dil veya görüntü kullanımı
- Trolleme, hakaret veya aşağılayıcı yorumlar
- Kişisel veya politik saldırılar
- Taciz (açık veya özel)
- Başkalarının özel bilgilerini izinsiz paylaşma

---

## Getting Started

### Prerequisites

```bash
# Required
- Python 3.10+
- Git
- GitHub account

# Recommended
- VS Code or PyCharm
- Basic machine learning knowledge
- Familiarity with pandas, scikit-learn
```

### Fork and Clone

```bash
# 1. Fork repository on GitHub
# Visit: https://github.com/sulegogh/kepler-new
# Click "Fork" button

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/kepler-new.git
cd kepler-new

# 3. Add upstream remote
git remote add upstream https://github.com/sulegogh/kepler-new.git

# 4. Verify remotes
git remote -v
# origin    https://github.com/YOUR_USERNAME/kepler-new.git (fetch)
# origin    https://github.com/YOUR_USERNAME/kepler-new.git (push)
# upstream  https://github.com/sulegogh/kepler-new.git (fetch)
# upstream  https://github.com/sulegogh/kepler-new.git (push)
```

### Setup Development Environment

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Install pre-commit hooks
pre-commit install

# 4. Run tests to verify setup
pytest tests/ -v

# 5. Check code quality
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
```

---

## How to Contribute

### Types of Contributions

#### 🐛 Bug Reports

**Before submitting:**

- Check existing [issues](https://github.com/sulegogh/kepler-new/issues)
- Verify bug in latest version
- Try to reproduce consistently

**Bug report should include:**

```markdown
### Bug Description

Clear description of the bug

### Steps to Reproduce

1. Step 1
2. Step 2
3. See error

### Expected Behavior

What should happen

### Actual Behavior

What actually happens

### Environment

- OS: Ubuntu 22.04
- Python: 3.10.12
- Version: v1.0.0

### Additional Context

Any other relevant information
```

---

#### ✨ Feature Requests

**Before requesting:**

- Check [existing issues](https://github.com/sulegogh/kepler-new/issues)
- Review [roadmap](../README.md#roadmap)
- Consider if it fits project scope

**Feature request template:**

```markdown
### Feature Description

Clear description of the feature

### Use Case

Why is this needed? Who benefits?

### Proposed Solution

How should it work?

### Alternatives Considered

What other approaches did you consider?

### Additional Context

Mockups, references, examples
```

---

#### 📝 Documentation

Documentation improvements are always welcome!

**Areas to contribute:**

- Fix typos and grammar
- Add examples and tutorials
- Improve clarity
- Add docstrings
- Create guides

---

#### 💻 Code Contributions

See [Development Workflow](#development-workflow)

---

## Development Workflow

### 1. Sync with Upstream

```bash
# Fetch latest changes
git fetch upstream

# Merge into your local main
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

### 2. Create Feature Branch

```bash
# Create and checkout new branch
git checkout -b feature/my-awesome-feature

# Branch naming conventions:
# feature/feature-name    - New features
# fix/bug-description     - Bug fixes
# docs/what-changed       - Documentation
# test/what-tested        - Test additions
# refactor/what-refactored - Code refactoring
# chore/what-maintained   - Maintenance tasks
```

### 3. Make Changes

```python
# Edit files
# src/module/new_feature.py

def new_awesome_function(data: pd.DataFrame) -> pd.DataFrame:
    """
    Do something awesome.

    Args:
        data: Input DataFrame

    Returns:
        Processed DataFrame

    Example:
        >>> df = pd.DataFrame({'col': [1, 2, 3]})
        >>> result = new_awesome_function(df)
    """
    # Implementation
    return processed_data
```

### 4. Write Tests

```python
# tests/test_module/test_new_feature.py

import pytest
from src.module.new_feature import new_awesome_function


class TestNewAwesomeFunction:
    """Test suite for new_awesome_function."""

    def test_returns_dataframe(self):
        """Test return type."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = new_awesome_function(df)
        assert isinstance(result, pd.DataFrame)

    def test_handles_empty_input(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            new_awesome_function(pd.DataFrame())
```

### 5. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_module/test_new_feature.py -v

# Check coverage
pytest tests/ --cov=src --cov-report=term-missing

# Coverage should be >70%
```

### 6. Format Code

```bash
# Auto-format with black
black src/ tests/

# Sort imports
isort src/ tests/

# Check linting
flake8 src/ tests/

# Or let pre-commit do it
git add .
git commit -m "feat: Add awesome feature"
# Pre-commit hooks run automatically
```

### 7. Commit Changes

```bash
# Stage changes
git add src/module/new_feature.py
git add tests/test_module/test_new_feature.py

# Commit with conventional commit message
git commit -m "feat(module): Add awesome feature

- Implement new_awesome_function
- Add comprehensive tests
- Update documentation

Closes #123"
```

### 8. Push to Fork

```bash
# Push feature branch to your fork
git push origin feature/my-awesome-feature
```

### 9. Create Pull Request

1. Visit your fork on GitHub
2. Click "Compare & pull request"
3. Fill in PR template
4. Submit for review

---

## Pull Request Process

### PR Checklist

Before submitting, ensure:

```markdown
- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] New tests added for new features
- [ ] Coverage remains >70%
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Pre-commit hooks pass
- [ ] Branch is up-to-date with main
- [ ] PR description is clear
```

### PR Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update

## Testing

Describe testing performed:

- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manual testing done

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] Coverage >70%

## Related Issues

Closes #123 Related to #456

## Screenshots (if applicable)

Add screenshots for UI changes
```

### Review Process

1. **Automated Checks:** CI/CD runs tests
2. **Code Review:** Maintainer reviews code
3. **Feedback:** Address review comments
4. **Approval:** PR approved by maintainer
5. **Merge:** PR merged into main

### After PR Merge

```bash
# Sync your fork
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# Delete feature branch
git branch -d feature/my-awesome-feature
git push origin --delete feature/my-awesome-feature
```

---

## Coding Standards

### Python Style

Follow [PEP 8](https://pep8.org/) with modifications:

```python
# Line length: 120 characters (not 79)
# String quotes: Double quotes preferred
# Imports: Sorted by isort

# ✅ Good
def calculate_feature(
    data: pd.DataFrame,
    column: str,
    method: str = "mean"
) -> float:
    """
    Calculate feature statistic.

    Args:
        data: Input DataFrame
        column: Column to process
        method: Calculation method

    Returns:
        Calculated value

    Raises:
        ValueError: If column not found
    """
    if column not in data.columns:
        raise ValueError(f"Column {column} not found")

    if method == "mean":
        return data[column].mean()
    elif method == "median":
        return data[column].median()
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np

# ✅ Good
def process_data(
    df: pd.DataFrame,
    columns: List[str],
    threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Process data with type hints."""
    ...

# ❌ Bad
def process_data(df, columns, threshold=None):
    """No type hints."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def complex_function(
    param1: str,
    param2: int,
    param3: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    One-line summary.

    Detailed description of function behavior.
    Can span multiple lines.

    Args:
        param1: Description of param1
        param2: Description of param2
        param3: Optional list of strings. Defaults to None.

    Returns:
        Dictionary containing results with keys:
        - 'result': Processing result
        - 'metadata': Additional information

    Raises:
        ValueError: If param2 is negative
        TypeError: If param1 is not string

    Example:
        >>> result = complex_function("test", 42)
        >>> print(result['result'])
        'processed'

    Note:
        This is a compute-intensive operation.

    Warning:
        Modifies input data in-place.
    """
    ...
```

### Import Organization

```python
# Standard library (alphabetical)
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Third-party packages (alphabetical)
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score

# Local imports (alphabetical)
from src.core.constants import TARGET_COLUMN
from src.data.cleaners import clean_data
from src.features.engineering import ExoplanetFeatureEngineer
```

---

## Commit Guidelines

### Conventional Commits

Format: `<type>(<scope>): <subject>`

**Types:**

```
feat:     New feature
fix:      Bug fix
docs:     Documentation only
test:     Adding/updating tests
refactor: Code refactoring
style:    Code style (formatting, missing semi-colons, etc)
perf:     Performance improvement
chore:    Maintenance (dependencies, build, etc)
ci:       CI/CD changes
build:    Build system changes
revert:   Revert previous commit
```

**Examples:**

```bash
# Feature
git commit -m "feat(features): Add planetary density calculation"

# Bug fix
git commit -m "fix(data): Handle missing values in preprocessor"

# Documentation
git commit -m "docs(api): Update API endpoint documentation"

# Test
git commit -m "test(models): Add CatBoost training tests"

# Refactor
git commit -m "refactor(core): Simplify label encoding logic"

# Breaking change
git commit -m "feat(api)!: Change prediction endpoint format

BREAKING CHANGE: Response format changed from array to object"
```

### Commit Message Body

```bash
git commit -m "feat(features): Add feature importance calculation

- Implement Random Forest based importance
- Add mutual information option
- Support custom feature selection
- Update tests and documentation

Closes #123
Related to #456"
```

---

## Testing Requirements

### Minimum Requirements

- **Coverage:** >70% for new code
- **All tests pass:** No failing tests
- **Test quality:** Meaningful assertions
- **Edge cases:** Test boundary conditions

### Test Structure

```python
class TestNewFeature:
    """Test suite for new feature."""

    # 1. Basic functionality
    def test_basic_operation(self):
        """Test normal use case."""
        pass

    # 2. Return types
    def test_returns_correct_type(self):
        """Test return type."""
        pass

    # 3. Edge cases
    def test_handles_empty_input(self):
        """Test with empty input."""
        pass

    def test_handles_single_item(self):
        """Test with single item."""
        pass

    # 4. Error handling
    def test_raises_error_on_invalid_input(self):
        """Test error handling."""
        pass

    # 5. Integration
    def test_integrates_with_pipeline(self):
        """Test integration."""
        pass
```

---

## Getting Help

### Resources

- **Documentation:** [docs/](../docs/)
- **Issues:** [GitHub Issues](https://github.com/sulegogh/kepler-new/issues)
- **Discussions:** [GitHub Discussions](https://github.com/sulegogh/kepler-new/discussions)

### Questions

1. Check documentation first
2. Search existing issues
3. Ask in Discussions
4. Create new issue with `question` label

---

## Recognition

Contributors will be recognized in:

- README.md (Contributors section)
- CHANGELOG.md (Release notes)
- GitHub Contributors page

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! 🎉**

**Maintainer:** sulegogh  
**Last Updated:** 2025-11-13

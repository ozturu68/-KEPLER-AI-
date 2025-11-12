"""
pytest configuration and fixtures
Shared test utilities and fixtures for all tests

Author: sulegogh
Date: 2025-11-12
Version: 1.0
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# FIXTURES: PATHS
# ============================================================================


@pytest.fixture
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_path():
    """Path to test data"""
    return Path('data/selected/test_selected.csv')


@pytest.fixture
def model_path():
    """Path to production model"""
    return Path('models/v2_final/catboost_v2_final.pkl')


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


# ============================================================================
# FIXTURES: LABELS
# ============================================================================


@pytest.fixture
def sample_labels_string():
    """Sample string labels for testing"""
    return np.array(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'])


@pytest.fixture
def sample_labels_numeric():
    """Sample numeric labels for testing"""
    return np.array([0, 1, 2, 0])


@pytest.fixture
def sample_labels_mixed():
    """Sample mixed format labels (invalid)"""
    return ['CANDIDATE', 0, 'CONFIRMED', 1]


@pytest.fixture
def label_map():
    """Standard label mapping"""
    return {'CANDIDATE': 0, 'CONFIRMED': 1, 'FALSE POSITIVE': 2}


@pytest.fixture
def reverse_label_map():
    """Reverse label mapping"""
    return {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}


# ============================================================================
# FIXTURES: PREDICTIONS
# ============================================================================


@pytest.fixture
def sample_predictions_string():
    """Sample string predictions"""
    return np.array(['FALSE POSITIVE', 'FALSE POSITIVE', 'CONFIRMED', 'CANDIDATE'])


@pytest.fixture
def sample_predictions_2d():
    """Sample 2D predictions (needs flatten)"""
    return np.array([['FALSE POSITIVE'], ['CONFIRMED'], ['CANDIDATE']])


@pytest.fixture
def sample_predictions_numeric():
    """Sample numeric predictions"""
    return np.array([2, 2, 1, 0])


# ============================================================================
# FIXTURES: DATA
# ============================================================================


@pytest.fixture
def sample_test_data():
    """Sample test DataFrame with 3 samples"""
    return pd.DataFrame(
        {
            'koi_score': [0.9, 0.7, 0.3],
            'koi_period': [10.5, 20.3, 5.7],
            'koi_depth': [100.0, 200.0, 50.0],
            'koi_disposition': ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
        }
    )


@pytest.fixture
def sample_features_50():
    """Sample feature matrix with 50 features (model input)"""
    np.random.seed(42)
    return pd.DataFrame(np.random.randn(10, 50), columns=[f'feature_{i}' for i in range(50)])


@pytest.fixture
def sample_probabilities():
    """Sample prediction probabilities"""
    return np.array(
        [
            [0.8, 0.15, 0.05],  # High CANDIDATE
            [0.1, 0.85, 0.05],  # High CONFIRMED
            [0.05, 0.10, 0.85],  # High FALSE POSITIVE
        ]
    )


# ============================================================================
# FIXTURES: MODEL
# ============================================================================


@pytest.fixture
def mock_model_dict():
    """Mock model dictionary structure (for testing loader)"""
    from unittest.mock import MagicMock

    mock_model = MagicMock()
    mock_model.predict = MagicMock(return_value=np.array(['CANDIDATE', 'CONFIRMED']))
    mock_model.predict_proba = MagicMock(return_value=np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]))

    return {
        'model': mock_model,
        'model_name': 'CatBoost',
        'params': {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'class_weights': [3.0, 1.0, 0.5],
        },
        'feature_names': [f'feature_{i}' for i in range(50)],
        'training_history': {'learn': [0.5, 0.4, 0.3], 'validation': [0.6, 0.5, 0.4]},
        'training_time': 5.87,
        'created_at': '2025-11-11 22:13:54',
        'is_trained': True,
    }


# ============================================================================
# FIXTURES: METRICS
# ============================================================================


@pytest.fixture
def perfect_predictions():
    """Perfect predictions (100% accuracy)"""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    return y_true, y_pred


@pytest.fixture
def poor_predictions():
    """Poor predictions (random)"""
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([2, 1, 2, 0, 2, 0, 1, 0, 1])
    return y_true, y_pred


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_sample_csv(path: Path, n_samples: int = 100):
    """Helper function to create sample CSV file"""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'koi_disposition': np.random.choice(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'], n_samples),
        }
    )
    df.to_csv(path, index=False)
    return path


# ============================================================================
# PYTEST HOOKS
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "model: Tests requiring model")
    config.addinivalue_line("markers", "data: Tests requiring data")

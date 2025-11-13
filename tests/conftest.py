"""
pytest configuration and fixtures
Shared test utilities and fixtures for all tests

Author: sulegogh
Date: 2025-11-13
Version: 3.0 (Updated with features marker and more fixtures)
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
    return Path("data/selected/test_selected.csv")


@pytest.fixture
def model_path():
    """Path to production model"""
    return Path("models/v2_final/catboost_v2_final.pkl")


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
    return np.array(["CANDIDATE", "CONFIRMED", "FALSE POSITIVE", "CANDIDATE"])


@pytest.fixture
def sample_labels_numeric():
    """Sample numeric labels for testing"""
    return np.array([0, 1, 2, 0])


@pytest.fixture
def sample_labels_mixed():
    """Sample mixed format labels (invalid)"""
    return ["CANDIDATE", 0, "CONFIRMED", 1]


@pytest.fixture
def label_map():
    """Standard label mapping"""
    return {"CANDIDATE": 0, "CONFIRMED": 1, "FALSE POSITIVE": 2}


@pytest.fixture
def reverse_label_map():
    """Reverse label mapping"""
    return {0: "CANDIDATE", 1: "CONFIRMED", 2: "FALSE POSITIVE"}


# ============================================================================
# FIXTURES: PREDICTIONS
# ============================================================================


@pytest.fixture
def sample_predictions_string():
    """Sample string predictions"""
    return np.array(["FALSE POSITIVE", "FALSE POSITIVE", "CONFIRMED", "CANDIDATE"])


@pytest.fixture
def sample_predictions_2d():
    """Sample 2D predictions (needs flatten)"""
    return np.array([["FALSE POSITIVE"], ["CONFIRMED"], ["CANDIDATE"]])


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
            "koi_score": [0.9, 0.7, 0.3],
            "koi_period": [10.5, 20.3, 5.7],
            "koi_depth": [100.0, 200.0, 50.0],
            "koi_disposition": ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"],
        }
    )


@pytest.fixture
def sample_features_50():
    """Sample feature matrix with 50 features (model input)"""
    np.random.seed(42)
    return pd.DataFrame(np.random.randn(10, 50), columns=[f"feature_{i}" for i in range(50)])


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


@pytest.fixture
def sample_data_with_target():
    """Sample DataFrame with target column for cleaning and preprocessing tests"""
    # Create larger dataset for proper stratified splitting (30 samples)
    np.random.seed(42)
    n_samples = 30  # 10 per class - enough for stratified split

    return pd.DataFrame(
        {
            "koi_disposition": (["CONFIRMED"] * 10 + ["CANDIDATE"] * 10 + ["FALSE POSITIVE"] * 10),
            "koi_score": np.random.uniform(0.3, 0.9, n_samples),
            "koi_period": np.random.uniform(5.0, 50.0, n_samples),
            "koi_depth": np.random.uniform(50.0, 500.0, n_samples),
        }
    )


@pytest.fixture
def sample_data_with_duplicates():
    """Sample DataFrame with duplicate rows"""
    return pd.DataFrame(
        {
            "koi_disposition": ["CONFIRMED", "CANDIDATE", "CANDIDATE", "CONFIRMED"],
            "koi_score": [0.9, 0.7, 0.7, 0.9],
            "koi_period": [10.5, 20.3, 20.3, 10.5],
        }
    )


@pytest.fixture
def sample_data_with_outliers():
    """Sample DataFrame with outliers"""
    return pd.DataFrame(
        {
            "koi_disposition": ["CONFIRMED"] * 6,
            "koi_score": [0.8, 0.85, 0.9, 0.87, 0.82, 10.0],  # 10.0 is outlier
            "koi_period": [10, 11, 12, 11.5, 10.5, 1000],  # 1000 is outlier
        }
    )


@pytest.fixture
def sample_data_with_missing():
    """Sample DataFrame with missing values"""
    return pd.DataFrame(
        {
            "koi_disposition": ["CONFIRMED", "CANDIDATE", np.nan, "CONFIRMED"],
            "koi_score": [0.9, np.nan, 0.3, 0.85],
            "koi_period": [10.5, 20.3, np.nan, 12.1],
        }
    )


# ============================================================================
# FIXTURES: FEATURE ENGINEERING DATA
# ============================================================================


@pytest.fixture
def sample_exoplanet_data():
    """Sample exoplanet data with typical KOI features"""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "koi_period": [10.5, 20.3, 5.7, 30.2, 15.8],
            "koi_depth": [100.0, 200.0, 50.0, 150.0, 120.0],
            "koi_duration": [2.5, 3.0, 1.5, 4.0, 2.8],
            "koi_impact": [0.5, 0.7, 0.3, 0.6, 0.4],
            "koi_prad": [2.0, 3.5, 1.5, 2.8, 2.2],
            "koi_teq": [300, 450, 200, 380, 320],
        }
    )


@pytest.fixture
def sample_numerical_features():
    """Sample numerical features for scaling/normalization"""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.uniform(0, 100, 20),
            "feature2": np.random.uniform(0, 1, 20),
            "feature3": np.random.normal(50, 10, 20),
        }
    )


# ============================================================================
# FIXTURES: MODEL
# ============================================================================


@pytest.fixture
def mock_model_dict():
    """Mock model dictionary structure (for testing loader)"""
    from unittest.mock import MagicMock

    mock_model = MagicMock()
    mock_model.predict = MagicMock(return_value=np.array(["CANDIDATE", "CONFIRMED"]))
    mock_model.predict_proba = MagicMock(return_value=np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]))

    return {
        "model": mock_model,
        "model_name": "CatBoost",
        "params": {
            "iterations": 1000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3,
            "class_weights": [3.0, 1.0, 0.5],
        },
        "feature_names": [f"feature_{i}" for i in range(50)],
        "training_history": {
            "learn": [0.5, 0.4, 0.3],
            "validation": [0.6, 0.5, 0.4],
        },
        "training_time": 5.87,
        "created_at": "2025-11-11 22:13:54",
        "is_trained": True,
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
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "koi_disposition": np.random.choice(["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"], n_samples),
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
    config.addinivalue_line("markers", "features: Feature engineering tests")
    config.addinivalue_line("markers", "evaluation: Evaluation and metrics tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "cli: CLI command tests")

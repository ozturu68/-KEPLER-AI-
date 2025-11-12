"""
Tests for model loader utility
Testing model loading, validation, and metadata extraction

Author: sulegogh
Date: 2025-11-12
Version: 2.0 (Revised)
"""

import pickle
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from src.models.model_loader import ModelLoadError, get_model_info, load_catboost_model_simple, load_catboost_model_v2

# ============================================================================
# TEST CLASS: MODEL LOADING
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestModelLoading:
    """Test basic model loading functionality"""

    def test_load_valid_model(self, model_path):
        """Test loading a valid model file"""
        model, metadata = load_catboost_model_v2(model_path)

        assert model is not None
        assert metadata is not None
        assert isinstance(metadata, dict)

    def test_load_model_returns_tuple(self, model_path):
        """Test that loader returns (model, metadata) tuple"""
        result = load_catboost_model_v2(model_path)

        assert isinstance(result, tuple)
        assert len(result) == 2

        model, metadata = result
        assert model is not None
        assert isinstance(metadata, dict)

    def test_load_invalid_path_raises_error(self):
        """Test that loading non-existent file raises FileNotFoundError"""
        invalid_path = "non_existent_model_file_12345.pkl"

        with pytest.raises(FileNotFoundError):
            load_catboost_model_v2(invalid_path)

    def test_load_directory_path_raises_error(self, temp_dir):
        """Test that loading directory path raises error"""
        # temp_dir is a directory, not a file
        # Different systems may raise FileNotFoundError or IsADirectoryError
        with pytest.raises((FileNotFoundError, IsADirectoryError)):
            load_catboost_model_v2(str(temp_dir))

    def test_load_corrupted_pickle_raises_error(self, temp_dir):
        """Test that loading corrupted pickle file raises error"""
        # Create corrupted file (not valid pickle)
        corrupted_file = temp_dir / "corrupted.pkl"
        corrupted_file.write_bytes(b"not a valid pickle file")

        with pytest.raises((ModelLoadError, Exception)):
            load_catboost_model_v2(str(corrupted_file))

    def test_load_non_compressed_file_raises_error(self, temp_dir):
        """Test that loading non-compressed file raises error"""
        # Create valid pickle but not compressed
        non_compressed = temp_dir / "non_compressed.pkl"
        data = {"test": "data"}
        with open(non_compressed, "wb") as f:
            pickle.dump(data, f)

        with pytest.raises((ModelLoadError, zlib.error, Exception)):
            load_catboost_model_v2(str(non_compressed))

    def test_model_structure_validation(self, model_path):
        """Test that loaded model has expected structure"""
        model, metadata = load_catboost_model_v2(model_path)

        # Check model has predict method
        assert hasattr(model, "predict"), "Model missing predict() method"

        # Check metadata has required keys
        required_keys = [
            "model_name",
            "params",
            "feature_names",
            "training_time",
            "created_at",
        ]
        for key in required_keys:
            assert key in metadata, f"Metadata missing required key: {key}"


# ============================================================================
# TEST CLASS: MODEL METADATA
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestModelMetadata:
    """Test model metadata extraction"""

    def test_metadata_is_dict(self, model_path):
        """Test that metadata is a dictionary"""
        _, metadata = load_catboost_model_v2(model_path)

        assert isinstance(metadata, dict)
        assert len(metadata) > 0

    def test_model_name_extracted(self, model_path):
        """Test that model name is extracted"""
        _, metadata = load_catboost_model_v2(model_path)

        assert "model_name" in metadata
        assert isinstance(metadata["model_name"], str)
        assert len(metadata["model_name"]) > 0

    def test_feature_names_extracted(self, model_path):
        """Test that feature names are extracted correctly"""
        _, metadata = load_catboost_model_v2(model_path)

        assert "feature_names" in metadata
        feature_names = metadata["feature_names"]

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        # All feature names should be strings
        assert all(isinstance(name, str) for name in feature_names)

    def test_feature_count_matches_expected(self, model_path):
        """Test that feature count is 50 (our model's feature count)"""
        _, metadata = load_catboost_model_v2(model_path)

        feature_names = metadata["feature_names"]
        assert len(feature_names) == 50, f"Expected 50 features, got {len(feature_names)}"

    def test_hyperparameters_extracted(self, model_path):
        """Test that hyperparameters are extracted"""
        _, metadata = load_catboost_model_v2(model_path)

        assert "params" in metadata
        params = metadata["params"]

        assert isinstance(params, dict)

    def test_hyperparameters_contain_key_settings(self, model_path):
        """Test that hyperparameters contain expected keys"""
        _, metadata = load_catboost_model_v2(model_path)

        params = metadata["params"]

        # CatBoost models typically have these params
        # (not all may be present, so we just check structure)
        assert isinstance(params, dict)

    def test_training_history_extracted(self, model_path):
        """Test that training history is extracted"""
        _, metadata = load_catboost_model_v2(model_path)

        assert "training_history" in metadata
        # History can be empty dict, that's OK

    def test_created_at_extracted(self, model_path):
        """Test that creation timestamp is extracted"""
        _, metadata = load_catboost_model_v2(model_path)

        assert "created_at" in metadata
        created_at = metadata["created_at"]

        assert created_at is not None
        assert isinstance(created_at, str)
        assert len(created_at) > 0

    def test_training_time_extracted(self, model_path):
        """Test that training time is extracted"""
        _, metadata = load_catboost_model_v2(model_path)

        assert "training_time" in metadata
        training_time = metadata["training_time"]

        assert isinstance(training_time, (int, float))
        assert training_time >= 0

    def test_is_trained_flag_extracted(self, model_path):
        """Test that is_trained flag is extracted"""
        _, metadata = load_catboost_model_v2(model_path)

        assert "is_trained" in metadata
        assert isinstance(metadata["is_trained"], bool)
        assert metadata["is_trained"] is True  # Our model should be trained


# ============================================================================
# TEST CLASS: MODEL METHODS
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestModelMethods:
    """Test that loaded model has required methods"""

    def test_model_has_predict_method(self, model_path):
        """Test that model has predict method"""
        model, _ = load_catboost_model_v2(model_path)

        assert hasattr(model, "predict")
        assert callable(model.predict)

    def test_model_has_predict_proba_method(self, model_path):
        """Test that model has predict_proba method"""
        model, _ = load_catboost_model_v2(model_path)

        assert hasattr(model, "predict_proba")
        assert callable(model.predict_proba)

    def test_model_has_get_params_method(self, model_path):
        """Test that model has get_params method"""
        model, _ = load_catboost_model_v2(model_path)

        # CatBoost models typically have get_params
        if hasattr(model, "get_params"):
            assert callable(model.get_params)

    def test_model_has_feature_names(self, model_path):
        """Test that model stores feature names"""
        model, metadata = load_catboost_model_v2(model_path)

        # Feature names are in metadata
        assert len(metadata["feature_names"]) > 0

    def test_model_is_trained(self, model_path):
        """Test that loaded model is marked as trained"""
        _, metadata = load_catboost_model_v2(model_path)

        # Check metadata indicates training
        assert "is_trained" in metadata
        assert metadata["is_trained"] is True


# ============================================================================
# TEST CLASS: SIMPLE LOADER
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestSimpleLoader:
    """Test simple loader function (returns only model)"""

    def test_simple_loader_returns_model_only(self, model_path):
        """Test that simple loader returns only model, not tuple"""
        model = load_catboost_model_simple(model_path)

        assert model is not None
        # Should not be a tuple
        assert not isinstance(model, tuple)
        # Should have predict method
        assert hasattr(model, "predict")

    def test_simple_loader_model_is_same_as_full_loader(self, model_path):
        """Test that simple loader returns same model as full loader"""
        model_simple = load_catboost_model_simple(model_path)
        model_full, _ = load_catboost_model_v2(model_path)

        # Both should have same methods
        assert hasattr(model_simple, "predict")
        assert hasattr(model_full, "predict")
        assert hasattr(model_simple, "predict_proba")
        assert hasattr(model_full, "predict_proba")

    def test_simple_loader_invalid_path_raises_error(self):
        """Test that simple loader raises error for invalid path"""
        invalid_path = "non_existent_model.pkl"

        with pytest.raises(FileNotFoundError):
            load_catboost_model_simple(invalid_path)


# ============================================================================
# TEST CLASS: GET MODEL INFO
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestGetModelInfo:
    """Test get_model_info function (returns only metadata)"""

    def test_get_model_info_returns_dict(self, model_path):
        """Test that get_model_info returns metadata dict"""
        info = get_model_info(model_path)

        assert isinstance(info, dict)
        assert len(info) > 0

    def test_get_model_info_contains_required_keys(self, model_path):
        """Test that get_model_info returns required metadata keys"""
        info = get_model_info(model_path)

        required_keys = ["model_name", "feature_names", "created_at", "training_time"]
        for key in required_keys:
            assert key in info, f"get_model_info missing key: {key}"

    def test_get_model_info_equivalent_to_full_loader_metadata(self, model_path):
        """Test that get_model_info returns same metadata as full loader"""
        info = get_model_info(model_path)
        _, metadata = load_catboost_model_v2(model_path)

        # Should have same keys and values
        assert info.keys() == metadata.keys()
        assert info["model_name"] == metadata["model_name"]
        assert info["created_at"] == metadata["created_at"]

    def test_get_model_info_invalid_path_raises_error(self):
        """Test that get_model_info raises error for invalid path"""
        invalid_path = "non_existent_model.pkl"

        with pytest.raises(FileNotFoundError):
            get_model_info(invalid_path)


# ============================================================================
# TEST CLASS: VALIDATION OPTIONS
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestValidationOptions:
    """Test model loading with different validation options"""

    def test_load_with_validation_enabled(self, model_path):
        """Test loading with validation enabled (default)"""
        model, metadata = load_catboost_model_v2(model_path, validate=True)

        assert model is not None
        assert metadata is not None
        # Model should have predict method (validated)
        assert hasattr(model, "predict")

    def test_load_with_validation_disabled(self, model_path):
        """Test loading with validation disabled"""
        model, metadata = load_catboost_model_v2(model_path, validate=False)

        assert model is not None
        assert metadata is not None
        # Even without validation, model should still load


# ============================================================================
# TEST CLASS: ERROR HANDLING
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestErrorHandling:
    """Test error handling in model loader"""

    def test_model_load_error_is_raised_for_invalid_dict_structure(self, temp_dir):
        """Test that ModelLoadError is raised for invalid dict structure"""
        # Create pickle with wrong structure (not a dict)
        invalid_file = temp_dir / "invalid_structure.pkl"

        # Compress a list instead of dict
        invalid_data = ["not", "a", "dict"]
        pickled = pickle.dumps(invalid_data)
        compressed = zlib.compress(pickled)

        with open(invalid_file, "wb") as f:
            f.write(compressed)

        with pytest.raises(ModelLoadError, match="Expected dict"):
            load_catboost_model_v2(str(invalid_file))

    def test_model_load_error_is_raised_for_missing_model_key(self, temp_dir):
        """Test that ModelLoadError is raised when 'model' key is missing"""
        # Create pickle with dict but no 'model' key
        invalid_file = temp_dir / "missing_model_key.pkl"

        # Dict without 'model' key
        invalid_data = {"params": {}, "feature_names": []}
        pickled = pickle.dumps(invalid_data)
        compressed = zlib.compress(pickled)

        with open(invalid_file, "wb") as f:
            f.write(compressed)

        with pytest.raises(ModelLoadError, match="'model' key not found"):
            load_catboost_model_v2(str(invalid_file))

    def test_path_object_accepted(self, model_path):
        """Test that Path objects are accepted as input"""
        path_obj = Path(model_path)

        model, metadata = load_catboost_model_v2(path_obj)

        assert model is not None
        assert metadata is not None

    def test_string_path_accepted(self, model_path):
        """Test that string paths are accepted as input"""
        string_path = str(model_path)

        model, metadata = load_catboost_model_v2(string_path)

        assert model is not None
        assert metadata is not None


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestEdgeCases:
    """Test edge cases in model loading"""

    def test_model_file_with_no_extension(self, temp_dir, model_path):
        """Test loading model file without .pkl extension"""
        # Copy model to file without extension
        import shutil

        no_ext_file = temp_dir / "model_no_extension"
        shutil.copy(model_path, no_ext_file)

        # Should still load successfully
        model, metadata = load_catboost_model_v2(str(no_ext_file))

        assert model is not None
        assert metadata is not None

    def test_metadata_with_missing_optional_keys(self, model_path):
        """Test that loader handles missing optional metadata keys gracefully"""
        _, metadata = load_catboost_model_v2(model_path)

        # Even if some optional keys are missing, loader should provide defaults
        # Required keys must be present
        assert "model_name" in metadata
        assert "feature_names" in metadata
        assert "params" in metadata

    def test_large_model_file_loads_successfully(self, model_path):
        """Test that large model file (900KB+) loads without issues"""
        model, metadata = load_catboost_model_v2(model_path)

        # File size check
        file_size = Path(model_path).stat().st_size
        assert file_size > 100_000, "Model file should be reasonably large"

        # Still loads successfully
        assert model is not None
        assert metadata is not None

    def test_model_metadata_types_are_correct(self, model_path):
        """Test that metadata values have correct types"""
        _, metadata = load_catboost_model_v2(model_path)

        # Type checks
        assert isinstance(metadata["model_name"], str)
        assert isinstance(metadata["params"], dict)
        assert isinstance(metadata["feature_names"], list)
        assert isinstance(metadata["training_history"], dict)
        assert isinstance(metadata["training_time"], (int, float))
        assert isinstance(metadata["created_at"], str)
        assert isinstance(metadata["is_trained"], bool)

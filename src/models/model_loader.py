"""
Model Loader Utility - Production Ready
Handles zlib compressed pickle with dictionary wrapper format

Author: sulegogh
Date: 2025-11-12
Version: 1.0
"""

import logging
import pickle
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Custom exception for model loading errors"""

    pass


def load_catboost_model_v2(model_path: str, validate: bool = True) -> tuple[Any, dict[str, Any]]:
    """
    Load CatBoost model from zlib compressed pickle (dictionary wrapper format)

    Args:
        model_path: Path to .pkl file
        validate: Whether to validate model structure

    Returns:
        tuple: (model, metadata)
            - model: CatBoost model object
            - metadata: Dictionary with params, feature_names, history, etc.

    Raises:
        ModelLoadError: If model cannot be loaded or is invalid
        FileNotFoundError: If model file doesn't exist

    Example:
        >>> model, metadata = load_catboost_model_v2('model.pkl')
        >>> predictions = model.predict(X_test)
        >>> print(f"Features: {len(metadata['feature_names'])}")
    """
    model_path = Path(model_path)

    # Check file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")

    try:
        # Read compressed data
        with open(model_path, 'rb') as f:
            compressed = f.read()

        logger.info(f"Read {len(compressed):,} bytes (compressed)")

        # Decompress
        try:
            decompressed = zlib.decompress(compressed)
            logger.info(f"Decompressed to {len(decompressed):,} bytes")
        except zlib.error as e:
            raise ModelLoadError(f"Decompression failed: {str(e)}")

        # Unpickle
        try:
            data = pickle.loads(decompressed)
        except pickle.UnpicklingError as e:
            raise ModelLoadError(f"Unpickling failed: {str(e)}")

        # Validate structure
        if not isinstance(data, dict):
            raise ModelLoadError(f"Expected dict, got {type(data).__name__}")

        if 'model' not in data:
            raise ModelLoadError(f"'model' key not found. Available keys: {list(data.keys())}")

        # Extract model
        model = data['model']

        # Validate model has required methods
        if validate:
            if not hasattr(model, 'predict'):
                raise ModelLoadError("Model doesn't have predict() method")

            if not hasattr(model, 'predict_proba'):
                logger.warning("Model doesn't have predict_proba() method")

        # Extract metadata
        metadata = {
            'model_name': data.get('model_name', 'Unknown'),
            'params': data.get('params', {}),
            'feature_names': data.get('feature_names', []),
            'training_history': data.get('training_history', {}),
            'training_time': data.get('training_time', 0),
            'created_at': data.get('created_at', 'Unknown'),
            'is_trained': data.get('is_trained', False),
        }

        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Model: {metadata['model_name']}")
        logger.info(f"   Created: {metadata['created_at']}")
        logger.info(f"   Features: {len(metadata['feature_names'])}")

        return model, metadata

    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


def load_catboost_model_simple(model_path: str) -> Any:
    """
    Simple loader - only returns model object

    Args:
        model_path: Path to .pkl file

    Returns:
        CatBoost model object

    Example:
        >>> model = load_catboost_model_simple('model.pkl')
        >>> predictions = model.predict(X_test)
    """
    model, _ = load_catboost_model_v2(model_path, validate=True)
    return model


def get_model_info(model_path: str) -> dict[str, Any]:
    """
    Get model metadata without loading the full model

    Args:
        model_path: Path to .pkl file

    Returns:
        Dictionary with model metadata

    Example:
        >>> info = get_model_info('model.pkl')
        >>> print(info['created_at'])
    """
    _, metadata = load_catboost_model_v2(model_path, validate=False)
    return metadata


# CLI usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_loader.py <model_path>")
        print("\nExample:")
        print("  python model_loader.py models/v2_final/catboost_v2_final.pkl")
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        model, metadata = load_catboost_model_v2(model_path)

        print("\n" + "=" * 70)
        print("MODEL INFORMATION")
        print("=" * 70)
        print(f"\nName: {metadata['model_name']}")
        print(f"Created: {metadata['created_at']}")
        print(f"Features: {len(metadata['feature_names'])}")
        print(f"Training time: {metadata['training_time']:.2f}s")
        print(f"Trained: {metadata['is_trained']}")

        if metadata['params']:
            print(f"\nHyperparameters:")
            for key, value in list(metadata['params'].items())[:10]:
                print(f"  {key}: {value}")

            if len(metadata['params']) > 10:
                print(f"  ... and {len(metadata['params']) - 10} more")

        print("\n✅ Model is valid and ready to use")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

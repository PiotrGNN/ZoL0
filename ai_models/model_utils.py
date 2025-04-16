"""
model_utils.py
--------------
Utility functions for AI model management.
"""

import os
import logging
import pickle
import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global constants
MODELS_DIR = "models"
METADATA_EXTENSION = "_metadata.json"
MODEL_EXTENSION = "_model.pkl"

def ensure_model_directory():
    """Ensure the models directory exists."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.abspath(MODELS_DIR)

def get_model_path(model_name: str) -> str:
    """Get the path for a model file."""
    model_name = model_name.lower().replace(" ", "_")
    return os.path.join(MODELS_DIR, f"{model_name}{MODEL_EXTENSION}")

def get_metadata_path(model_name: str) -> str:
    """Get the path for a model's metadata file."""
    model_name = model_name.lower().replace(" ", "_")
    return os.path.join(MODELS_DIR, f"{model_name}{METADATA_EXTENSION}")

def save_model_with_metadata(model: Any, model_name: str, metadata: Dict[str, Any]) -> bool:
    """
    Save a model along with its metadata.

    Args:
        model: The model object to save
        model_name: Name of the model
        metadata: Dictionary of metadata about the model

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        ensure_model_directory()

        # Add timestamp if not present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()

        # Add model name to metadata
        metadata['model_name'] = model_name

        # Save model
        model_path = get_model_path(model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save metadata
        metadata_path = get_metadata_path(model_name)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model {model_name} saved successfully with metadata")
        return True
    except Exception as e:
        logger.error(f"Error saving model {model_name}: {e}")
        return False

def load_model_with_metadata(model_name: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Load a model and its metadata.

    Args:
        model_name: Name of the model to load

    Returns:
        Tuple of (model, metadata) or (None, None) if loading fails
    """
    try:
        model_path = get_model_path(model_name)
        metadata_path = get_metadata_path(model_name)

        # Check if files exist
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None, None

        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}")

        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load metadata if available
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        logger.info(f"Model {model_name} loaded successfully")
        return model, metadata
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None, None

def model_needs_update(model_name: str, data_hash: str) -> bool:
    """
    Check if a model needs to be updated based on data hash.

    Args:
        model_name: Name of the model
        data_hash: Hash of current data

    Returns:
        bool: True if model needs update, False otherwise
    """
    try:
        metadata_path = get_metadata_path(model_name)

        # If metadata doesn't exist, model needs update
        if not os.path.exists(metadata_path):
            return True

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check if data hash exists and matches
        if 'data_hash' not in metadata:
            return True

        return metadata['data_hash'] != data_hash
    except Exception as e:
        logger.error(f"Error checking if model {model_name} needs update: {e}")
        return True  # Assume update needed on error

def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available models with their metadata.

    Returns:
        Dict mapping model names to their metadata
    """
    models = {}

    try:
        ensure_model_directory()

        for filename in os.listdir(MODELS_DIR):
            if filename.endswith(METADATA_EXTENSION):
                model_name = filename[:-len(METADATA_EXTENSION)]
                metadata_path = os.path.join(MODELS_DIR, filename)

                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    models[model_name] = metadata
                except Exception as e:
                    logger.error(f"Error loading metadata for {model_name}: {e}")
    except Exception as e:
        logger.error(f"Error listing models: {e}")

    return models

def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """
    Load metadata for a model file.

    Args:
        model_path: Path to the model file

    Returns:
        Dict containing metadata or empty dict if not found
    """
    try:
        # Get the metadata path
        base_path = os.path.splitext(model_path)[0]
        metadata_path = f"{base_path}{METADATA_EXTENSION}"

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        logger.error(f"Error loading model metadata: {e}")
        return {}
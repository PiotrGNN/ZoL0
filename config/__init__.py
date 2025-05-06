"""Configuration management package."""

import os
import importlib
from typing import List, Dict, Any
from pathlib import Path
import logging
from .config_loader import ConfigLoader, merge_dicts
from data.logging.system_logger import get_logger

logger = get_logger()

def _is_python_file(file_path: Path) -> bool:
    """Check if file is a Python module."""
    return file_path.is_file() and file_path.suffix == '.py' and not file_path.name.startswith('_')

def _is_excluded_package(package_name: str) -> bool:
    """Check if package should be excluded from auto-import."""
    excluded = {'tests', 'examples', '__pycache__'}
    if package_name in excluded:
        logger.log_info(f"Excluding package '{package_name}' from auto-import")
        return True
    return False

def _import_module(module_path: str, package: str) -> None:
    """Import a module and handle any errors."""
    try:
        importlib.import_module(module_path, package)
    except ImportError as e:
        logger.log_warning(
            f"Skipping package '{package}' due to error: {str(e)}",
            warning_type="ImportWarning"
        )

def _import_all_modules_from_directory(directory: Path, package: str) -> None:
    """Import all Python modules from a directory."""
    for item in directory.iterdir():
        if item.is_file() and _is_python_file(item):
            module_name = item.stem
            _import_module(f".{module_name}", package)
        elif item.is_dir() and not _is_excluded_package(item.name):
            subpackage = f"{package}.{item.name}"
            _import_module(subpackage, package)

def _load_config() -> Dict[str, Any]:
    """Load configuration from all sources."""
    config_loader = ConfigLoader()
    return config_loader.load()

# Current directory path
_current_dir = Path(__file__).parent

# Import all modules
_import_all_modules_from_directory(_current_dir, __name__)

# Load configuration
CONFIG = _load_config()

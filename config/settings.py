"""System settings configuration."""

import os
import logging
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from .config_loader import ConfigLoader

# Load environment variables
load_dotenv()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Determine environment
APP_ENV: str = os.getenv("APP_ENV", "development").lower()
logging.info("Loading settings for environment: %s", APP_ENV)

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Default settings for different environments
default_settings: Dict[str, Any] = {
    "development": {
        "DEBUG": True,
        "DATABASE": {
            "host": "localhost",
            "port": 5432,
            "user": "devuser",
            "password": "devpassword",
            "name": "dev_db",
        },
        "API": {
            "key": os.getenv("BINANCE_API_KEY", "binance_public_key"),
            "secret": os.getenv("BINANCE_API_SECRET", "binance_public_secret"),
            "base_url": "https://testnet.binance.vision",
        },
        "BYBIT": {
            "api_key": os.getenv("BYBIT_API_KEY", "bybit_dummy_key"),
            "api_secret": os.getenv("BYBIT_API_SECRET", "bybit_dummy_secret"),
            "use_testnet": True,
        },
        "TRADING": {
            "commission": 0.001,
            "spread": 0.0005,
            "slippage": 0.0005
        },
        "SECURITY": {
            "access_restrictions": False,
            "password_encryption": False
        },
        "LOGGING": {
            "level": "DEBUG",
            "file": "dev_app.log"
        },
        "PATHS": {
            "logs_dir": "./logs",
            "data_dir": "./data"
        }
    },
    "test": {
        "DEBUG": True,
        "DATABASE": {
            "host": "localhost",
            "port": 5432,
            "user": "testuser",
            "password": "testpassword",
            "name": "test_db",
        }
    },
    "production": {
        "DEBUG": False,
        "DATABASE": {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "user": os.getenv("DB_USER", "prod_user"),
            "password": os.getenv("DB_PASSWORD", ""),
            "name": os.getenv("DB_NAME", "prod_db"),
        }
    }
}

# Get settings for current environment
env_settings = default_settings.get(APP_ENV, default_settings["development"])

# Config files based on environment
config_files = [
    os.path.join(CONFIG_DIR, "config.yaml"),
    os.path.join(CONFIG_DIR, "settings.yml"),
]

if APP_ENV != "production":
    local_config = os.path.join(CONFIG_DIR, f"config.{APP_ENV}.yaml")
    if os.path.exists(local_config):
        config_files.append(local_config)

# Initialize config loader and load configuration
loader = ConfigLoader(config_files=config_files)
file_settings = loader.load()

def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

# Merge all settings
CONFIG = merge_dicts(env_settings, file_settings)

# Export all settings to global namespace
globals().update(CONFIG)

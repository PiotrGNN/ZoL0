"""
Unified configuration management system for the trading platform.
Implements a thread-safe singleton pattern with validation and hot-reloading support.
"""

import os
import json
import yaml
import logging
import threading
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from functools import lru_cache
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ConfigValidationError(Exception):
    """Configuration validation error."""

    def __init__(self, path: str, message: str, value: Any):
        self.path = path
        self.value = value
        super().__init__(f"{path}: {message}")


class ConfigFileWatcher(FileSystemEventHandler):
    """Watch for configuration file changes."""

    def __init__(self, manager):
        self.manager = manager

    def on_modified(self, event):
        if not event.is_directory and event.src_path in self.manager.watched_files:
            self.manager.reload_config()


class ConfigManager:
    """Thread-safe configuration management with validation and hot-reloading."""

    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if ConfigManager._initialized:
            return

        self.config_dir = os.getenv("CONFIG_DIR", "config")
        self.config: Dict[str, Any] = {}
        self.schema: Dict[str, Any] = {}
        self.watched_files: List[str] = []
        self.validation_errors: List[ConfigValidationError] = []
        self._observers: List[Observer] = []

        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)

        # Initialize configuration
        self._load_all()
        self._setup_file_watching()

        ConfigManager._initialized = True

    def _load_all(self) -> None:
        """Load configuration from all sources."""
        with self._lock:
            # Reset configuration
            self.config = {}
            self.validation_errors = []

            # Load base config
            base_config = Path(self.config_dir) / "config.yaml"
            if base_config.exists():
                self._load_yaml_file(base_config)

            # Load environment-specific config
            env = os.getenv("APP_ENV", "development")
            env_config = Path(self.config_dir) / f"config.{env}.yaml"
            if env_config.exists():
                self._load_yaml_file(env_config)

            # Load schema
            schema_file = Path(self.config_dir) / "schema.yaml"
            if schema_file.exists():
                self._load_schema_file(schema_file)

            # Load environment variables
            self._load_env_variables()

            # Validate configuration
            self._validate_config()

    def _load_yaml_file(self, file_path: Path) -> None:
        """Load YAML configuration file."""
        try:
            with open(file_path) as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    self._deep_update(self.config, yaml_config)
                    if str(file_path) not in self.watched_files:
                        self.watched_files.append(str(file_path))
        except Exception as e:
            logging.error(f"Error loading config file {file_path}: {e}")

    def _load_schema_file(self, file_path: Path) -> None:
        """Load schema file."""
        try:
            with open(file_path) as f:
                self.schema = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading schema file {file_path}: {e}")

    def _load_env_variables(self) -> None:
        """Load configuration from environment variables."""
        prefix = "TRADING_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                self._set_env_value(config_key, value)

    def _set_env_value(self, key: str, value: str) -> None:
        """Set configuration value from environment variable."""
        try:
            # Convert value to appropriate type
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)

            # Navigate to correct position
            config = self.config
            parts = key.split("_")
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        except Exception as e:
            logging.error(f"Error setting env value {key}: {e}")

    def _validate_config(self) -> bool:
        """Validate configuration against schema."""
        self.validation_errors = []
        if not self.schema:
            return True
        return self._validate_dict(self.config, self.schema.get("required", {}))

    def _validate_dict(self, config: Dict, schema: Dict, path: str = "") -> bool:
        """Recursively validate configuration dictionary against schema."""
        valid = True
        for key, value_schema in schema.items():
            new_path = f"{path}.{key}" if path else key

            # Check required fields
            if key not in config:
                self.validation_errors.append(
                    ConfigValidationError(new_path, "Required field missing", None)
                )
                valid = False
                continue

            # Validate value
            if not self._validate_value(config[key], value_schema, new_path):
                valid = False

        return valid

    def _validate_value(self, value: Any, schema: Dict, path: str) -> bool:
        """Validate a single value against its schema."""
        try:
            if schema["type"] == "dict":
                if not isinstance(value, dict):
                    self.validation_errors.append(
                        ConfigValidationError(path, "Must be a dictionary", value)
                    )
                    return False
                return self._validate_dict(value, schema.get("required", {}), path)

            if schema["type"] == "list":
                if not isinstance(value, list):
                    self.validation_errors.append(
                        ConfigValidationError(path, "Must be a list", value)
                    )
                    return False
                if "schema" in schema:
                    return all(
                        self._validate_value(item, schema["schema"], f"{path}[{i}]")
                        for i, item in enumerate(value)
                    )
                return True

            if schema["type"] == "int":
                if not isinstance(value, int):
                    self.validation_errors.append(
                        ConfigValidationError(path, "Must be an integer", value)
                    )
                    return False
                return self._validate_number(value, schema, path)

            if schema["type"] == "float":
                if not isinstance(value, (int, float)):
                    self.validation_errors.append(
                        ConfigValidationError(path, "Must be a number", value)
                    )
                    return False
                return self._validate_number(value, schema, path)

            if schema["type"] == "bool":
                if not isinstance(value, bool):
                    self.validation_errors.append(
                        ConfigValidationError(path, "Must be a boolean", value)
                    )
                    return False
                return True

            if schema["type"] == "str":
                if not isinstance(value, str):
                    self.validation_errors.append(
                        ConfigValidationError(path, "Must be a string", value)
                    )
                    return False
                if "allowed" in schema and value not in schema["allowed"]:
                    self.validation_errors.append(
                        ConfigValidationError(
                            path,
                            f"Must be one of: {', '.join(schema['allowed'])}",
                            value,
                        )
                    )
                    return False
                return True

        except Exception as e:
            self.validation_errors.append(
                ConfigValidationError(path, f"Validation error: {str(e)}", value)
            )
            return False

        return True

    def _validate_number(
        self, value: Union[int, float], schema: Dict, path: str
    ) -> bool:
        """Validate numeric value against min/max constraints."""
        if "min" in schema and value < schema["min"]:
            self.validation_errors.append(
                ConfigValidationError(
                    path, f"Must be greater than or equal to {schema['min']}", value
                )
            )
            return False
        if "max" in schema and value > schema["max"]:
            self.validation_errors.append(
                ConfigValidationError(
                    path, f"Must be less than or equal to {schema['max']}", value
                )
            )
            return False
        return True

    def _setup_file_watching(self) -> None:
        """Setup configuration file watching."""
        try:
            observer = Observer()
            observer.schedule(ConfigFileWatcher(self), self.config_dir, recursive=False)
            observer.start()
            self._observers.append(observer)
        except Exception as e:
            logging.error(f"Error setting up config file watching: {e}")

    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Recursively update nested dictionaries."""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path."""
        try:
            value = self.config
            for part in key.split("."):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any, persist: bool = True) -> bool:
        """Set configuration value."""
        with self._lock:
            try:
                # Navigate to correct position
                config = self.config
                parts = key.split(".")
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]

                # Set value
                old_value = config.get(parts[-1])
                config[parts[-1]] = value

                # Validate change
                if not self._validate_config():
                    # Revert change if validation fails
                    if old_value is not None:
                        config[parts[-1]] = old_value
                    else:
                        del config[parts[-1]]
                    return False

                # Clear cache since config changed
                self.get.cache_clear()

                # Persist if requested
                if persist:
                    self._save_config()

                return True
            except Exception as e:
                logging.error(f"Error setting config value {key}: {e}")
                return False

    def _save_config(self) -> bool:
        """Save configuration to files."""
        try:
            for filepath in self.watched_files:
                with open(filepath, "w") as f:
                    if filepath.endswith(".yaml") or filepath.endswith(".yml"):
                        yaml.dump(self.config, f, default_flow_style=False)
                    elif filepath.endswith(".json"):
                        json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving config: {e}")
            return False

    def reload_config(self) -> None:
        """Reload configuration from all sources."""
        self._load_all()

    def get_validation_errors(self) -> List[ConfigValidationError]:
        """Get list of validation errors."""
        return self.validation_errors

    def __del__(self):
        """Cleanup observers on deletion."""
        for observer in self._observers:
            observer.stop()
        for observer in self._observers:
            observer.join()


# Global instance
_config_manager = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

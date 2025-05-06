"""Configuration management system."""

import os
import json
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
import threading
from functools import lru_cache
from data.logging.system_logger import get_logger

logger = get_logger()

class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, 
                 config_dir: str = "config",
                 env_prefix: str = "APP_"):
        self.config_dir = config_dir
        self.env_prefix = env_prefix
        self.config: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._watchers: Dict[str, list] = {}
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Load initial configuration
        self.reload_config()
    
    def load_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.log_warning(
                f"Configuration file not found: {file_path}",
                warning_type="ConfigFileNotFound"
            )
            return {}
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix == '.json':
                    return json.load(f)
                elif file_path.suffix in ['.yml', '.yaml']:
                    return yaml.safe_load(f)
                else:
                    logger.log_warning(
                        f"Unsupported config file format: {file_path}",
                        warning_type="UnsupportedConfigFormat"
                    )
                    return {}
        except Exception as e:
            logger.log_error(
                f"Error loading config file {file_path}: {e}",
                error_type="ConfigLoadError"
            )
            return {}
    
    def save_file(self, config: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'w') as f:
                if file_path.suffix == '.json':
                    json.dump(config, f, indent=2)
                elif file_path.suffix in ['.yml', '.yaml']:
                    yaml.dump(config, f)
                else:
                    logger.log_warning(
                        f"Unsupported config file format: {file_path}",
                        warning_type="UnsupportedConfigFormat"
                    )
        except Exception as e:
            logger.log_error(
                f"Error saving config file {file_path}: {e}",
                error_type="ConfigSaveError"
            )
    
    def reload_config(self) -> None:
        """Reload configuration from all sources."""
        with self._lock:
            # Start with empty config
            self.config = {}
            
            # Load from config files
            for config_file in Path(self.config_dir).glob("*.{json,yml,yaml}"):
                file_config = self.load_file(config_file)
                self._deep_update(self.config, file_config)
            
            # Load from environment variables
            env_config = {}
            for key, value in os.environ.items():
                if key.startswith(self.env_prefix):
                    config_key = key[len(self.env_prefix):].lower()
                    self._set_nested_value(env_config, config_key.split('_'), value)
            
            # Environment variables override file config
            self._deep_update(self.config, env_config)
            
            # Notify watchers
            self._notify_watchers()
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target:
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _set_nested_value(self, config: Dict, keys: list, value: Any) -> None:
        """Set value in nested dictionary using list of keys."""
        current = config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        
        # Try to parse value as JSON, fall back to string if failed
        try:
            current[keys[-1]] = json.loads(value)
        except json.JSONDecodeError:
            current[keys[-1]] = value
    
    @lru_cache(maxsize=128)
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        try:
            value = self.config
            for part in key.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """Set configuration value."""
        with self._lock:
            keys = key.split('.')
            current = self.config
            
            # Navigate to the correct nested level
            for part in keys[:-1]:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[keys[-1]] = value
            
            # Clear the get() cache since config changed
            self.get.cache_clear()
            
            # Notify watchers
            self._notify_watchers(key)
            
            # Persist if requested
            if persist:
                self.save_file(self.config, Path(self.config_dir) / "config.json")
    
    def register_watcher(self, key: str, callback: callable) -> None:
        """Register a callback for configuration changes."""
        with self._lock:
            if key not in self._watchers:
                self._watchers[key] = []
            self._watchers[key].append(callback)
    
    def _notify_watchers(self, changed_key: Optional[str] = None) -> None:
        """Notify watchers of configuration changes."""
        for key, watchers in self._watchers.items():
            if changed_key is None or key.startswith(changed_key):
                value = self.get(key)
                for watcher in watchers:
                    try:
                        watcher(value)
                    except Exception as e:
                        logger.log_error(
                            f"Error in config watcher: {e}",
                            error_type="ConfigWatcherError"
                        )

# Global config manager instance
_config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """Get global config manager instance."""
    return _config_manager
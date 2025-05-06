"""Configuration loading module."""

import os
import json
import yaml
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

class ConfigLoader:
    def __init__(self, config_files: Optional[List[str]] = None):
        self.config_files = config_files or ["config/config.yaml", "config/settings.yml"]
        self.config: Dict[str, Any] = {}
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            self.logger.addHandler(handler)

    def load(self) -> Dict[str, Any]:
        """Load configuration from all sources."""
        config = {}
        
        # Load from YAML files
        for file_path in self.config_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    config.update(yaml.safe_load(f) or {})
                    self.logger.info(f"Loaded configuration from {file_path}")

        # Override with environment variables
        self._update_from_env(config)
        self.config = config
        
        return config

    def _update_from_env(self, config: Dict[str, Any]) -> None:
        """Update config with environment variables."""
        for key, value in os.environ.items():
            if key.startswith('APP_'):
                config_key = key[4:].lower()  # Remove APP_ prefix
                self._set_nested_dict(config, config_key.split('_'), value)

    def _set_nested_dict(self, d: Dict, keys: List[str], value: Any) -> None:
        """Set value in nested dictionary."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        try:
            d[keys[-1]] = json.loads(value)  # Try to parse JSON
        except (json.JSONDecodeError, TypeError):
            d[keys[-1]] = value  # Fall back to string if JSON parsing fails

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            value = self.config
            for part in key.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        current = self.config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    def save(self, file_path: Optional[str] = None) -> None:
        """
        Save the current configuration to file.
        
        Args:
            file_path (str, optional): Path to save the config to. If None, uses the first config file.
        """
        save_path = file_path or (self.config_files[0] if self.config_files else 'config/settings.json')
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                if save_path.endswith('.json'):
                    json.dump(self.config, f, indent=4)
                elif save_path.endswith(('.yaml', '.yml')):
                    yaml.safe_dump(self.config, f)
                else:
                    json.dump(self.config, f, indent=4)  # Default to JSON
            self.logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset the configuration to empty state."""
        self.config = {}
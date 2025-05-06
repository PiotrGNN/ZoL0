"""
Enhanced configuration management system with better validation and handling.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import system logger
from data.logging.system_logger import get_logger
logger = get_logger()

@dataclass
class ConfigValidationError:
    """Configuration validation error."""
    path: str
    message: str
    value: Any

class ConfigWatcher(FileSystemEventHandler):
    """Watch for configuration file changes."""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.last_reload = datetime.now()
        self.min_reload_interval = 1  # seconds

    def on_modified(self, event):
        """Handle file modification event."""
        if not event.is_directory and event.src_path in self.config_manager.watched_files:
            now = datetime.now()
            if (now - self.last_reload).total_seconds() >= self.min_reload_interval:
                self.config_manager.reload_config()
                self.last_reload = now

class ConfigManager:
    """Enhanced configuration management system."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.config: Dict[str, Any] = {}
        self.schema: Dict[str, Any] = {}
        self.watched_files: List[str] = []
        self._lock = threading.Lock()
        self._validation_errors: List[ConfigValidationError] = []
        self._observers: List[Observer] = []
        self._setup_file_watching()

    def _setup_file_watching(self):
        """Setup configuration file watching."""
        try:
            observer = Observer()
            observer.schedule(ConfigWatcher(self), self.config_dir, recursive=False)
            observer.start()
            self._observers.append(observer)
        except Exception as e:
            logger.log_error(f"Error setting up config file watching: {e}")

    def load_config(self, filename: str) -> bool:
        """Load configuration from file."""
        try:
            filepath = os.path.join(self.config_dir, filename)
            
            with self._lock:
                with open(filepath, 'r') as f:
                    if filename.endswith('.yaml') or filename.endswith('.yml'):
                        config = yaml.safe_load(f)
                    elif filename.endswith('.json'):
                        config = json.load(f)
                    else:
                        raise ValueError(f"Unsupported file format: {filename}")
                
                # Add to watched files
                if filepath not in self.watched_files:
                    self.watched_files.append(filepath)
                
                # Update config
                self.config.update(config)
                
                # Validate after loading
                self._validate_config()
                
                logger.log_info(f"Loaded configuration from {filename}")
                return True
        except Exception as e:
            logger.log_error(f"Error loading config {filename}: {e}")
            return False

    def load_schema(self, filename: str) -> bool:
        """Load configuration schema from file."""
        try:
            filepath = os.path.join(self.config_dir, filename)
            
            with self._lock:
                with open(filepath, 'r') as f:
                    if filename.endswith('.yaml') or filename.endswith('.yml'):
                        schema = yaml.safe_load(f)
                    elif filename.endswith('.json'):
                        schema = json.load(f)
                    else:
                        raise ValueError(f"Unsupported file format: {filename}")
                
                self.schema = schema
                logger.log_info(f"Loaded schema from {filename}")
                return True
        except Exception as e:
            logger.log_error(f"Error loading schema {filename}: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            value = self.config
            for part in key.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any, persist: bool = True) -> bool:
        """Set configuration value."""
        try:
            with self._lock:
                # Navigate to the correct position
                config = self.config
                parts = key.split('.')
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]
                
                # Set the value
                config[parts[-1]] = value
                
                # Validate the change
                if not self._validate_config():
                    # Revert the change if validation fails
                    config.pop(parts[-1], None)
                    return False
                
                # Persist if requested
                if persist:
                    self._save_config()
                
                return True
        except Exception as e:
            logger.log_error(f"Error setting config value {key}: {e}")
            return False

    def _validate_config(self) -> bool:
        """Validate configuration against schema."""
        self._validation_errors = []
        
        if not self.schema:
            return True
        
        def validate_object(config: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> bool:
            valid = True
            
            # Check required fields
            for key, value in schema.get('required', {}).items():
                full_path = f"{path}.{key}" if path else key
                
                if key not in config:
                    self._validation_errors.append(
                        ConfigValidationError(full_path, "Required field missing", None)
                    )
                    valid = False
                    continue
                
                if not self._validate_value(config[key], value, full_path):
                    valid = False
            
            # Check optional fields
            for key, value in schema.get('optional', {}).items():
                full_path = f"{path}.{key}" if path else key
                
                if key in config and not self._validate_value(config[key], value, full_path):
                    valid = False
            
            return valid
        
        return validate_object(self.config, self.schema)

    def _validate_value(self, value: Any, schema: Dict[str, Any], path: str) -> bool:
        """Validate single value against schema."""
        try:
            # Check type
            expected_type = schema.get('type')
            if expected_type:
                actual_type = type(value).__name__
                if expected_type != actual_type:
                    self._validation_errors.append(
                        ConfigValidationError(
                            path,
                            f"Type mismatch. Expected {expected_type}, got {actual_type}",
                            value
                        )
                    )
                    return False
            
            # Check range
            if 'min' in schema and value < schema['min']:
                self._validation_errors.append(
                    ConfigValidationError(
                        path,
                        f"Value {value} is less than minimum {schema['min']}",
                        value
                    )
                )
                return False
            
            if 'max' in schema and value > schema['max']:
                self._validation_errors.append(
                    ConfigValidationError(
                        path,
                        f"Value {value} is greater than maximum {schema['max']}",
                        value
                    )
                )
                return False
            
            # Check pattern
            if 'pattern' in schema and isinstance(value, str):
                import re
                if not re.match(schema['pattern'], value):
                    self._validation_errors.append(
                        ConfigValidationError(
                            path,
                            f"Value {value} does not match pattern {schema['pattern']}",
                            value
                        )
                    )
                    return False
            
            # Check enum
            if 'enum' in schema and value not in schema['enum']:
                self._validation_errors.append(
                    ConfigValidationError(
                        path,
                        f"Value {value} not in allowed values: {schema['enum']}",
                        value
                    )
                )
                return False
            
            return True
        except Exception as e:
            self._validation_errors.append(
                ConfigValidationError(path, f"Validation error: {str(e)}", value)
            )
            return False

    def _save_config(self) -> bool:
        """Save configuration to files."""
        try:
            for filepath in self.watched_files:
                with open(filepath, 'w') as f:
                    if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                        yaml.dump(self.config, f)
                    elif filepath.endswith('.json'):
                        json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logger.log_error(f"Error saving config: {e}")
            return False

    def reload_config(self) -> bool:
        """Reload configuration from all watched files."""
        success = True
        with self._lock:
            # Store current config as backup
            backup = self.config.copy()
            
            try:
                self.config = {}
                for filepath in self.watched_files:
                    filename = os.path.basename(filepath)
                    if not self.load_config(filename):
                        success = False
                        break
                
                if not success:
                    # Restore backup if any file failed to load
                    self.config = backup
            except Exception as e:
                logger.log_error(f"Error reloading config: {e}")
                self.config = backup
                success = False
        
        return success

    def get_validation_errors(self) -> List[ConfigValidationError]:
        """Get list of validation errors."""
        return self._validation_errors.copy()

    def __del__(self):
        """Cleanup observers on deletion."""
        for observer in self._observers:
            observer.stop()
        for observer in self._observers:
            observer.join()

# Create global config manager instance
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """Get global config manager instance."""
    return config_manager

# Example usage
if __name__ == "__main__":
    # Create example configuration
    os.makedirs("config", exist_ok=True)
    
    # Write example config
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'max_connections': 100
        },
        'api': {
            'port': 8080,
            'rate_limit': 1000
        }
    }
    
    with open('config/config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Write example schema
    schema = {
        'required': {
            'database': {
                'type': 'dict',
                'required': {
                    'host': {'type': 'str'},
                    'port': {'type': 'int', 'min': 1, 'max': 65535},
                    'max_connections': {'type': 'int', 'min': 1}
                }
            },
            'api': {
                'type': 'dict',
                'required': {
                    'port': {'type': 'int', 'min': 1, 'max': 65535},
                    'rate_limit': {'type': 'int', 'min': 1}
                }
            }
        }
    }
    
    with open('config/schema.yaml', 'w') as f:
        yaml.dump(schema, f)
    
    # Test configuration management
    manager = get_config_manager()
    
    print("\nLoading configuration...")
    manager.load_schema('schema.yaml')
    manager.load_config('config.yaml')
    
    print("\nConfiguration values:")
    print(f"Database host: {manager.get('database.host')}")
    print(f"API port: {manager.get('api.port')}")
    
    print("\nSetting invalid value...")
    success = manager.set('database.port', 70000)  # Invalid port number
    if not success:
        print("Validation errors:")
        for error in manager.get_validation_errors():
            print(f"- {error.path}: {error.message}")
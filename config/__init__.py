"""Configuration management system for the trading platform."""

from .manager import get_config, ConfigManager, ConfigValidationError
from .logging import get_logger
from .verify import verify_all as verify, check_configuration

# Initialize the global configuration instance
config = get_config()

__all__ = [
    "config",
    "get_config",
    "ConfigManager",
    "ConfigValidationError",
    "get_logger",
    "verify",
    "check_configuration",
]

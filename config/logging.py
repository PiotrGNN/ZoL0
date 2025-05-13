"""Logging configuration for the trading platform."""

import os
import sys
import yaml
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(config_path: Optional[str] = None) -> None:
    """Initialize logging configuration.

    Args:
        config_path: Path to logging configuration file
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "logging.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                config = yaml.safe_load(f)

                # Ensure log directories exist
                for handler in config.get("handlers", {}).values():
                    if "filename" in handler:
                        log_dir = os.path.dirname(handler["filename"])
                        os.makedirs(log_dir, exist_ok=True)

                logging.config.dictConfig(config)
                logging.info("Logging system initialized")

            except Exception as e:
                print(f"Error loading logging configuration: {e}", file=sys.stderr)
                sys.exit(1)
    else:
        print(f"Logging configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (defaults to root logger)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ContextLogger:
    """Logger with context tracking."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context = {}

    def set_context(self, **kwargs) -> None:
        """Set context values."""
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context values."""
        self.context.clear()

    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self.context:
            context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} [{context_str}]"
        return message

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        ctx = {**self.context, **kwargs}
        self.logger.debug(self._format_message(message), extra=ctx)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        ctx = {**self.context, **kwargs}
        self.logger.info(self._format_message(message), extra=ctx)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        ctx = {**self.context, **kwargs}
        self.logger.warning(self._format_message(message), extra=ctx)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        ctx = {**self.context, **kwargs}
        self.logger.error(self._format_message(message), extra=ctx)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        ctx = {**self.context, **kwargs}
        self.logger.critical(self._format_message(message), extra=ctx)


def get_context_logger(name: Optional[str] = None) -> ContextLogger:
    """Get a context-aware logger instance.

    Args:
        name: Logger name (defaults to root logger)

    Returns:
        ContextLogger instance
    """
    return ContextLogger(get_logger(name))

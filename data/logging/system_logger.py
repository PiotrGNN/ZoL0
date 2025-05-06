"""System logging module."""

import logging
import os
from typing import Optional
from logging.handlers import RotatingFileHandler

class SystemLogger:
    """System-wide logging functionality."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure main logger
        self.logger = logging.getLogger('system')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Prevent duplicate logs
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'system.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log_info(self, message: str, **kwargs):
        """Log info level message."""
        self.logger.info(message, extra=kwargs)

    def log_warning(self, message: str, warning_type: Optional[str] = None, **kwargs):
        """Log warning level message."""
        if warning_type:
            message = f"[{warning_type}] {message}"
        self.logger.warning(message, extra=kwargs)

    def log_error(self, message: str, error_type: Optional[str] = None, **kwargs):
        """Log error level message."""
        if error_type:
            message = f"[{error_type}] {message}"
        self.logger.error(message, extra=kwargs)

    def log_critical(self, message: str, error_type: Optional[str] = None, **kwargs):
        """Log critical level message."""
        if error_type:
            message = f"[{error_type}] {message}"
        self.logger.critical(message, extra=kwargs)

# Global logger instance
_system_logger = SystemLogger()

def get_logger() -> SystemLogger:
    """Get the global system logger instance."""
    return _system_logger
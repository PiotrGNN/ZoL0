"""
Enhanced error handling system with better classification and handling.
"""

import logging
import os
import sys
import traceback
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class BaseError(Exception):
    """Base error class for the system."""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code or 'UNKNOWN'
        self.context = context or {}
        self.timestamp = datetime.now()

class DataError(BaseError):
    """Data related errors."""
    pass

class APIError(BaseError):
    """API related errors."""
    pass

class TradingError(BaseError):
    """Trading related errors."""
    pass

class ModelError(BaseError):
    """AI model related errors."""
    pass

class ConfigError(BaseError):
    """Configuration related errors."""
    pass

class SecurityError(BaseError):
    """Security related errors."""
    pass

class ErrorHandler:
    """Centralized error handling system."""

    def __init__(self, log_dir: str = "logs", notify_critical: bool = True):
        self.log_dir = log_dir
        self.notify_critical = notify_critical
        self.error_counts: Dict[str, int] = {}
        self.error_thresholds: Dict[str, int] = {
            'API': 5,
            'DATA': 3,
            'TRADING': 3,
            'MODEL': 5,
            'CONFIG': 2,
            'SECURITY': 1
        }
        self._setup_logging()
        self._notification_callbacks: List[Callable] = []

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create separate log files for different error types
        handlers = {
            'api': logging.FileHandler(os.path.join(self.log_dir, 'api_errors.log')),
            'data': logging.FileHandler(os.path.join(self.log_dir, 'data_errors.log')),
            'trading': logging.FileHandler(os.path.join(self.log_dir, 'trading_errors.log')),
            'model': logging.FileHandler(os.path.join(self.log_dir, 'model_errors.log')),
            'security': logging.FileHandler(os.path.join(self.log_dir, 'security_errors.log'))
        }
        
        for handler in handlers.values():
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            logger.addHandler(handler)

    def add_notification_callback(self, callback: Callable) -> None:
        """Add notification callback for critical errors."""
        self._notification_callbacks.append(callback)

    def _notify(self, error: BaseError) -> None:
        """Send notifications for critical errors."""
        if not self.notify_critical:
            return

        message = f"Critical error: {error.error_code}\n{str(error)}\nContext: {error.context}"
        
        for callback in self._notification_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

    def handle_error(self, error: BaseError, raise_error: bool = True) -> Optional[Dict[str, Any]]:
        """
        Handle an error with proper logging and notification.
        
        Args:
            error: The error to handle
            raise_error: Whether to raise the error after handling

        Returns:
            Optional[Dict[str, Any]]: Error context if not raising
        """
        try:
            error_type = error.__class__.__name__
            error_category = error_type.replace('Error', '').upper()

            # Update error count
            self.error_counts[error_category] = self.error_counts.get(error_category, 0) + 1

            # Log the error
            log_message = f"{error_type}: {str(error)}"
            if error.context:
                log_message += f"\nContext: {error.context}"
            
            logger.error(log_message, exc_info=True)

            # Check if threshold exceeded
            if self.error_counts[error_category] >= self.error_thresholds.get(error_category, 5):
                self._notify(error)
                logger.critical(f"{error_category} error threshold exceeded!")

            error_info = {
                'error_code': error.error_code,
                'message': str(error),
                'type': error_type,
                'context': error.context,
                'timestamp': error.timestamp.isoformat(),
                'stack_trace': traceback.format_exc()
            }

            if raise_error:
                raise error
            return error_info

        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            if raise_error:
                raise e
            return {
                'error_code': 'ERROR_HANDLER_FAILED',
                'message': str(e),
                'type': 'ErrorHandlerError',
                'timestamp': datetime.now().isoformat()
            }

    def clear_error_counts(self) -> None:
        """Reset error counts."""
        self.error_counts.clear()

def handle_api_error(e: Exception) -> None:
    """Handle API-related errors."""
    error = APIError(
        message=str(e),
        error_code='API_ERROR',
        context={'api_name': 'bybit', 'timestamp': datetime.now().isoformat()}
    )
    error_handler.handle_error(error)

def handle_data_error(e: Exception) -> None:
    """Handle data-related errors."""
    error = DataError(
        message=str(e),
        error_code='DATA_ERROR',
        context={'data_type': 'market_data', 'timestamp': datetime.now().isoformat()}
    )
    error_handler.handle_error(error)

def handle_trading_error(e: Exception) -> None:
    """Handle trading-related errors."""
    error = TradingError(
        message=str(e),
        error_code='TRADING_ERROR',
        context={'trading_type': 'spot', 'timestamp': datetime.now().isoformat()}
    )
    error_handler.handle_error(error)

def handle_model_error(e: Exception) -> None:
    """Handle AI model-related errors."""
    error = ModelError(
        message=str(e),
        error_code='MODEL_ERROR',
        context={'model_type': 'sentiment', 'timestamp': datetime.now().isoformat()}
    )
    error_handler.handle_error(error)

def handle_config_error(e: Exception) -> None:
    """Handle configuration-related errors."""
    error = ConfigError(
        message=str(e),
        error_code='CONFIG_ERROR',
        context={'config_type': 'system', 'timestamp': datetime.now().isoformat()}
    )
    error_handler.handle_error(error)

def handle_security_error(e: Exception) -> None:
    """Handle security-related errors."""
    error = SecurityError(
        message=str(e),
        error_code='SECURITY_ERROR',
        context={'security_type': 'authentication', 'timestamp': datetime.now().isoformat()}
    )
    error_handler.handle_error(error)

def error_boundary(error_type: str = None):
    """
    Decorator to catch and handle errors.
    
    Args:
        error_type: Type of error to handle specifically
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_type == 'api':
                    handle_api_error(e)
                elif error_type == 'data':
                    handle_data_error(e)
                elif error_type == 'trading':
                    handle_trading_error(e)
                elif error_type == 'model':
                    handle_model_error(e)
                elif error_type == 'config':
                    handle_config_error(e)
                elif error_type == 'security':
                    handle_security_error(e)
                else:
                    # Default error handling
                    error = BaseError(
                        message=str(e),
                        error_code='UNKNOWN_ERROR',
                        context={'function': func.__name__, 'timestamp': datetime.now().isoformat()}
                    )
                    error_handler.handle_error(error)
                raise
        return wrapper
    return decorator

# Create global error handler instance
error_handler = ErrorHandler()

# Example usage
if __name__ == "__main__":
    # Add example notification callback
    def example_notification(message: str):
        print(f"NOTIFICATION: {message}")

    error_handler.add_notification_callback(example_notification)

    # Test error handling
    @error_boundary('api')
    def test_api_function():
        raise APIError("API test error", "TEST_API_ERROR")

    @error_boundary('data')
    def test_data_function():
        raise DataError("Data test error", "TEST_DATA_ERROR")

    try:
        test_api_function()
    except APIError:
        print("API error caught and handled")

    try:
        test_data_function()
    except DataError:
        print("Data error caught and handled")

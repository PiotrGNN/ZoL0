"""Error handling and recovery system."""

import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime

class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self, notify_critical: bool = True):
        self.notify_critical = notify_critical
        self.error_counts: Dict[str, int] = {}
        self._notification_callbacks = []
        
        # Configure basic logging
        self.logger = logging.getLogger('error_handler')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """Handle an error with proper logging and notification."""
        error_type = error.__class__.__name__
        
        # Update error count
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log the error
        log_message = f"{error_type}: {str(error)}"
        if context:
            log_message += f"\nContext: {context}"
        
        self.logger.error(log_message)
        
        # Send notifications for critical errors
        if self.notify_critical and context and context.get('is_critical', False):
            self._notify(error_type, str(error), context)
        
        return True

    def _notify(self, error_type: str, message: str, context: Dict[str, Any]) -> None:
        """Send notifications for critical errors."""
        for callback in self._notification_callbacks:
            try:
                callback({
                    'type': error_type,
                    'message': message,
                    'context': context,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                self.logger.error(f"Failed to send notification: {e}")

    def add_notification_callback(self, callback: Callable) -> None:
        """Add notification callback for critical errors."""
        self._notification_callbacks.append(callback)

    def clear_error_counts(self) -> None:
        """Reset error counts."""
        self.error_counts.clear()

# Global error handler instance
_error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _error_handler

def with_error_handling(func):
    """Decorator for automatic error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs),
                'timestamp': datetime.now().isoformat()
            }
            get_error_handler().handle_error(e, context)
            raise
    return wrapper
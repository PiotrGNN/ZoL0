"""
retry_handler.py - Moduł do obsługi ponownych prób połączeń
"""

import time
import logging
from functools import wraps
from typing import Callable, Dict, Any, Optional, List, Union

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 5.0,
        backoff_factor: float = 2.0,
        retryable_errors: Optional[List[Union[int, Exception]]] = None
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retryable_errors = retryable_errors or [502, 503, 504]

class RetryHandler:
    """Klasa obsługująca mechanizm ponownych prób"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 0
            delay = self.config.initial_delay
            last_exception = None
            
            while attempt < self.config.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
                    
                    # Sprawdź, czy błąd kwalifikuje się do ponownej próby
                    if error_code in self.config.retryable_errors or \
                       any(isinstance(e, err) for err in self.config.retryable_errors if isinstance(err, type)):
                        attempt += 1
                        
                        if attempt < self.config.max_retries:
                            wait_time = min(delay, self.config.max_delay)
                            logger.warning(
                                f"Próba {attempt}/{self.config.max_retries} nie powiodła się "
                                f"dla {func.__name__}. Następna próba za {wait_time}s. Błąd: {str(e)}"
                            )
                            time.sleep(wait_time)
                            delay *= self.config.backoff_factor
                            continue
                    
                    # Jeśli błąd nie kwalifikuje się do ponownej próby, rzuć go od razu
                    raise
            
            # Jeśli wszystkie próby się nie powiodły
            logger.error(
                f"Wszystkie {self.config.max_retries} prób nie powiodły się "
                f"dla {func.__name__}. Ostatni błąd: {str(last_exception)}"
            )
            raise last_exception

        return wrapper

# Domyślna konfiguracja dla całej aplikacji
default_retry_config = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=5.0,
    backoff_factor=2.0,
    retryable_errors=[502, 503, 504, TimeoutError, ConnectionError]
)

# Dekorator z domyślną konfiguracją
retry_with_backoff = RetryHandler(default_retry_config)

# Przykład użycia:
if __name__ == "__main__":
    @retry_with_backoff
    def example_function():
        # Symulacja błędu 502
        raise Exception("Error 502")
    
    try:
        example_function()
    except Exception as e:
        print(f"Final error: {e}")
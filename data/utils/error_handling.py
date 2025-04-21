"""
error_handling.py
---------------
Moduł obsługi błędów i wyjątków.
"""

import logging
from typing import Any, Callable, Dict, Optional

def handle_api_error(e: Exception, retry_count: int = 0) -> Dict[str, Any]:
    """Obsługa błędów API."""
    logging.error("API Error: %s (retry: %d)", str(e), retry_count)
    return {"status": "error", "message": str(e), "retry_count": retry_count}

def retry_on_error(max_retries: int = 3, delay: float = 1.0) -> Callable:
    """Dekorator do ponawiania nieudanych operacji."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(
                        "Attempt %d failed: %s. Retrying...",
                        attempt + 1,
                        str(e)
                    )
        return wrapper
    return decorator

def validate_input(value: Any, expected_type: type) -> Optional[str]:
    """Sprawdzanie poprawności typu danych wejściowych."""
    if not isinstance(value, expected_type):
        return f"Expected {expected_type.__name__}, got {type(value).__name__}"
    return None

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

class DataError(Exception):
    """Wyjątek związany z błędami przetwarzania danych."""
class TradingError(Exception):
    """Wyjątek związany z błędami podczas wykonywania transakcji."""
class ConfigError(Exception):
    """Wyjątek związany z błędami konfiguracji."""

def log_error(error, level=logging.ERROR):
    """Loguje błąd z podanym poziomem i ewentualnie wysyła powiadomienie."""
    logging.log(level, "Błąd: %s", str(error))

def handle_data_error(error):
    """Obsługuje błąd związany z danymi, logując go i rzucając DataError."""
    log_error(error, level=logging.ERROR)
    raise DataError(error)

def handle_trading_error(error):
    """Obsługuje błąd związany z transakcjami, logując go i rzucając TradingError."""
    log_error(error, level=logging.CRITICAL)
    raise TradingError(error)

def handle_config_error(error):
    """Obsługuje błąd związany z konfiguracją, logując go i rzucając ConfigError."""
    log_error(error, level=logging.WARNING)
    raise ConfigError(error)

# -------------------- Przykładowe testy jednostkowe --------------------
if __name__ == "__main__":
    try:
        try:
            raise ValueError("Test data error")
        except Exception as e:
            handle_data_error(e)
    except DataError as de:
        logging.info("Prawidłowo przechwycony DataError: %s", de)

    try:
        try:
            raise ValueError("Test trading error")
        except Exception as e:
            handle_trading_error(e)
    except TradingError as te:
        logging.info("Prawidłowo przechwycony TradingError: %s", te)

    try:
        try:
            raise ValueError("Test config error")
        except Exception as e:
            handle_config_error(e)
    except ConfigError as ce:
        logging.info("Prawidłowo przechwycony ConfigError: %s", ce)

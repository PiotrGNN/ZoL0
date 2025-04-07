"""
error_handling.py
-----------------
Moduł centralizujący obsługę błędów w projekcie.

Funkcjonalności:
- Definiuje niestandardowe klasy wyjątków dla różnych warstw aplikacji (np. DataError, TradingError, ConfigError).
- Zapewnia funkcje do jednolitego logowania błędów oraz powiadamiania (np. przez e-mail, Slack) w przypadku krytycznych problemów.
- Integruje się z modułami logowania i systemami monitoringu, aby automatycznie przechwytywać i klasyfikować błędy.
- Zawiera testy weryfikujące, czy błędy są poprawnie przechwytywane i zgłaszane.
"""

import logging

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
    """
    Loguje błąd z podanym poziomem i ewentualnie wysyła powiadomienie.

    Parameters:
        error (Exception): Błąd do zalogowania.
        level (int): Poziom logowania (np. logging.ERROR, logging.CRITICAL).
    """
    logging.log(level, "Błąd: %s", str(error))
    # Tutaj można zintegrować wysyłkę powiadomień np. e-mail, Slack, Sentry itp.


def handle_data_error(error):
    """
    Obsługuje błąd związany z danymi, logując go i rzucając DataError.

    Parameters:
        error (Exception): Błąd do obsłużenia.
    """
    log_error(error, level=logging.WARNING)
    raise DataError(error)


def handle_trading_error(error):
    """
    Obsługuje błąd związany z tradingiem, logując go i rzucając TradingError.

    Parameters:
        error (Exception): Błąd do obsłużenia.
    """
    log_error(error, level=logging.CRITICAL)
    raise TradingError(error)


def handle_config_error(error):
    """
    Obsługuje błąd związany z konfiguracją, logując go i rzucając ConfigError.

    Parameters:
        error (Exception): Błąd do obsłużenia.
    """
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
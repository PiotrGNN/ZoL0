"""
error_logger.py
---------------
Moduł do rejestrowania błędów i wyjątków.
Funkcjonalności:
- Wielopoziomowe logi: INFO, WARNING, ERROR, CRITICAL.
- Możliwość wysyłania powiadomień w przypadku krytycznych usterek.
- Maskowanie poufnych informacji w logach (np. kluczy API).
- Kompatybilność z frameworkiem logging.
"""

import logging
import os
import sys
import traceback
import re

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

# Konfiguracja podstawowa logowania
LOG_FILE = "error.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)


def mask_sensitive_data(message: str) -> str:
    """
    Maskuje poufne dane w komunikatach logów.

    Parameters:
        message (str): Komunikat do zamaskowania.

    Returns:
        str: Zamaskowany komunikat.
    """
    # Maskowanie kluczy API, sekretów, URL z tokenami
    patterns = [
        r'(api_key|secret|password|token)(\s*[=:]\s*["\'])([^"\']+)(["\'])',
        r'(https?://[^:]+:)([^@]+)(@)',
    ]

    masked_message = message
    for pattern in patterns:
        masked_message = re.sub(pattern, r'\1\2***\4', masked_message)

    return masked_message


class ErrorLogger:
    def __init__(self, notify_critical: bool = False, sentry_dsn: str = None):
        """
        Inicjalizuje moduł ErrorLogger.

        Parameters:
            notify_critical (bool): Jeśli True, wysyła powiadomienia przy krytycznych błędach.
            sentry_dsn (str): DSN do Sentry, jeśli integracja jest wymagana.
        """
        self.notify_critical = notify_critical
        if notify_critical and sentry_dsn and SENTRY_AVAILABLE:
            sentry_sdk.init(sentry_dsn)
            logging.info("Integracja z Sentry aktywowana.")
        else:
            if notify_critical:
                logging.warning(
                    "Powiadomienia krytyczne są włączone, ale Sentry nie jest skonfigurowane."
                )

    def log_info(self, message: str):
        logging.info(mask_sensitive_data(message))

    def log_warning(self, message: str):
        logging.warning(mask_sensitive_data(message))

    def log_error(self, message: str, exc_info: bool = True):
        logging.error(mask_sensitive_data(message), exc_info=exc_info)
        if self.notify_critical:
            self.send_notification("ERROR", message, exc_info)

    def log_critical(self, message: str, exc_info: bool = True):
        logging.critical(mask_sensitive_data(message), exc_info=exc_info)
        if self.notify_critical:
            self.send_notification("CRITICAL", message, exc_info)

    def send_notification(self, level: str, message: str, exc_info: bool):
        """
        Wysyła powiadomienie przy krytycznych błędach.
        Obecnie symulowane poprzez logowanie, można zintegrować z zewnętrznymi narzędziami.

        Parameters:
            level (str): Poziom błędu ("ERROR" lub "CRITICAL").
            message (str): Komunikat błędu.
            exc_info (bool): Informacja, czy dołączyć traceback.
        """
        notification_message = f"[{level}] {mask_sensitive_data(message)}"
        if exc_info:
            notification_message += "\n" + traceback.format_exc()
        # Tutaj można dodać integrację z systemami powiadomień (np. e-mail, Slack, Sentry)
        logging.info("Wysłano powiadomienie krytyczne: %s", notification_message)

# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    error_logger = ErrorLogger(notify_critical=True, sentry_dsn=os.getenv("SENTRY_DSN"))

    try:
        # Przykładowa operacja, która wywoła błąd
        result = 10 / 0
    except Exception as e:
        error_logger.log_error(f"Błąd przy wykonywaniu operacji: {str(e)}")

    # Przykładowe logowanie informacji
    error_logger.log_info("Moduł error_logger działa poprawnie.")

    # Przykładowe logowanie ostrzeżeń
    error_logger.log_warning("To jest ostrzeżenie testowe.")

    # Przykładowe logowanie krytyczne
    error_logger.log_critical("To jest krytyczny błąd testowy.", exc_info=True)
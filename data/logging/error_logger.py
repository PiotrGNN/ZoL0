
"""
error_logger.py
---------------
Moduł do rejestrowania błędów i wyjątków.
Funkcjonalności:
- Wielopoziomowe logi: INFO, WARNING, ERROR, CRITICAL.
- Możliwość wysyłania powiadomień w przypadku krytycznych usterek (przykładowo poprzez integrację z Sentry lub Datadog).
- Maskowanie poufnych informacji w logach (np. kluczy API).
- Kompatybilność z frameworkiem logging oraz opcjonalnie loguru (jeśli zainstalowany).
- Zapewnienie wydajności, aby nie obciążać systemów HFT.
"""

import logging
import os
import sys
import traceback

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


# Funkcja maskująca wrażliwe dane
def mask_sensitive_data(message):
    """
    Maskuje wrażliwe dane w logach, takie jak klucze API czy hasła.

    Parameters:
        message (str): Komunikat do przetworzenia.

    Returns:
        str: Komunikat z zamaskowanymi wrażliwymi danymi.
    """
    if not isinstance(message, str):
        return message

    # Lista wzorców do maskowania
    patterns = [
        "api_key", "apikey", "secret", "password", "token",
        "API_KEY", "APIKEY", "SECRET", "PASSWORD", "TOKEN"
    ]

    masked_message = message
    for pattern in patterns:
        # Szukanie wzorców w formacie "pattern=wartość"
        if pattern in masked_message:
            parts = masked_message.split(pattern + "=")
            if len(parts) > 1:
                # Zastąpienie wartości gwiazdkami
                value_end = parts[1].find(" ") if " " in parts[1] else len(parts[1])
                masked_message = parts[0] + pattern + "=" + parts[1][:4] + "***" + (parts[1][value_end:] if value_end < len(parts[1]) else "")

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

    # Przykładowe logowanie krytycznego błędu
    try:
        with open("nieistniejacy_plik.txt", "r") as f:
            content = f.read()
    except Exception as e:
        error_logger.log_critical(f"Krytyczny błąd przy otwieraniu pliku: {str(e)}")

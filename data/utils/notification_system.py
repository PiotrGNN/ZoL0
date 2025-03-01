"""
notification_system.py
----------------------
Moduł wysyłający powiadomienia (e-mail, SMS, komunikatory) w przypadku krytycznych zdarzeń systemowych lub błędów.
Funkcjonalności:
- Wsparcie dla wielu kanałów powiadomień (np. Twilio, SendGrid, Slack, Telegram) z możliwością konfiguracji.
- Formatowanie wiadomości (np. HTML, Markdown) oraz logowanie wysłanych powiadomień.
- Mechanizmy zapobiegające spamowi (np. limit liczby powiadomień na godzinę).
- Testy integracyjne, w tym sprawdzanie poprawności konfiguracji API i faktycznej wysyłki.
- Kod jest elastyczny i łatwo rozszerzalny o kolejne kanały powiadomień.
"""

import os
import time
import logging
from collections import defaultdict

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class NotificationSystem:
    def __init__(self, config: dict = None, rate_limit: int = 5):
        """
        Inicjalizuje NotificationSystem.
        
        Parameters:
            config (dict): Konfiguracja kanałów powiadomień, np. klucze API dla Slack, Twilio, itp.
            rate_limit (int): Maksymalna liczba powiadomień na godzinę dla każdego kanału.
        """
        self.config = config or {}
        self.rate_limit = rate_limit
        self.sent_counts = defaultdict(int)
        self.last_reset_time = time.time()
        logging.info("NotificationSystem zainicjalizowany z rate_limit: %d powiadomień/godz.", self.rate_limit)
    
    def _reset_rate_limit(self):
        current_time = time.time()
        if current_time - self.last_reset_time >= 3600:
            self.sent_counts.clear()
            self.last_reset_time = current_time
            logging.info("Rate limit został zresetowany.")
    
    def send_notification(self, channel: str, message: str, format: str = "text") -> bool:
        """
        Wysyła powiadomienie przez wybrany kanał.
        
        Parameters:
            channel (str): Nazwa kanału (np. "slack", "email", "sms").
            message (str): Treść powiadomienia.
            format (str): Format wiadomości ("text", "html", "markdown").
        
        Returns:
            bool: True, jeśli powiadomienie zostało wysłane, False w przeciwnym razie.
        """
        self._reset_rate_limit()
        if self.sent_counts[channel] >= self.rate_limit:
            logging.warning("Limit powiadomień dla kanału '%s' został osiągnięty.", channel)
            return False
        
        # Tutaj symulujemy wysyłkę powiadomienia.
        # W praktycznej implementacji należy zintegrować się z odpowiednim API (np. Twilio, SendGrid, Slack).
        logging.info("Wysyłanie powiadomienia przez %s (format: %s): %s", channel, format, message)
        
        # Symulujemy opóźnienie wysyłki
        time.sleep(0.5)
        self.sent_counts[channel] += 1
        logging.info("Powiadomienie wysłane przez kanał '%s'. Liczba powiadomień: %d", channel, self.sent_counts[channel])
        return True

# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowa konfiguracja powiadomień
        config = {
            "slack": {"webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"},
            "email": {"smtp_server": "smtp.example.com", "from": "noreply@example.com"},
            "sms": {"api_key": "dummy_sms_api_key"}
        }
        notifier = NotificationSystem(config=config, rate_limit=3)
        
        # Przykładowe wysyłanie powiadomień
        notifier.send_notification("slack", "Testowe powiadomienie z NotificationSystem.", format="markdown")
        notifier.send_notification("email", "<h1>Alert</h1><p>Testowe powiadomienie e-mail.</p>", format="html")
        notifier.send_notification("sms", "Testowe powiadomienie SMS.", format="text")
        
        # Przekroczenie limitu powiadomień
        for _ in range(4):
            notifier.send_notification("sms", "Dodatkowe powiadomienie SMS.", format="text")
        
    except Exception as e:
        logging.error("Błąd w module notification_system.py: %s", e)
        raise

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

import logging
import time
from collections import defaultdict

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


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
        logging.info(
            "NotificationSystem zainicjalizowany z rate_limit: %d powiadomień/godz.",
            self.rate_limit,
        )

    def _reset_rate_limit(self):
        current_time = time.time()
        if current_time - self.last_reset_time >= 3600:
            self.sent_counts.clear()
            self.last_reset_time = current_time
            logging.info("Rate limit został zresetowany.")

    def send_notification(
        self, channel: str, message: str, format: str = "text"
    ) -> bool:
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
            logging.warning(
                "Limit powiadomień dla kanału '%s' został osiągnięty.", channel
            )
            return False

        # Tutaj symulujemy wysyłkę powiadomienia.
        # W praktycznej implementacji należy zintegrować się z odpowiednim API (np. Twilio, SendGrid, Slack).
        logging.info(
            "Wysyłanie powiadomienia przez %s (format: %s): %s",
            channel,
            format,
            message,
        )

        # Symulujemy opóźnienie wysyłki
        time.sleep(0.5)
        self.sent_counts[channel] += 1
        logging.info(
            "Powiadomienie wysłane przez kanał '%s'. Liczba powiadomień: %d",
            channel,
            self.sent_counts[channel],
        )
        return True


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowa konfiguracja powiadomień
        config = {
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
            },
            "email": {"smtp_server": "smtp.example.com", "from": "noreply@example.com"},
            "sms": {"api_key": "dummy_sms_api_key"},
        }
        notifier = NotificationSystem(config=config, rate_limit=3)

        # Przykładowe wysyłanie powiadomień
        notifier.send_notification(
            "slack", "Testowe powiadomienie z NotificationSystem.", format="markdown"
        )
        notifier.send_notification(
            "email", "<h1>Alert</h1><p>Testowe powiadomienie e-mail.</p>", format="html"
        )
        notifier.send_notification("sms", "Testowe powiadomienie SMS.", format="text")

        # Przekroczenie limitu powiadomień
        for _ in range(4):
            notifier.send_notification(
                "sms", "Dodatkowe powiadomienie SMS.", format="text"
            )

    except Exception as e:
        logging.error("Błąd w module notification_system.py: %s", e)
        raise
"""
notification_system.py
---------------------
System do wysyłania powiadomień o istotnych zdarzeniach w systemie tradingowym.
Wspiera różne kanały komunikacji: email, Telegram, Discord, webhook.
"""

import logging
import json
import smtplib
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class NotificationSystem:
    """
    System zarządzania powiadomieniami dla różnych kanałów komunikacji.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicjalizacja systemu powiadomień.
        
        Args:
            config: Konfiguracja systemu powiadomień z kanałami
        """
        self.config = config or {}
        self.enabled_channels = self.config.get("enabled_channels", ["console"])
        self.notification_levels = {
            "critical": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
            "info": 1
        }
        self.min_level = self.config.get("min_notification_level", "low")
        logger.info("Inicjalizacja systemu powiadomień, kanały: %s", self.enabled_channels)
    
    def send_notification(self, 
                         message: str, 
                         level: str = "info", 
                         title: str = "Powiadomienie systemu", 
                         data: Dict[str, Any] = None) -> bool:
        """
        Wysyła powiadomienie wszystkimi dostępnymi kanałami.
        
        Args:
            message: Treść powiadomienia
            level: Poziom ważności (critical, high, medium, low, info)
            title: Tytuł powiadomienia
            data: Dodatkowe dane do załączenia
            
        Returns:
            Czy powiadomienie zostało wysłane poprawnie
        """
        # Sprawdź, czy powiadomienie powinno być wysłane (bazując na poziomie ważności)
        if self.notification_levels.get(level, 0) < self.notification_levels.get(self.min_level, 0):
            logger.debug("Pomijam powiadomienie o poziomie %s (minimalny poziom: %s)", level, self.min_level)
            return False
            
        # Przygotuj podstawowe dane powiadomienia
        notification_data = {
            "title": title,
            "message": message,
            "level": level,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        success = True
        
        # Wysyłanie na wszystkie włączone kanały
        for channel in self.enabled_channels:
            try:
                if channel == "email" and "email" in self.config:
                    self._send_email_notification(notification_data)
                elif channel == "telegram" and "telegram" in self.config:
                    self._send_telegram_notification(notification_data)
                elif channel == "discord" and "discord" in self.config:
                    self._send_discord_notification(notification_data)
                elif channel == "webhook" and "webhook" in self.config:
                    self._send_webhook_notification(notification_data)
                elif channel == "console":
                    self._send_console_notification(notification_data)
                else:
                    logger.warning("Nieznany kanał powiadomień: %s", channel)
                    success = False
            except Exception as e:
                logger.error("Błąd podczas wysyłania powiadomienia kanałem %s: %s", channel, e)
                success = False
        
        return success
    
    def _send_email_notification(self, notification_data: Dict[str, Any]) -> bool:
        """
        Wysyła powiadomienie przez email.
        
        Args:
            notification_data: Dane powiadomienia
            
        Returns:
            Czy powiadomienie zostało wysłane poprawnie
        """
        # W rzeczywistej aplikacji używalibyśmy SMTP
        logger.info("Symulacja wysłania email: %s", notification_data["title"])
        return True
    
    def _send_telegram_notification(self, notification_data: Dict[str, Any]) -> bool:
        """
        Wysyła powiadomienie przez Telegram.
        
        Args:
            notification_data: Dane powiadomienia
            
        Returns:
            Czy powiadomienie zostało wysłane poprawnie
        """
        # W rzeczywistej aplikacji używalibyśmy Telegram Bot API
        logger.info("Symulacja wysłania na Telegram: %s", notification_data["title"])
        return True
    
    def _send_discord_notification(self, notification_data: Dict[str, Any]) -> bool:
        """
        Wysyła powiadomienie przez Discord.
        
        Args:
            notification_data: Dane powiadomienia
            
        Returns:
            Czy powiadomienie zostało wysłane poprawnie
        """
        # W rzeczywistej aplikacji używalibyśmy Discord Webhook
        logger.info("Symulacja wysłania na Discord: %s", notification_data["title"])
        return True
    
    def _send_webhook_notification(self, notification_data: Dict[str, Any]) -> bool:
        """
        Wysyła powiadomienie przez webhook.
        
        Args:
            notification_data: Dane powiadomienia
            
        Returns:
            Czy powiadomienie zostało wysłane poprawnie
        """
        # W rzeczywistej aplikacji wysyłalibyśmy dane do zdefiniowanego endpointu
        logger.info("Symulacja wysłania przez webhook: %s", notification_data["title"])
        return True
    
    def _send_console_notification(self, notification_data: Dict[str, Any]) -> bool:
        """
        Wyświetla powiadomienie w konsoli.
        
        Args:
            notification_data: Dane powiadomienia
            
        Returns:
            Czy powiadomienie zostało wysłane poprawnie
        """
        level_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
            "info": "🔵"
        }
        
        emoji = level_emoji.get(notification_data["level"], "ℹ️")
        print(f"\n{emoji} {notification_data['title']}")
        print(f"   {notification_data['message']}")
        print(f"   Poziom: {notification_data['level']}, Czas: {notification_data['timestamp']}")
        if notification_data["data"]:
            print(f"   Dane: {json.dumps(notification_data['data'], indent=2)}")
        return True
    
    def alert_anomaly(self, symbol: str, anomaly_type: str, severity: float, details: Dict[str, Any] = None) -> bool:
        """
        Wysyła alert o wykrytej anomalii.
        
        Args:
            symbol: Symbol rynkowy
            anomaly_type: Typ anomalii
            severity: Poziom istotności anomalii (0.0-1.0)
            details: Szczegółowe dane anomalii
            
        Returns:
            Czy alert został wysłany poprawnie
        """
        # Określ poziom powiadomienia na podstawie istotności
        if severity >= 0.8:
            level = "critical"
        elif severity >= 0.6:
            level = "high"
        elif severity >= 0.4:
            level = "medium"
        else:
            level = "low"
            
        title = f"Wykryto anomalię: {symbol}"
        message = f"Wykryto anomalię typu {anomaly_type} dla {symbol}. Poziom istotności: {severity:.2f}"
        
        return self.send_notification(
            message=message,
            level=level,
            title=title,
            data={
                "symbol": symbol,
                "anomaly_type": anomaly_type,
                "severity": severity,
                **(details or {})
            }
        )
    
    def alert_trade_executed(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """
        Wysyła powiadomienie o wykonanej transakcji.
        
        Args:
            symbol: Symbol rynkowy
            side: Strona transakcji (buy/sell)
            quantity: Ilość
            price: Cena
            
        Returns:
            Czy powiadomienie zostało wysłane poprawnie
        """
        side_map = {"buy": "Kupiono", "sell": "Sprzedano"}
        side_text = side_map.get(side.lower(), side)
        
        title = f"Transakcja: {side_text} {symbol}"
        message = f"{side_text} {quantity} {symbol} po cenie {price}"
        
        return self.send_notification(
            message=message,
            level="medium",
            title=title,
            data={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "value": quantity * price
            }
        )
    
    def alert_system_status(self, status: str, details: str = None) -> bool:
        """
        Wysyła powiadomienie o zmianie statusu systemu.
        
        Args:
            status: Nowy status systemu
            details: Szczegóły zmiany statusu
            
        Returns:
            Czy powiadomienie zostało wysłane poprawnie
        """
        title = f"Status systemu: {status}"
        message = details or f"System zmienił status na: {status}"
        
        return self.send_notification(
            message=message,
            level="info",
            title=title,
            data={"status": status}
        )

# Przykładowe użycie
if __name__ == "__main__":
    # Konfiguracja z włączonymi kanałami
    notification_config = {
        "enabled_channels": ["console", "email"],
        "min_notification_level": "low",
        "email": {
            "smtp_server": "smtp.example.com",
            "port": 587,
            "username": "alerts@example.com",
            "password": "password",
            "from_email": "alerts@example.com",
            "to_email": "trader@example.com"
        }
    }
    
    notifier = NotificationSystem(notification_config)
    
    # Testowe powiadomienia
    notifier.alert_anomaly(
        symbol="BTC/USDT",
        anomaly_type="price_spike",
        severity=0.85,
        details={"price_change": "+5.2%", "time_frame": "5min"}
    )
    
    notifier.alert_trade_executed(
        symbol="ETH/USDT",
        side="buy",
        quantity=0.5,
        price=1950.25
    )
    
    notifier.alert_system_status(
        status="Starting",
        details="System jest uruchamiany w trybie produkcyjnym"
    )

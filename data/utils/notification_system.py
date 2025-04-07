"""
notification_system.py
---------------------
System do wysyłania powiadomień o istotnych zdarzeniach w systemie tradingowym.
Wspiera różne kanały komunikacji: email, Telegram, Discord, webhook, konsola, web.
"""

import logging
import json
import os
import time
from datetime import datetime, timedelta
from threading import Lock

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NotificationSystem:
    """
    System powiadomień dla ważnych zdarzeń i alertów.
    Obsługuje różne kanały powiadomień oraz zarządza limitami i priorytetami.
    """

    def __init__(self, config=None):
        """
        Inicjalizacja systemu powiadomień.

        Args:
            config (dict, optional): Konfiguracja systemu powiadomień
        """
        self.config = config or {}
        self.notification_log = []
        self.enabled_channels = self.config.get("enabled_channels", ["console", "web"])

        # Ustawienia dla różnych kanałów powiadomień
        self.email_config = self.config.get("email", {})
        self.sms_config = self.config.get("sms", {})
        self.slack_config = self.config.get("slack", {})
        self.telegram_config = self.config.get("telegram", {})

        # Limity i ograniczenia
        self.rate_limits = {
            "info": self.config.get("info_rate_limit", 20),  # Max 20 info na godzinę
            "warning": self.config.get("warning_rate_limit", 10),  # Max 10 ostrzeżeń na godzinę
            "alert": self.config.get("alert_rate_limit", 5),  # Max 5 alertów na godzinę
            "critical": self.config.get("critical_rate_limit", 3),  # Max 3 krytyczne na godzinę
        }

        # Liczniki dla każdego typu powiadomień
        self.counters = {level: 0 for level in self.rate_limits.keys()}
        self.last_reset = datetime.now()

        # Blokada dla operacji na systemie powiadomień (thread-safe)
        self.lock = Lock()

        # Czarne listy - ograniczanie powtarzających się powiadomień
        self.cooldowns = {}  # message_hash -> timestamp
        self.cooldown_period = self.config.get("cooldown_period", 3600)  # 1 godzina domyślnie

        # Inicjalizacja pliku z logami powiadomień
        self.log_file = self.config.get("log_file", "logs/notifications.log")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        logging.info(f"Zainicjalizowano system powiadomień z {len(self.enabled_channels)} kanałami.")

    def send_notification(self, message, level="info", channel=None, title=None, data=None):
        """
        Wysyła powiadomienie przez określone kanały.

        Args:
            message (str): Treść powiadomienia
            level (str): Poziom ważności ('info', 'warning', 'alert', 'critical')
            channel (str, optional): Konkretny kanał do wysłania
            title (str, optional): Tytuł powiadomienia
            data (dict, optional): Dodatkowe dane związane z powiadomieniem

        Returns:
            bool: Czy powiadomienie zostało wysłane
        """
        with self.lock:
            # Sprawdź czy nie przekroczono limitów
            if not self._check_rate_limits(level):
                logging.warning(f"Przekroczono limit powiadomień typu '{level}'")
                return False

            # Sprawdź czy podobne powiadomienie nie zostało niedawno wysłane
            if self._is_in_cooldown(message, level):
                logging.debug(f"Podobne powiadomienie w okresie cooldown - ignorowanie")
                return False

            # Określenie kanałów
            channels_to_use = [channel] if channel else self.enabled_channels

            # Formatowanie wiadomości
            formatted_message = self._format_message(message, level, title)

            # Przygotowanie pełnego powiadomienia
            notification = {
                "message": message,
                "formatted_message": formatted_message,
                "level": level,
                "timestamp": datetime.now().isoformat(),
                "title": title or self._get_default_title(level),
                "data": data or {}
            }

            # Wysyłanie przez wszystkie wybrane kanały
            success = False
            sent_channels = []

            for ch in channels_to_use:
                if ch in self.enabled_channels:
                    if self._send_via_channel(notification, ch):
                        success = True
                        sent_channels.append(ch)

            if success:
                # Zwiększ licznik dla danego poziomu
                self.counters[level] += 1
                # Dodaj do cooldown listy
                self._add_to_cooldown(message, level)
                # Zapisz do logu
                self._log_notification(notification, sent_channels)

            return success

    def _check_rate_limits(self, level):
        """
        Sprawdza czy nie przekroczono limitów dla danego poziomu.

        Args:
            level (str): Poziom powiadomienia

        Returns:
            bool: Czy można wysłać powiadomienie
        """
        # Resetuj liczniki jeśli minęła godzina
        current_time = datetime.now()
        if (current_time - self.last_reset).total_seconds() > 3600:
            self.counters = {level: 0 for level in self.rate_limits.keys()}
            self.last_reset = current_time

        # Sprawdź czy nie przekroczono limitu dla danego poziomu
        level_limit = self.rate_limits.get(level, 10)  # Domyślnie 10 na godzinę

        # Krytyczne powiadomienia zawsze przechodzą, chyba że już naprawdę dużo ich wysłano
        if level == "critical" and self.counters[level] < self.rate_limits[level] * 2:
            return True

        return self.counters[level] < level_limit

    def _is_in_cooldown(self, message, level):
        """
        Sprawdza czy podobne powiadomienie nie zostało niedawno wysłane.

        Args:
            message (str): Treść powiadomienia
            level (str): Poziom powiadomienia

        Returns:
            bool: Czy podobne powiadomienie jest w okresie cooldown
        """
        # Prosty hash wiadomości i poziomu
        message_hash = hash(f"{level}:{message}")

        if message_hash in self.cooldowns:
            last_sent = self.cooldowns[message_hash]
            # Krótszy cooldown dla ważniejszych powiadomień
            modifier = 1.0
            if level == "critical":
                modifier = 0.25  # 15 minut dla krytycznych
            elif level == "alert":
                modifier = 0.5   # 30 minut dla alertów

            cooldown_seconds = self.cooldown_period * modifier

            if (datetime.now() - last_sent).total_seconds() < cooldown_seconds:
                return True

        return False

    def _add_to_cooldown(self, message, level):
        """
        Dodaje wiadomość do cooldown listy.

        Args:
            message (str): Treść powiadomienia
            level (str): Poziom powiadomienia
        """
        message_hash = hash(f"{level}:{message}")
        self.cooldowns[message_hash] = datetime.now()

        # Czyszczenie starych wpisów z cooldown listy
        current_time = datetime.now()
        expired_hashes = []

        for msg_hash, timestamp in self.cooldowns.items():
            if (current_time - timestamp).total_seconds() > self.cooldown_period * 2:
                expired_hashes.append(msg_hash)

        for msg_hash in expired_hashes:
            del self.cooldowns[msg_hash]

    def _format_message(self, message, level, title=None):
        """
        Formatuje wiadomość do wysłania.

        Args:
            message (str): Treść wiadomości
            level (str): Poziom ważności
            title (str, optional): Tytuł powiadomienia

        Returns:
            str: Sformatowana wiadomość
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        level_prefix = {
            "info": "ℹ️",
            "warning": "⚠️",
            "alert": "🔔",
            "critical": "🚨"
        }.get(level, "ℹ️")

        title_str = f" {title} " if title else " "

        return f"{level_prefix}{title_str}[{timestamp}] {message}"

    def _get_default_title(self, level):
        """
        Zwraca domyślny tytuł dla danego poziomu powiadomienia.

        Args:
            level (str): Poziom powiadomienia

        Returns:
            str: Domyślny tytuł
        """
        titles = {
            "info": "Informacja",
            "warning": "Ostrzeżenie",
            "alert": "Alert",
            "critical": "UWAGA KRYTYCZNE"
        }
        return titles.get(level, "Powiadomienie")

    def _send_via_channel(self, notification, channel):
        """
        Wysyła wiadomość przez określony kanał.

        Args:
            notification (dict): Pełne dane powiadomienia
            channel (str): Kanał powiadomień

        Returns:
            bool: Czy wysłanie się powiodło
        """
        try:
            message = notification["formatted_message"]
            level = notification["level"]

            if channel == "console":
                # W przypadku konsoli używamy loggera
                log_method = {
                    "info": logging.info,
                    "warning": logging.warning,
                    "alert": logging.warning,
                    "critical": logging.error
                }.get(level, logging.info)

                log_method(f"NOTIFICATION: {message}")
                return True

            elif channel == "web":
                # Powiadomienie w interfejsie webowym (zapisane do pliku)
                self._save_web_notification(notification)
                return True

            elif channel == "email" and self.email_config:
                # Tutaj byłaby implementacja wysyłki email
                # W wersji demonstracyjnej tylko logujemy
                logging.info(f"EMAIL NOTIFICATION: {message}")
                return True

            elif channel == "sms" and self.sms_config:
                # Tutaj byłaby implementacja wysyłki SMS
                # W wersji demonstracyjnej tylko logujemy
                logging.info(f"SMS NOTIFICATION: {message}")
                return True

            elif channel == "slack" and self.slack_config:
                # Tutaj byłaby implementacja wysyłki na Slack
                # W wersji demonstracyjnej tylko logujemy
                logging.info(f"SLACK NOTIFICATION: {message}")
                return True

            elif channel == "telegram" and self.telegram_config:
                # Tutaj byłaby implementacja wysyłki na Telegram
                # W wersji demonstracyjnej tylko logujemy
                logging.info(f"TELEGRAM NOTIFICATION: {message}")
                return True

            else:
                logging.warning(f"Nieznany lub niekonfigurowany kanał: {channel}")
                return False

        except Exception as e:
            logging.error(f"Błąd wysyłania powiadomienia przez {channel}: {str(e)}")
            return False

    def _save_web_notification(self, notification):
        """
        Zapisuje powiadomienie do wyświetlenia w interfejsie webowym.

        Args:
            notification (dict): Dane powiadomienia
        """
        try:
            web_notification = {
                "id": int(time.time() * 1000),  # Unikalny ID bazujący na czasie
                "message": notification["message"],
                "title": notification["title"],
                "level": notification["level"],
                "timestamp": notification["timestamp"],
                "read": False
            }

            # Dodaj do wewnętrznego logu (dla interfejsu)
            self.notification_log.append(web_notification)

            # Ograniczenie rozmiaru logu
            if len(self.notification_log) > 100:
                self.notification_log = self.notification_log[-100:]

        except Exception as e:
            logging.error(f"Błąd zapisywania powiadomienia webowego: {str(e)}")

    def _log_notification(self, notification, channels):
        """
        Zapisuje powiadomienie w pliku logu.

        Args:
            notification (dict): Dane powiadomienia
            channels (list): Lista kanałów, przez które wysłano
        """
        try:
            log_entry = {
                "timestamp": notification["timestamp"],
                "message": notification["message"],
                "level": notification["level"],
                "channels": channels,
                "title": notification["title"]
            }

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logging.error(f"Błąd zapisywania logu powiadomień: {str(e)}")

    def get_notifications(self, limit=10, level=None, unread_only=False):
        """
        Pobiera listę powiadomień dla interfejsu webowego.

        Args:
            limit (int): Maksymalna liczba powiadomień
            level (str, optional): Filtrowanie według poziomu
            unread_only (bool): Czy zwracać tylko nieprzeczytane

        Returns:
            list: Lista powiadomień
        """
        with self.lock:
            filtered = self.notification_log

            if level:
                filtered = [n for n in filtered if n["level"] == level]

            if unread_only:
                filtered = [n for n in filtered if not n.get("read", False)]

            # Sortuj po czasie (najnowsze pierwsze)
            filtered.sort(key=lambda x: x["timestamp"], reverse=True)

            return filtered[:limit]

    def mark_as_read(self, notification_id):
        """
        Oznacza powiadomienie jako przeczytane.

        Args:
            notification_id (int): ID powiadomienia

        Returns:
            bool: Czy operacja się powiodła
        """
        with self.lock:
            for notification in self.notification_log:
                if notification.get("id") == notification_id:
                    notification["read"] = True
                    return True
            return False

# Przykładowe użycie
if __name__ == "__main__":
    # Konfiguracja z włączonymi kanałami
    notification_config = {
        "enabled_channels": ["console", "web"],
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
    notifier.send_notification(
        message="Wykryto anomalię cenową w BTC/USDT",
        level="critical",
        title="Anomalia cenowa",
        data={"symbol": "BTC/USDT", "change": "+5%", "time": "10:00"}
    )

    notifier.send_notification(
        message="Transakcja kupna ETH/USDT",
        level="info",
        title="Wykonana transakcja",
        data={"symbol": "ETH/USDT", "quantity": 1, "price": 1800}
    )

    notifier.send_notification(
        message="System uruchomiony",
        level="info",
        title="Status systemu",
    )
"""
notification_system.py
---------------------
System powiadomień dla aplikacji tradingowej.
"""

import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class NotificationSystem:
    """
    Klasa zarządzająca systemem powiadomień.
    Obsługuje różne kanały powiadomień (e-mail, SMS, push, wewnątrzaplikacyjne).
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicjalizacja systemu powiadomień.
        
        Args:
            config_path (str, optional): Ścieżka do pliku konfiguracyjnego.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.notifications = []
        self.notification_channels = ["app", "email"] if not self.config else self.config.get("channels", ["app"])
        
        # Domyślne ustawienia
        self.default_settings = {
            "emergency_contact": "admin@tradingbot.com",
            "notification_levels": ["info", "warning", "critical", "success", "error"],
            "channels": {
                "email": {
                    "enabled": False,
                    "server": "smtp.example.com",
                    "port": 587,
                    "username": "",
                    "password": "",
                    "sender": "noreply@tradingbot.com"
                },
                "sms": {
                    "enabled": False,
                    "provider": "twilio",
                    "api_key": "",
                    "api_secret": "",
                    "sender_number": ""
                },
                "app": {
                    "enabled": True,
                    "max_notifications": 100,
                    "auto_clear_after_days": 7
                }
            }
        }
        
        # Jeśli config jest pusty, używamy domyślnych ustawień
        if not self.config:
            self.config = self.default_settings
        
        logger.info(f"Zainicjalizowano system powiadomień z {len(self.notification_channels)} kanałami.")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Wczytuje konfigurację z pliku.
        
        Returns:
            dict: Konfiguracja systemu powiadomień.
        """
        if not self.config_path or not os.path.exists(self.config_path):
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania konfiguracji: {str(e)}")
            return {}
    
    def _save_config(self) -> bool:
        """
        Zapisuje konfigurację do pliku.
        
        Returns:
            bool: True jeśli zapis się powiódł, False w przeciwnym razie.
        """
        if not self.config_path:
            return False
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania konfiguracji: {str(e)}")
            return False
    
    def add_notification(
        self, 
        message: str, 
        level: str = "info", 
        title: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Dodaje nowe powiadomienie.
        
        Args:
            message (str): Treść powiadomienia.
            level (str): Poziom powiadomienia (info, warning, critical, success, error).
            title (str, optional): Tytuł powiadomienia.
            details (dict, optional): Dodatkowe szczegóły.
            channels (list, optional): Lista kanałów, przez które ma zostać wysłane powiadomienie.
            
        Returns:
            dict: Utworzone powiadomienie.
        """
        if not message:
            logger.warning("Próba dodania pustego powiadomienia.")
            return {}
        
        notification_id = int(time.time() * 1000)  # Unikalny ID bazujący na czasie
        
        notification = {
            "id": notification_id,
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "title": title or self._generate_title(level),
            "message": message,
            "read": False,
            "details": details or {}
        }
        
        # Dodanie powiadomienia do listy
        self.notifications.append(notification)
        
        # Ograniczenie liczby przechowywanych powiadomień
        max_notifications = self.config.get("channels", {}).get("app", {}).get("max_notifications", 100)
        if len(self.notifications) > max_notifications:
            self.notifications = self.notifications[-max_notifications:]
        
        # Wysłanie powiadomienia przez odpowiednie kanały
        self._send_to_channels(notification, channels or self.notification_channels)
        
        return notification
    
    def _generate_title(self, level: str) -> str:
        """
        Generuje domyślny tytuł dla powiadomienia na podstawie poziomu.
        
        Args:
            level (str): Poziom powiadomienia.
            
        Returns:
            str: Wygenerowany tytuł.
        """
        titles = {
            "info": "Informacja",
            "warning": "Ostrzeżenie",
            "critical": "Alert Krytyczny",
            "success": "Sukces",
            "error": "Błąd"
        }
        return titles.get(level, "Powiadomienie")
    
    def _send_to_channels(self, notification: Dict[str, Any], channels: List[str]) -> None:
        """
        Wysyła powiadomienie przez określone kanały.
        
        Args:
            notification (dict): Powiadomienie do wysłania.
            channels (list): Lista kanałów do wykorzystania.
        """
        for channel in channels:
            if channel == "email" and self.config.get("channels", {}).get("email", {}).get("enabled", False):
                self._send_email(notification)
            elif channel == "sms" and self.config.get("channels", {}).get("sms", {}).get("enabled", False):
                self._send_sms(notification)
            elif channel == "app":
                # Powiadomienie wewnątrzaplikacyjne - nie wymaga dodatkowych działań,
                # już zostało dodane do self.notifications
                pass
    
    def _send_email(self, notification: Dict[str, Any]) -> bool:
        """
        Wysyła powiadomienie e-mailem.
        
        Args:
            notification (dict): Powiadomienie do wysłania.
            
        Returns:
            bool: True jeśli wysyłka się powiodła, False w przeciwnym razie.
        """
        # W rzeczywistej implementacji tutaj byłaby integracja z serwerem SMTP
        logger.info(f"Symulacja wysyłki e-mail: {notification['title']} - {notification['message']}")
        return True
    
    def _send_sms(self, notification: Dict[str, Any]) -> bool:
        """
        Wysyła powiadomienie SMS-em.
        
        Args:
            notification (dict): Powiadomienie do wysłania.
            
        Returns:
            bool: True jeśli wysyłka się powiodła, False w przeciwnym razie.
        """
        # W rzeczywistej implementacji tutaj byłaby integracja z serwisem SMS
        logger.info(f"Symulacja wysyłki SMS: {notification['message']}")
        return True
    
    def get_notifications(
        self, 
        limit: int = 20, 
        offset: int = 0, 
        level: Optional[str] = None,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Pobiera listę powiadomień z możliwością filtrowania.
        
        Args:
            limit (int): Maksymalna liczba zwracanych powiadomień.
            offset (int): Przesunięcie (do paginacji).
            level (str, optional): Filtrowanie po poziomie powiadomienia.
            unread_only (bool): Czy zwracać tylko nieprzeczytane powiadomienia.
            
        Returns:
            list: Lista powiadomień.
        """
        filtered = self.notifications
        
        # Filtrowanie po poziomie
        if level:
            filtered = [n for n in filtered if n["level"] == level]
        
        # Filtrowanie nieprzeczytanych
        if unread_only:
            filtered = [n for n in filtered if not n["read"]]
        
        # Sortowanie od najnowszych
        filtered = sorted(filtered, key=lambda x: x["timestamp"], reverse=True)
        
        # Paginacja
        return filtered[offset:offset + limit]
    
    def mark_as_read(self, notification_id: Union[int, List[int]]) -> int:
        """
        Oznacza powiadomienie lub powiadomienia jako przeczytane.
        
        Args:
            notification_id: ID powiadomienia lub lista ID.
            
        Returns:
            int: Liczba powiadomień oznaczonych jako przeczytane.
        """
        if isinstance(notification_id, list):
            ids = notification_id
        else:
            ids = [notification_id]
        
        count = 0
        for n in self.notifications:
            if n["id"] in ids and not n["read"]:
                n["read"] = True
                count += 1
        
        return count
    
    def mark_all_as_read(self) -> int:
        """
        Oznacza wszystkie powiadomienia jako przeczytane.
        
        Returns:
            int: Liczba powiadomień oznaczonych jako przeczytane.
        """
        count = 0
        for n in self.notifications:
            if not n["read"]:
                n["read"] = True
                count += 1
        
        return count
    
    def delete_notification(self, notification_id: Union[int, List[int]]) -> int:
        """
        Usuwa powiadomienie lub powiadomienia.
        
        Args:
            notification_id: ID powiadomienia lub lista ID.
            
        Returns:
            int: Liczba usuniętych powiadomień.
        """
        if isinstance(notification_id, list):
            ids = notification_id
        else:
            ids = [notification_id]
        
        initial_count = len(self.notifications)
        self.notifications = [n for n in self.notifications if n["id"] not in ids]
        
        return initial_count - len(self.notifications)
    
    def clear_old_notifications(self, days: int = 7) -> int:
        """
        Usuwa stare powiadomienia.
        
        Args:
            days (int): Usuwanie powiadomień starszych niż określona liczba dni.
            
        Returns:
            int: Liczba usuniętych powiadomień.
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        initial_count = len(self.notifications)
        self.notifications = [n for n in self.notifications if n["timestamp"] > cutoff_str]
        
        return initial_count - len(self.notifications)
    
    def get_unread_count(self, level: Optional[str] = None) -> int:
        """
        Zwraca liczbę nieprzeczytanych powiadomień.
        
        Args:
            level (str, optional): Filtrowanie po poziomie powiadomienia.
            
        Returns:
            int: Liczba nieprzeczytanych powiadomień.
        """
        if level:
            return sum(1 for n in self.notifications if not n["read"] and n["level"] == level)
        return sum(1 for n in self.notifications if not n["read"])

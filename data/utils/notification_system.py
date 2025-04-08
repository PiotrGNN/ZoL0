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
from typing import List, Dict, Any, Optional

class NotificationSystem:
    """
    System zarządzania powiadomieniami aplikacji.
    """

    def __init__(self, channels: List[str] = None, max_history: int = 100):
        """
        Inicjalizuje system powiadomień.

        Parameters:
            channels (List[str], optional): Lista kanałów powiadomień.
            max_history (int): Maksymalna liczba przechowywanych powiadomień w historii.
        """
        self.channels = channels or ["email", "app"]
        self.max_history = max_history
        self.notifications_history = []

        # Konfiguracja logowania
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger("notification_system")

        if not self.logger.handlers:
            file_handler = logging.FileHandler(os.path.join(log_dir, "notifications.log"))
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info(f"Zainicjalizowano system powiadomień z {len(self.channels)} kanałami.")

    def send_notification(self, message: str, level: str = "info", 
                         channels: List[str] = None, title: str = None,
                         data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Wysyła powiadomienie.

        Parameters:
            message (str): Treść powiadomienia.
            level (str): Poziom powiadomienia ('info', 'warning', 'error', 'critical').
            channels (List[str], optional): Lista kanałów do wysłania powiadomienia.
            title (str, optional): Tytuł powiadomienia.
            data (Dict[str, Any], optional): Dodatkowe dane powiadomienia.

        Returns:
            Dict[str, Any]: Wynik wysłania powiadomienia.
        """
        if channels is None:
            channels = self.channels

        if level not in ["info", "warning", "error", "critical"]:
            level = "info"

        if title is None:
            title = f"Powiadomienie {level}"

        if data is None:
            data = {}

        timestamp = int(time.time())
        notification = {
            "id": f"notif-{timestamp}-{len(self.notifications_history)}",
            "message": message,
            "title": title,
            "level": level,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            "channels": channels,
            "data": data
        }

        # Symulacja wysyłania powiadomień na różne kanały
        for channel in channels:
            # Tutaj byłaby faktyczna implementacja wysyłania powiadomień na różne kanały
            self.logger.info(f"Wysłano powiadomienie na kanał {channel}: {title} - {message}")

        # Dodanie do historii
        self.notifications_history.append(notification)

        # Ograniczenie rozmiaru historii
        if len(self.notifications_history) > self.max_history:
            self.notifications_history = self.notifications_history[-self.max_history:]

        return {"success": True, "notification": notification}

    def get_notifications(self, limit: int = 10, level: str = None, 
                         since: int = None) -> List[Dict[str, Any]]:
        """
        Pobiera powiadomienia z historii.

        Parameters:
            limit (int): Maksymalna liczba powiadomień do pobrania.
            level (str, optional): Filtrowanie po poziomie.
            since (int, optional): Pobieranie powiadomień od określonego timestampu.

        Returns:
            List[Dict[str, Any]]: Lista powiadomień.
        """
        # Filtrujemy powiadomienia
        filtered = self.notifications_history

        if level:
            filtered = [n for n in filtered if n["level"] == level]

        if since:
            filtered = [n for n in filtered if n["timestamp"] >= since]

        # Sortujemy od najnowszych
        filtered.sort(key=lambda x: x["timestamp"], reverse=True)

        return filtered[:limit]

    def get_recent_notifications(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Pobiera powiadomienia z ostatnich godzin.

        Parameters:
            hours (int): Liczba godzin wstecz.

        Returns:
            List[Dict[str, Any]]: Lista powiadomień.
        """
        since = int(time.time()) - (hours * 3600)
        return self.get_notifications(limit=100, since=since)

    def mark_as_read(self, notification_id: str) -> bool:
        """
        Oznacza powiadomienie jako przeczytane.

        Parameters:
            notification_id (str): ID powiadomienia.

        Returns:
            bool: True jeśli powiadomienie zostało oznaczone jako przeczytane.
        """
        for notification in self.notifications_history:
            if notification["id"] == notification_id:
                notification["read"] = True
                self.logger.info(f"Oznaczono powiadomienie {notification_id} jako przeczytane")
                return True

        self.logger.warning(f"Nie znaleziono powiadomienia o ID {notification_id}")
        return False

    def get_notifications_count(self, level: str = None) -> int:
        """
        Pobiera liczbę powiadomień określonego poziomu.

        Parameters:
            level (str, optional): Poziom powiadomień.

        Returns:
            int: Liczba powiadomień.
        """
        if level:
            return len([n for n in self.notifications_history if n["level"] == level])
        return len(self.notifications_history)

# Dodanie przykładowych powiadomień do systemu
if __name__ == "__main__":
    notifier = NotificationSystem()

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
--------------------
System powiadomień dla aplikacji.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/notifications.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NotificationChannel:
    """Bazowa klasa kanału powiadomień."""
    
    def __init__(self, name: str):
        """
        Inicjalizacja kanału powiadomień.
        
        Args:
            name: Nazwa kanału
        """
        self.name = name
        self.enabled = True
        logger.info(f"Kanał powiadomień '{name}' utworzony")
    
    def send(self, message: str, level: str = "info") -> bool:
        """
        Wysyła powiadomienie.
        
        Args:
            message: Treść powiadomienia
            level: Poziom ważności powiadomienia
            
        Returns:
            bool: True jeśli wysłano pomyślnie, False w przeciwnym razie
        """
        if not self.enabled:
            logger.warning(f"Kanał '{self.name}' jest wyłączony, nie wysłano powiadomienia")
            return False
            
        logger.info(f"[{self.name}] Wysłano powiadomienie ({level}): {message}")
        return True

class ConsoleNotificationChannel(NotificationChannel):
    """Kanał powiadomień konsolowych."""
    
    def __init__(self):
        """Inicjalizacja kanału powiadomień konsolowych."""
        super().__init__("Console")
    
    def send(self, message: str, level: str = "info") -> bool:
        """
        Wysyła powiadomienie do konsoli.
        
        Args:
            message: Treść powiadomienia
            level: Poziom ważności powiadomienia
            
        Returns:
            bool: True jeśli wysłano pomyślnie, False w przeciwnym razie
        """
        if not self.enabled:
            return False
            
        level_prefixes = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅"
        }
        prefix = level_prefixes.get(level, "ℹ️")
        
        print(f"{prefix} {message}")
        return super().send(message, level)

class LogNotificationChannel(NotificationChannel):
    """Kanał powiadomień do pliku logów."""
    
    def __init__(self):
        """Inicjalizacja kanału powiadomień do plików logów."""
        super().__init__("Log")
    
    def send(self, message: str, level: str = "info") -> bool:
        """
        Wysyła powiadomienie do pliku logów.
        
        Args:
            message: Treść powiadomienia
            level: Poziom ważności powiadomienia
            
        Returns:
            bool: True jeśli wysłano pomyślnie, False w przeciwnym razie
        """
        if not self.enabled:
            return False
            
        log_methods = {
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
            "debug": logger.debug
        }
        
        log_method = log_methods.get(level, logger.info)
        log_method(f"NOTIFICATION: {message}")
        
        return super().send(message, level)

class NotificationSystem:
    """System zarządzania powiadomieniami."""
    
    def __init__(self):
        """Inicjalizacja systemu powiadomień."""
        self.channels = {
            "console": ConsoleNotificationChannel(),
            "log": LogNotificationChannel()
        }
        self.notifications_history = []
        logger.info(f"System powiadomień zainicjalizowany z {len(self.channels)} kanałami")
    
    def send(self, message: str, level: str = "info", channel: Optional[str] = None) -> bool:
        """
        Wysyła powiadomienie przez wybrany kanał lub wszystkie kanały.
        
        Args:
            message: Treść powiadomienia
            level: Poziom ważności powiadomienia
            channel: Nazwa określonego kanału lub None dla wszystkich kanałów
            
        Returns:
            bool: True jeśli wysłano pomyślnie, False w przeciwnym razie
        """
        success = False
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Zapisanie powiadomienia w historii
        self.notifications_history.append({
            "message": message,
            "level": level,
            "timestamp": timestamp
        })
        
        # Ograniczenie historii do 100 ostatnich powiadomień
        if len(self.notifications_history) > 100:
            self.notifications_history = self.notifications_history[-100:]
            
        # Wysłanie powiadomienia przez wybrane kanały
        if channel is not None:
            if channel in self.channels:
                return self.channels[channel].send(message, level)
            else:
                logger.warning(f"Kanał '{channel}' nie istnieje")
                return False
        
        # Wysłanie przez wszystkie kanały
        for channel_name, channel_obj in self.channels.items():
            if channel_obj.send(message, level):
                success = True
                
        return success
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Zwraca historię powiadomień.
        
        Args:
            limit: Maksymalna liczba powiadomień do zwrócenia
            
        Returns:
            List[Dict[str, Any]]: Lista powiadomień
        """
        return self.notifications_history[-limit:]

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
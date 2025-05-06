"""Notification system for the trading platform."""

import os
import json
import threading
import queue
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

class NotificationPriority:
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationChannel:
    """Available notification channels."""
    CONSOLE = "console"
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"

@dataclass
class Notification:
    """Container for notification messages."""
    title: str
    message: str
    priority: str = NotificationPriority.MEDIUM
    channel: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.channel is None:
            self.channel = [NotificationChannel.CONSOLE]
        if self.metadata is None:
            self.metadata = {}
        self.metadata['timestamp'] = datetime.now().isoformat()

class NotificationManager:
    """Manages system notifications."""
    
    def __init__(self):
        self.notification_queue = queue.Queue()
        self._handlers = {}
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_notifications)
        self._worker_thread.daemon = True
        self._worker_thread.start()
        
        # Register default handlers
        self.register_handler(NotificationChannel.CONSOLE, self._console_handler)

    def register_handler(self, channel: str, handler_func):
        """Register a notification handler for a channel."""
        self._handlers[channel] = handler_func

    def send_notification(self, notification: Notification):
        """Send a notification through specified channels."""
        self.notification_queue.put(notification)

    def _process_notifications(self):
        """Process notifications in the queue."""
        while self._running:
            try:
                notification = self.notification_queue.get(timeout=1.0)
                for channel in notification.channel:
                    handler = self._handlers.get(channel)
                    if handler:
                        try:
                            handler(notification)
                        except Exception as e:
                            print(f"Error in notification handler ({channel}): {e}")
                self.notification_queue.task_done()
            except queue.Empty:
                continue

    def _console_handler(self, notification: Notification):
        """Default console notification handler."""
        priority_colors = {
            NotificationPriority.LOW: "\033[32m",      # Green
            NotificationPriority.MEDIUM: "\033[33m",   # Yellow
            NotificationPriority.HIGH: "\033[31m",     # Red
            NotificationPriority.CRITICAL: "\033[41m"  # Red background
        }
        color = priority_colors.get(notification.priority, "\033[0m")
        reset = "\033[0m"
        
        print(f"\n{color}[{notification.priority.upper()}] {notification.title}{reset}")
        print(f"{notification.message}")
        if notification.metadata:
            print("Metadata:", json.dumps(notification.metadata, indent=2))

    def stop(self):
        """Stop the notification manager."""
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join()

# Global notification manager instance
_notification_manager = NotificationManager()

def get_notification_manager() -> NotificationManager:
    """Get the global notification manager instance."""
    return _notification_manager

# Example usage
if __name__ == "__main__":
    manager = get_notification_manager()
    
    # Send test notifications
    notifications = [
        Notification(
            "Test Notification",
            "This is a test message",
            priority=NotificationPriority.LOW,
            channel=[NotificationChannel.CONSOLE, NotificationChannel.EMAIL]
        ),
        Notification(
            "Critical Alert",
            "System resources critically low!",
            priority=NotificationPriority.CRITICAL,
            channel=NotificationChannel.CONSOLE,
            metadata={"cpu_usage": "95%", "memory_available": "100MB"}
        )
    ]
    
    for notification in notifications:
        manager.send_notification(notification)
    
    # Wait a bit for notifications to be processed
    import time
    time.sleep(2)
    
    # Stop the notification manager
    manager.stop()
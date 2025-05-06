"""
Enhanced notification system with better error reporting and alerts.
"""

import logging
import smtplib
import json
import requests
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import queue
from dataclasses import dataclass

# Import system logger
from data.logging.system_logger import get_logger
logger = get_logger()

@dataclass
class NotificationConfig:
    """Notification configuration container."""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    slack_webhook: str = ""
    discord_webhook: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    notification_batch_size: int = 10
    notification_batch_interval: int = 300  # seconds

class NotificationLevel:
    """Notification priority levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

    @staticmethod
    def to_string(level: int) -> str:
        return {
            0: "DEBUG",
            1: "INFO",
            2: "WARNING",
            3: "ERROR",
            4: "CRITICAL"
        }.get(level, "UNKNOWN")

class Notification:
    """Container for notification messages."""
    
    def __init__(self, 
                 message: str,
                 level: int = NotificationLevel.INFO,
                 context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.level = level
        self.context = context or {}
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary."""
        return {
            'message': self.message,
            'level': NotificationLevel.to_string(self.level),
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }

class NotificationSystem:
    """Enhanced notification system."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.notification_queue = queue.Queue()
        self._notification_handlers: Dict[str, Callable] = {}
        self._notification_filters: Dict[str, Callable] = {}
        self._batch_notifications: List[Notification] = []
        self._last_batch_send = datetime.now()
        
        # Start notification processing thread
        self._stop_processing = threading.Event()
        self._processing_thread = threading.Thread(target=self._process_notifications)
        self._processing_thread.start()
        
        # Register default handlers
        self.register_handler('email', self._send_email_notification)
        self.register_handler('slack', self._send_slack_notification)
        self.register_handler('discord', self._send_discord_notification)
        self.register_handler('telegram', self._send_telegram_notification)
        
        logger.log_info("Notification system initialized")

    def notify(self, 
               message: str,
               level: int = NotificationLevel.INFO,
               context: Optional[Dict[str, Any]] = None,
               immediate: bool = False) -> bool:
        """Send notification."""
        try:
            notification = Notification(message, level, context)
            
            if immediate:
                return self._send_notification(notification)
            else:
                self.notification_queue.put(notification)
                return True
        except Exception as e:
            logger.log_error(f"Error sending notification: {e}")
            return False

    def _process_notifications(self) -> None:
        """Process notifications from queue."""
        while not self._stop_processing.is_set():
            try:
                # Get notification with timeout
                try:
                    notification = self.notification_queue.get(timeout=1)
                except queue.Empty:
                    # Check if we should send batched notifications
                    if self._should_send_batch():
                        self._send_notification_batch()
                    continue
                
                # Add to batch if not immediate
                self._batch_notifications.append(notification)
                
                # Send batch if full
                if len(self._batch_notifications) >= self.config.notification_batch_size:
                    self._send_notification_batch()
            except Exception as e:
                logger.log_error(f"Error in notification processing: {e}")

    def _should_send_batch(self) -> bool:
        """Check if batched notifications should be sent."""
        if not self._batch_notifications:
            return False
            
        seconds_since_last = (datetime.now() - self._last_batch_send).total_seconds()
        return seconds_since_last >= self.config.notification_batch_interval

    def _send_notification_batch(self) -> None:
        """Send batched notifications."""
        if not self._batch_notifications:
            return
            
        try:
            # Group notifications by level
            notifications_by_level = {}
            for notification in self._batch_notifications:
                if notification.level not in notifications_by_level:
                    notifications_by_level[notification.level] = []
                notifications_by_level[notification.level].append(notification)
            
            # Send grouped notifications
            for level, notifications in notifications_by_level.items():
                if notifications:
                    self._send_grouped_notifications(notifications)
            
            self._batch_notifications = []
            self._last_batch_send = datetime.now()
        except Exception as e:
            logger.log_error(f"Error sending notification batch: {e}")

    def _send_grouped_notifications(self, notifications: List[Notification]) -> None:
        """Send group of notifications."""
        try:
            level = notifications[0].level
            level_str = NotificationLevel.to_string(level)
            
            # Create summary message
            summary = f"{len(notifications)} {level_str} notifications:\n\n"
            for notification in notifications:
                summary += f"- {notification.message}\n"
                if notification.context:
                    summary += f"  Context: {json.dumps(notification.context)}\n"
            
            # Send through all handlers
            for handler in self._notification_handlers.values():
                try:
                    handler(summary, level)
                except Exception as e:
                    logger.log_error(f"Error in notification handler: {e}")
        except Exception as e:
            logger.log_error(f"Error sending grouped notifications: {e}")

    def _send_notification(self, notification: Notification) -> bool:
        """Send single notification through all handlers."""
        success = True
        
        # Apply filters
        for filter_func in self._notification_filters.values():
            if not filter_func(notification):
                return True
        
        # Send through all handlers
        for handler in self._notification_handlers.values():
            try:
                handler(notification.message, notification.level)
            except Exception as e:
                logger.log_error(f"Error in notification handler: {e}")
                success = False
        
        return success

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register notification handler."""
        self._notification_handlers[name] = handler

    def register_filter(self, name: str, filter_func: Callable) -> None:
        """Register notification filter."""
        self._notification_filters[name] = filter_func

    def _send_email_notification(self, message: str, level: int) -> None:
        """Send email notification."""
        if not (self.config.smtp_user and self.config.smtp_password):
            return
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_user
            msg['To'] = self.config.smtp_user
            msg['Subject'] = f"{NotificationLevel.to_string(level)} Notification"
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
        except Exception as e:
            logger.log_error(f"Error sending email notification: {e}")

    def _send_slack_notification(self, message: str, level: int) -> None:
        """Send Slack notification."""
        if not self.config.slack_webhook:
            return
            
        try:
            color = {
                NotificationLevel.DEBUG: "#808080",
                NotificationLevel.INFO: "#2196F3",
                NotificationLevel.WARNING: "#FFC107",
                NotificationLevel.ERROR: "#F44336",
                NotificationLevel.CRITICAL: "#9C27B0"
            }.get(level, "#000000")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": NotificationLevel.to_string(level),
                    "text": message,
                    "ts": int(datetime.now().timestamp())
                }]
            }
            
            requests.post(self.config.slack_webhook, json=payload)
        except Exception as e:
            logger.log_error(f"Error sending Slack notification: {e}")

    def _send_discord_notification(self, message: str, level: int) -> None:
        """Send Discord notification."""
        if not self.config.discord_webhook:
            return
            
        try:
            color = {
                NotificationLevel.DEBUG: 0x808080,
                NotificationLevel.INFO: 0x2196F3,
                NotificationLevel.WARNING: 0xFFC107,
                NotificationLevel.ERROR: 0xF44336,
                NotificationLevel.CRITICAL: 0x9C27B0
            }.get(level, 0x000000)
            
            payload = {
                "embeds": [{
                    "title": NotificationLevel.to_string(level),
                    "description": message,
                    "color": color,
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            requests.post(self.config.discord_webhook, json=payload)
        except Exception as e:
            logger.log_error(f"Error sending Discord notification: {e}")

    def _send_telegram_notification(self, message: str, level: int) -> None:
        """Send Telegram notification."""
        if not (self.config.telegram_bot_token and self.config.telegram_chat_id):
            return
            
        try:
            level_str = NotificationLevel.to_string(level)
            formatted_message = f"*{level_str}*\n{message}"
            
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.config.telegram_chat_id,
                "text": formatted_message,
                "parse_mode": "Markdown"
            }
            
            requests.post(url, json=payload)
        except Exception as e:
            logger.log_error(f"Error sending Telegram notification: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        self._stop_processing.set()
        if hasattr(self, '_processing_thread'):
            self._processing_thread.join()

# Create global notification system instance with default config
notification_system = NotificationSystem(NotificationConfig())

def get_notification_system() -> NotificationSystem:
    """Get global notification system instance."""
    return notification_system

# Example usage
if __name__ == "__main__":
    # Configure notification system
    config = NotificationConfig(
        smtp_user="your-email@gmail.com",
        smtp_password="your-password",
        slack_webhook="your-slack-webhook-url",
        discord_webhook="your-discord-webhook-url",
        telegram_bot_token="your-telegram-bot-token",
        telegram_chat_id="your-telegram-chat-id"
    )
    
    system = NotificationSystem(config)
    
    # Send some test notifications
    system.notify(
        "Test info message",
        NotificationLevel.INFO,
        {"test_key": "test_value"}
    )
    
    system.notify(
        "Test error message",
        NotificationLevel.ERROR,
        {"error_code": "TEST_ERROR"},
        immediate=True
    )
    
    # Add custom notification handler
    def custom_handler(message: str, level: int):
        print(f"Custom handler: [{NotificationLevel.to_string(level)}] {message}")
    
    system.register_handler("custom", custom_handler)
    
    # Add notification filter
    def debug_filter(notification: Notification) -> bool:
        return notification.level > NotificationLevel.DEBUG
    
    system.register_filter("no_debug", debug_filter)
    
    # Send another test notification
    system.notify(
        "Test warning message",
        NotificationLevel.WARNING
    )
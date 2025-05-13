"""
core.notifications - Stub implementation for notifications system.
"""

class Notification:
    def __init__(self, title, message, priority=None):
        self.title = title
        self.message = message
        self.priority = priority

class NotificationPriority:
    LOW = "low"
    CRITICAL = "critical"

def get_notification_manager():
    class NotificationManager:
        def send_notification(self, notification):
            pass  # Stub: No-op
    return NotificationManager()

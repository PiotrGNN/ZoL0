"""
Moduł zarządzający systemem powiadomień w czasie rzeczywistym.
"""

import json
import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, List, Optional
import sqlite3
from queue import PriorityQueue
import threading
import time

logger = logging.getLogger(__name__)

class NotificationManager:
    def __init__(self, db_path: str = 'users.db'):
        self.db_path = db_path
        self.notification_queue = PriorityQueue()
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _get_channel_config(self, user_id: int, channel_type: str) -> Optional[Dict]:
        """Pobiera konfigurację kanału komunikacji."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                SELECT config_json, rate_limit 
                FROM communication_channels 
                WHERE user_id = ? AND channel_type = ? AND is_active = 1
            """, (user_id, channel_type))
            result = c.fetchone()
            if result:
                return {
                    'config': json.loads(result[0]),
                    'rate_limit': result[1]
                }
            return None
        except Exception as e:
            logger.error(f"Błąd podczas pobierania konfiguracji kanału: {e}")
            return None
        finally:
            conn.close()

    def send_email(self, user_id: int, subject: str, body: str) -> bool:
        """Wysyła powiadomienie email."""
        config = self._get_channel_config(user_id, 'email')
        if not config:
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = config['config']['sender']
            msg['To'] = config['config']['recipient']
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(config['config']['smtp_server'], config['config']['smtp_port'])
            server.starttls()
            server.login(config['config']['username'], config['config']['password'])
            server.send_message(msg)
            server.quit()

            self._log_notification(user_id, 'email', subject, 'delivered')
            return True
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania emaila: {e}")
            self._log_notification(user_id, 'email', subject, 'failed', str(e))
            return False

    def send_telegram(self, user_id: int, message: str) -> bool:
        """Wysyła powiadomienie przez Telegram."""
        config = self._get_channel_config(user_id, 'telegram')
        if not config:
            return False

        try:
            url = f"https://api.telegram.org/bot{config['config']['bot_token']}/sendMessage"
            payload = {
                'chat_id': config['config']['chat_id'],
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()

            self._log_notification(user_id, 'telegram', message, 'delivered')
            return True
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania wiadomości Telegram: {e}")
            self._log_notification(user_id, 'telegram', message, 'failed', str(e))
            return False

    def send_notification(self, user_id: int, notification_type: str, message: str, priority: int = 1) -> bool:
        """Dodaje powiadomienie do kolejki z określonym priorytetem."""
        try:
            self.notification_queue.put((
                priority,
                {
                    'user_id': user_id,
                    'type': notification_type,
                    'message': message,
                    'timestamp': datetime.now()
                }
            ))
            return True
        except Exception as e:
            logger.error(f"Błąd podczas dodawania powiadomienia do kolejki: {e}")
            return False

    def _process_queue(self):
        """Przetwarza kolejkę powiadomień w osobnym wątku."""
        while self.running:
            try:
                if not self.notification_queue.empty():
                    priority, notification = self.notification_queue.get()
                    self._send_notification_by_type(
                        notification['user_id'],
                        notification['type'],
                        notification['message']
                    )
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Błąd podczas przetwarzania kolejki powiadomień: {e}")

    def _send_notification_by_type(self, user_id: int, notification_type: str, message: str):
        """Wysyła powiadomienie odpowiednim kanałem na podstawie typu."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Pobierz preferowane kanały dla danego typu powiadomienia
            c.execute("""
                SELECT channel_type, priority 
                FROM communication_channels 
                WHERE user_id = ? AND is_active = 1 
                AND notification_types LIKE ?
                ORDER BY priority
            """, (user_id, f"%{notification_type}%"))
            
            channels = c.fetchall()
            
            for channel_type, _ in channels:
                if channel_type == 'email':
                    if self.send_email(user_id, f"Powiadomienie: {notification_type}", message):
                        break
                elif channel_type == 'telegram':
                    if self.send_telegram(user_id, message):
                        break
                # Można dodać więcej kanałów
        
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania powiadomienia: {e}")
        finally:
            conn.close()

    def _log_notification(self, user_id: int, channel: str, message: str, status: str, error_message: str = None):
        """Zapisuje historię powiadomienia w bazie danych."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                INSERT INTO notification_history (
                    user_id, channel, message, status, error_message
                ) VALUES (?, ?, ?, ?, ?)
            """, (user_id, channel, message, status, error_message))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Błąd podczas logowania powiadomienia: {e}")
        finally:
            conn.close()

    def get_notification_history(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Pobiera historię powiadomień użytkownika."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                SELECT * FROM notification_history 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (user_id, limit))
            
            columns = [description[0] for description in c.description]
            history = [dict(zip(columns, row)) for row in c.fetchall()]
            
            return history
        except Exception as e:
            logger.error(f"Błąd podczas pobierania historii powiadomień: {e}")
            return []
        finally:
            conn.close()

    def get_channel_settings(self, user_id: int) -> List[Dict[str, Any]]:
        """Pobiera ustawienia kanałów komunikacji użytkownika."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                SELECT * FROM communication_channels 
                WHERE user_id = ?
            """, (user_id,))
            
            columns = [description[0] for description in c.description]
            settings = [dict(zip(columns, row)) for row in c.fetchall()]
            
            return settings
        except Exception as e:
            logger.error(f"Błąd podczas pobierania ustawień kanałów: {e}")
            return []
        finally:
            conn.close()

    def update_channel_settings(self, user_id: int, channel_type: str, settings: Dict[str, Any]) -> bool:
        """Aktualizuje ustawienia kanału komunikacji."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                UPDATE communication_channels 
                SET config_json = ?,
                    is_active = ?,
                    priority = ?,
                    notification_types = ?,
                    rate_limit = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND channel_type = ?
            """, (
                json.dumps(settings.get('config', {})),
                settings.get('is_active', True),
                settings.get('priority', 1),
                json.dumps(settings.get('notification_types', [])),
                settings.get('rate_limit', 60),
                user_id,
                channel_type
            ))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji ustawień kanału: {e}")
            return False
        finally:
            conn.close()

    def cleanup(self):
        """Zatrzymuje wątek przetwarzający kolejkę."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join()
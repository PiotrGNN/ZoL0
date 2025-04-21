"""
advanced_alerts.py
-----------------
Moduł obsługujący zaawansowane powiadomienia i alerty.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Union
import sqlite3
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import requests
from data.database import DB_PATH

logger = logging.getLogger(__name__)

class NotificationManager:
    def __init__(self, user_id: Optional[int] = None, config_path: str = 'config/notifications.json'):
        self.user_id = user_id
        self.config_path = config_path
        
        try:
            self.conn = sqlite3.connect(DB_PATH)
        except Exception as e:
            logger.error(f"Błąd podczas łączenia z bazą danych: {e}")
            self.conn = None
            
        self.config = self._load_config(config_path)
        
    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def _load_config(self, config_path: str) -> Dict:
        """Ładuje konfigurację powiadomień z pliku JSON."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Jeśli plik nie istnieje, tworzymy katalog i domyślną konfigurację
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                default_config = {
                    "email": {
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 465,
                        "username": "",
                        "password": "",
                        "sender": ""
                    },
                    "telegram": {
                        "bot_token": "",
                        "chat_id": ""
                    }
                }
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                logger.info(f"Utworzono domyślny plik konfiguracyjny: {config_path}")
                return default_config
        except Exception as e:
            logger.error(f"Błąd podczas ładowania konfiguracji powiadomień: {e}")
            return {}

    def add_pattern_alert(self, alert_data: Dict) -> bool:
        """Dodaje nowy alert wzorca cenowego."""
        try:
            self.conn.execute("""
            INSERT INTO pattern_alerts (
                user_id, symbol, pattern_type, confidence,
                price_level, volume_confirmation, timeframe,
                additional_indicators, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.user_id,
                alert_data['symbol'],
                alert_data['pattern_type'],
                alert_data['confidence'],
                alert_data.get('price_level'),
                alert_data.get('volume_confirmation', False),
                alert_data.get('timeframe', '1h'),
                json.dumps(alert_data.get('additional_indicators', {})),
                'active'
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Błąd podczas dodawania alertu wzorca: {e}")
            return False

    def add_ai_anomaly_alert(self, alert_data: Dict) -> bool:
        """Dodaje nowy alert anomalii wykrytej przez AI."""
        try:
            self.conn.execute("""
            INSERT INTO ai_anomaly_alerts (
                user_id, model_name, anomaly_type, severity,
                description, affected_assets, confidence_score,
                false_positive_probability, resolution_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.user_id,
                alert_data['model_name'],
                alert_data['anomaly_type'],
                alert_data['severity'],
                alert_data['description'],
                json.dumps(alert_data.get('affected_assets', [])),
                alert_data['confidence_score'],
                alert_data.get('false_positive_probability', 0.0),
                'pending'
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Błąd podczas dodawania alertu anomalii: {e}")
            return False

    def configure_notification_channel(self, channel_data: Dict) -> bool:
        """Konfiguruje nowy kanał powiadomień."""
        try:
            self.conn.execute("""
            INSERT INTO notification_channels (
                user_id, channel_type, channel_name,
                configuration, priority
            ) VALUES (?, ?, ?, ?, ?)
            """, (
                self.user_id,
                channel_data['type'],
                channel_data['name'],
                json.dumps(channel_data['config']),
                channel_data.get('priority', 1)
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Błąd podczas konfiguracji kanału powiadomień: {e}")
            return False

    def send_notification(self, message: str, severity: str = 'info', channels: Optional[List[str]] = None) -> bool:
        """Wysyła powiadomienie przez skonfigurowane kanały."""
        try:
            if channels is None:
                # Pobierz wszystkie aktywne kanały
                cursor = self.conn.execute("""
                SELECT channel_type, configuration
                FROM notification_channels
                WHERE user_id = ? AND is_active = 1
                ORDER BY priority
                """, (self.user_id,))
                channels_config = cursor.fetchall()
            else:
                # Pobierz tylko określone kanały
                placeholders = ','.join('?' for _ in channels)
                cursor = self.conn.execute(f"""
                SELECT channel_type, configuration
                FROM notification_channels
                WHERE user_id = ? AND channel_type IN ({placeholders}) AND is_active = 1
                ORDER BY priority
                """, (self.user_id, *channels))
                channels_config = cursor.fetchall()

            success = False
            for channel_type, config in channels_config:
                config = json.loads(config)
                if channel_type == 'email':
                    success |= self._send_email(message, config, severity)
                elif channel_type == 'telegram':
                    success |= self._send_telegram(message, config)
                elif channel_type == 'webhook':
                    success |= self._send_webhook(message, config)
                
            return success
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania powiadomienia: {e}")
            return False

    def _send_email(self, message: str, config: Dict, severity: str) -> bool:
        """Wysyła powiadomienie przez email."""
        try:
            msg = MIMEText(message)
            msg['Subject'] = f"Trading Alert [{severity.upper()}]"
            msg['From'] = config['from_email']
            msg['To'] = config['to_email']

            with smtplib.SMTP_SSL(config['smtp_server'], config['smtp_port']) as server:
                server.login(config['username'], config['password'])
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania emaila: {e}")
            return False

    def _send_telegram(self, message: str, config: Dict) -> bool:
        """Wysyła powiadomienie przez Telegram."""
        try:
            url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
            data = {
                'chat_id': config['chat_id'],
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=data)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania powiadomienia Telegram: {e}")
            return False

    def _send_webhook(self, message: str, config: Dict) -> bool:
        """Wysyła powiadomienie przez webhook."""
        try:
            headers = config.get('headers', {'Content-Type': 'application/json'})
            data = {
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'source': 'trading_bot'
            }
            response = requests.post(config['url'], json=data, headers=headers)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania webhooka: {e}")
            return False

    def get_active_alerts(self) -> Dict[str, List]:
        """Pobiera wszystkie aktywne alerty."""
        try:
            # Pobierz alerty wzorców cenowych
            cursor = self.conn.execute("""
            SELECT *
            FROM pattern_alerts
            WHERE user_id = ? AND status = 'active'
            ORDER BY timestamp DESC
            """, (self.user_id,))
            pattern_alerts = [dict(zip([col[0] for col in cursor.description], row))
                            for row in cursor.fetchall()]

            # Pobierz alerty anomalii AI
            cursor = self.conn.execute("""
            SELECT *
            FROM ai_anomaly_alerts
            WHERE user_id = ? AND resolution_status = 'pending'
            ORDER BY timestamp DESC
            """, (self.user_id,))
            ai_alerts = [dict(zip([col[0] for col in cursor.description], row))
                        for row in cursor.fetchall()]

            return {
                'pattern_alerts': pattern_alerts,
                'ai_alerts': ai_alerts
            }
        except Exception as e:
            logger.error(f"Błąd podczas pobierania aktywnych alertów: {e}")
            return {'pattern_alerts': [], 'ai_alerts': []}

    def get_channel_settings(self, user_id: Optional[int] = None) -> List[Dict]:
        """Pobiera ustawienia kanałów powiadomień dla użytkownika."""
        try:
            # Użyj przekazanego user_id lub domyślnego z instancji
            user_id = user_id or self.user_id
            
            if not user_id:
                logger.warning("Brak ID użytkownika do pobrania ustawień powiadomień")
                return []
                
            cursor = self.conn.execute("""
            SELECT id, channel_type, channel_name, configuration, priority, is_active
            FROM notification_channels
            WHERE user_id = ?
            ORDER BY priority
            """, (user_id,))
            
            settings = []
            for row in cursor.fetchall():
                channel_id, channel_type, name, config_json, priority, is_active = row
                settings.append({
                    'id': channel_id,
                    'channel_type': channel_type,
                    'name': name,
                    'config_json': config_json,
                    'priority': priority,
                    'is_active': is_active == 1
                })
            
            return settings
        except Exception as e:
            logger.error(f"Błąd podczas pobierania ustawień kanałów powiadomień: {e}")
            return []
            
    def update_channel_settings(self, user_id: Optional[int], channel_type: str, settings: Dict) -> bool:
        """Aktualizuje ustawienia kanału powiadomień."""
        try:
            # Użyj przekazanego user_id lub domyślnego z instancji
            user_id = user_id or self.user_id
            
            if not user_id:
                logger.warning("Brak ID użytkownika do aktualizacji ustawień")
                return False
                
            is_active = settings.get('is_active', True)
            config = settings.get('config', {})
            
            # Sprawdź, czy kanał istnieje
            cursor = self.conn.execute("""
            SELECT id FROM notification_channels
            WHERE user_id = ? AND channel_type = ?
            """, (user_id, channel_type))
            
            channel_id = cursor.fetchone()
            
            if channel_id:
                # Aktualizuj istniejący kanał
                self.conn.execute("""
                UPDATE notification_channels
                SET configuration = ?, is_active = ?
                WHERE id = ?
                """, (json.dumps(config), 1 if is_active else 0, channel_id[0]))
            else:
                # Dodaj nowy kanał
                self.conn.execute("""
                INSERT INTO notification_channels (user_id, channel_type, channel_name, configuration, is_active, priority)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    user_id, 
                    channel_type, 
                    settings.get('name', channel_type), 
                    json.dumps(config),
                    1 if is_active else 0,
                    settings.get('priority', 1)
                ))
                
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji ustawień kanału powiadomień: {e}")
            return False
            
    def get_notification_history(self, user_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
        """Pobiera historię powiadomień dla użytkownika."""
        try:
            # Użyj przekazanego user_id lub domyślnego z instancji
            user_id = user_id or self.user_id
            
            if not user_id:
                logger.warning("Brak ID użytkownika do pobrania historii powiadomień")
                return []
                
            cursor = self.conn.execute("""
            SELECT id, notification_type, channel, message, status, error_message, timestamp
            FROM notification_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """, (user_id, limit))
            
            history = []
            for row in cursor.fetchall():
                id, notification_type, channel, message, status, error_message, timestamp = row
                history.append({
                    'id': id,
                    'type': notification_type,
                    'channel': channel,
                    'message': message,
                    'status': status,
                    'error_message': error_message,
                    'timestamp': timestamp
                })
            
            return history
        except Exception as e:
            logger.error(f"Błąd podczas pobierania historii powiadomień: {e}")
            return []
    
    def create_notification_tables(self):
        """Tworzy tabele do obsługi powiadomień, jeśli nie istnieją."""
        try:
            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS notification_channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                channel_type TEXT NOT NULL,
                channel_name TEXT NOT NULL,
                configuration TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                priority INTEGER DEFAULT 1,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS notification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                notification_type TEXT NOT NULL,
                channel TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS pattern_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                confidence REAL,
                price_level REAL,
                volume_confirmation INTEGER DEFAULT 0,
                timeframe TEXT DEFAULT '1h',
                additional_indicators TEXT,
                status TEXT DEFAULT 'active',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS ai_anomaly_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                severity REAL NOT NULL,
                description TEXT,
                affected_assets TEXT,
                confidence_score REAL,
                false_positive_probability REAL DEFAULT 0.0,
                resolution_status TEXT DEFAULT 'pending',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            self.conn.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                condition TEXT,
                value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_triggered TIMESTAMP,
                active INTEGER DEFAULT 1
            )''')
            
            self.conn.commit()
            logger.info("Tabele powiadomień zostały utworzone lub już istnieją")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia tabel powiadomień: {e}")
            return False
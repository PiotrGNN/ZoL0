"""
System powiadomień dla alertów monitorowania
"""
import os
import json
import smtplib
import requests
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime

class NotificationManager:
    """Zarządza wysyłaniem powiadomień przez różne kanały."""
    
    def __init__(self, config_path: str = 'config/notifications.json'):
        self.logger = logging.getLogger('notification_manager')
        if not self.logger.handlers:
            handler = logging.FileHandler('logs/notifications.log')
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Wczytuje konfigurację powiadomień."""
        default_config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_address': '',
                'recipients': []
            },
            'slack': {
                'enabled': False,
                'webhook_url': '',
                'channel': '',
                'username': 'Monitoring Bot'
            },
            'webhook': {
                'enabled': False,
                'url': '',
                'headers': {},
                'method': 'POST'
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Łączymy z domyślną konfiguracją
                    for key, default_value in default_config.items():
                        if key not in config:
                            config[key] = default_value
                        elif isinstance(default_value, dict):
                            for sub_key, sub_value in default_value.items():
                                if sub_key not in config[key]:
                                    config[key][sub_key] = sub_value
                    return config
            else:
                # Tworzymy domyślny plik konfiguracyjny
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
                
        except Exception as e:
            self.logger.error(f"Błąd podczas wczytywania konfiguracji: {e}")
            return default_config
            
    def send_notification(self, 
                         title: str, 
                         message: str, 
                         level: str = 'info',
                         data: Optional[Dict] = None) -> bool:
        """
        Wysyła powiadomienie przez wszystkie skonfigurowane kanały.
        
        Args:
            title: Tytuł powiadomienia
            message: Treść powiadomienia
            level: Poziom ważności (info/warning/error/critical)
            data: Dodatkowe dane do załączenia w powiadomieniu
        
        Returns:
            bool: True jeśli wysłano przez przynajmniej jeden kanał
        """
        success = False
        
        try:
            if self.config['email']['enabled']:
                if self._send_email(title, message, level, data):
                    success = True
                    
            if self.config['slack']['enabled']:
                if self._send_slack(title, message, level, data):
                    success = True
                    
            if self.config['webhook']['enabled']:
                if self._send_webhook(title, message, level, data):
                    success = True
                    
            if success:
                self.logger.info(f"Wysłano powiadomienie: {title}")
            else:
                self.logger.warning(f"Nie udało się wysłać powiadomienia: {title}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Błąd podczas wysyłania powiadomienia: {e}")
            return False
            
    def _send_email(self, 
                    title: str, 
                    message: str, 
                    level: str,
                    data: Optional[Dict]) -> bool:
        """Wysyła powiadomienie przez email."""
        try:
            config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = config['from_address']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"[{level.upper()}] {title}"
            
            # Tworzenie treści HTML
            html = f"""
            <html>
              <body>
                <h2>{title}</h2>
                <p>{message}</p>
                {self._format_data_html(data) if data else ''}
                <hr>
                <p><small>Wysłano: {datetime.now().isoformat()}</small></p>
              </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            # Połączenie z serwerem SMTP
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['username'], config['password'])
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Błąd podczas wysyłania emaila: {e}")
            return False
            
    def _send_slack(self, 
                    title: str, 
                    message: str, 
                    level: str,
                    data: Optional[Dict]) -> bool:
        """Wysyła powiadomienie przez Slack."""
        try:
            config = self.config['slack']
            
            # Kolory dla różnych poziomów
            colors = {
                'info': '#36a64f',
                'warning': '#ffcc00',
                'error': '#ff9900',
                'critical': '#ff0000'
            }
            
            attachment = {
                'color': colors.get(level, '#36a64f'),
                'title': title,
                'text': message,
                'fields': self._format_data_slack(data) if data else [],
                'footer': f"Monitoring • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
            
            payload = {
                'channel': config['channel'],
                'username': config['username'],
                'attachments': [attachment]
            }
            
            response = requests.post(
                config['webhook_url'],
                json=payload
            )
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Błąd podczas wysyłania do Slack: {e}")
            return False
            
    def _send_webhook(self, 
                      title: str, 
                      message: str, 
                      level: str,
                      data: Optional[Dict]) -> bool:
        """Wysyła powiadomienie przez webhook."""
        try:
            config = self.config['webhook']
            
            payload = {
                'title': title,
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            response = requests.request(
                method=config['method'],
                url=config['url'],
                headers=config['headers'],
                json=payload
            )
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Błąd podczas wysyłania do webhook: {e}")
            return False
            
    def _format_data_html(self, data: Dict) -> str:
        """Formatuje dodatkowe dane do HTML."""
        if not data:
            return ''
            
        html = '<h3>Szczegóły:</h3><table border="1" cellpadding="5">'
        for key, value in data.items():
            html += f'<tr><th>{key}</th><td>{value}</td></tr>'
        html += '</table>'
        
        return html
        
    def _format_data_slack(self, data: Dict) -> List[Dict]:
        """Formatuje dodatkowe dane do formatu Slack."""
        if not data:
            return []
            
        return [
            {
                'title': key,
                'value': str(value),
                'short': len(str(value)) < 50
            }
            for key, value in data.items()
        ]
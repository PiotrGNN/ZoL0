"""
System alertów dla monitorowania API
"""
import logging
from typing import Dict, Any
import time
from datetime import datetime
import json
import smtplib
from email.mime.text import MIMEText
from config.api_config import get_metrics_config

class AlertManager:
    """Zarządza alertami na podstawie metryk API."""
    
    def __init__(self):
        self.logger = logging.getLogger('api_alerts')
        if not self.logger.handlers:
            handler = logging.FileHandler('logs/api_alerts.log')
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        self.thresholds = get_metrics_config()["alert_thresholds"]
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minut między kolejnymi alertami tego samego typu
        
    def check_metrics(self, metrics: Dict[str, Any], api_name: str):
        """Sprawdza metryki pod kątem przekroczenia progów."""
        current_time = time.time()
        
        # Sprawdzenie współczynnika błędów
        total_requests = metrics["requests"]["total"]
        if total_requests > 0:
            error_rate = metrics["requests"]["failed"] / total_requests
            if error_rate > self.thresholds["error_rate"]:
                self._send_alert(
                    "error_rate",
                    f"Wysoki współczynnik błędów dla {api_name}: {error_rate:.1%}",
                    current_time,
                    {"error_rate": error_rate, "total_requests": total_requests}
                )
        
        # Sprawdzenie czasów odpowiedzi
        if metrics["response_times"]:
            avg_response_time = sum(metrics["response_times"]) / len(metrics["response_times"])
            if avg_response_time > self.thresholds["response_time"]:
                self._send_alert(
                    "response_time",
                    f"Długi czas odpowiedzi dla {api_name}: {avg_response_time:.2f}s",
                    current_time,
                    {"avg_response_time": avg_response_time}
                )
        
        # Sprawdzenie przekroczeń rate limit
        rate_limit_hits = metrics["rate_limits"]["hits"]
        if rate_limit_hits > self.thresholds["rate_limit_hits"]:
            self._send_alert(
                "rate_limit",
                f"Częste przekroczenia rate limit dla {api_name}: {rate_limit_hits} w ostatnim okresie",
                current_time,
                {"rate_limit_hits": rate_limit_hits}
            )
    
    def _send_alert(self, alert_type: str, message: str, current_time: float, details: Dict[str, Any]):
        """Wysyła alert jeśli upłynął czas cooldown."""
        if alert_type in self.last_alert_time:
            if current_time - self.last_alert_time[alert_type] < self.alert_cooldown:
                return
                
        self.last_alert_time[alert_type] = current_time
        
        # Logowanie alertu
        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.fromtimestamp(current_time).isoformat(),
            "details": details
        }
        
        self.logger.warning(json.dumps(alert_data))
        
        # Zapisz alert do pliku
        try:
            with open(f"logs/alerts_{datetime.now().strftime('%Y%m%d')}.json", 'a') as f:
                json.dump(alert_data, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania alertu: {e}")
        
        # Można tu dodać więcej kanałów powiadomień (email, Slack, etc.)
        self._notify_slack(alert_data)
    
    def _notify_slack(self, alert_data: Dict[str, Any]):
        """
        Wysyła powiadomienie na Slack (przykładowa implementacja).
        Należy dodać właściwą konfigurację webhooka.
        """
        try:
            import requests
            webhook_url = "TWÓJ_WEBHOOK_URL"  # TODO: Przenieść do konfiguracji
            
            payload = {
                "text": f"*Alert API*\n{alert_data['message']}",
                "attachments": [{
                    "color": "danger",
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in alert_data["details"].items()
                    ]
                }]
            }
            
            requests.post(webhook_url, json=payload)
        except Exception as e:
            self.logger.error(f"Błąd podczas wysyłania powiadomienia Slack: {e}")
            
    def reset_cooldowns(self):
        """Resetuje czasy cooldown dla wszystkich typów alertów."""
        self.last_alert_time.clear()
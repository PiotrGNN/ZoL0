"""
Moduł do zbierania i analizy metryk wydajności API.
"""
import time
from typing import Dict, List, Any
import logging
import json
from datetime import datetime

class APIMetricsCollector:
    """Kolektor metryk wydajności API z zaawansowaną analityką."""
    
    def __init__(self, api_name: str):
        self.api_name = api_name
        self.metrics = {
            "requests": {
                "total": 0,
                "success": 0,
                "failed": 0,
                "by_endpoint": {}
            },
            "response_times": [],
            "errors": {
                "by_type": {},
                "by_endpoint": {}
            },
            "rate_limits": {
                "hits": 0,
                "last_hit": None,
                "recovery_times": []
            },
            "volume": {
                "bytes_sent": 0,
                "bytes_received": 0
            }
        }
        self.start_time = time.time()
        
        # Konfiguracja logowania
        self.logger = logging.getLogger(f"{api_name}_metrics")
        if not self.logger.handlers:
            handler = logging.FileHandler(f"logs/{api_name}_metrics.log")
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def record_request(self, endpoint: str, success: bool, response_time: float, 
                      error_type: str = None, response_size: int = 0, request_size: int = 0):
        """Rejestruje pojedyncze zapytanie API."""
        # Aktualizacja podstawowych metryk
        self.metrics["requests"]["total"] += 1
        if success:
            self.metrics["requests"]["success"] += 1
        else:
            self.metrics["requests"]["failed"] += 1
            
        # Aktualizacja metryk dla endpointu
        if endpoint not in self.metrics["requests"]["by_endpoint"]:
            self.metrics["requests"]["by_endpoint"][endpoint] = {
                "total": 0, "success": 0, "failed": 0, "response_times": []
            }
            
        endpoint_metrics = self.metrics["requests"]["by_endpoint"][endpoint]
        endpoint_metrics["total"] += 1
        if success:
            endpoint_metrics["success"] += 1
        else:
            endpoint_metrics["failed"] += 1
            
        # Zapisywanie czasów odpowiedzi
        self.metrics["response_times"].append(response_time)
        endpoint_metrics["response_times"].append(response_time)
        
        # Aktualizacja metryk błędów
        if not success and error_type:
            if error_type not in self.metrics["errors"]["by_type"]:
                self.metrics["errors"]["by_type"][error_type] = 0
            self.metrics["errors"]["by_type"][error_type] += 1
            
            if endpoint not in self.metrics["errors"]["by_endpoint"]:
                self.metrics["errors"]["by_endpoint"][endpoint] = {}
            if error_type not in self.metrics["errors"]["by_endpoint"][endpoint]:
                self.metrics["errors"]["by_endpoint"][endpoint][error_type] = 0
            self.metrics["errors"]["by_endpoint"][endpoint][error_type] += 1
            
        # Aktualizacja metryk wielkości danych
        self.metrics["volume"]["bytes_sent"] += request_size
        self.metrics["volume"]["bytes_received"] += response_size
        
    def record_rate_limit_hit(self):
        """Rejestruje wystąpienie przekroczenia limitu zapytań."""
        self.metrics["rate_limits"]["hits"] += 1
        current_time = time.time()
        
        if self.metrics["rate_limits"]["last_hit"]:
            recovery_time = current_time - self.metrics["rate_limits"]["last_hit"]
            self.metrics["rate_limits"]["recovery_times"].append(recovery_time)
            
        self.metrics["rate_limits"]["last_hit"] = current_time
        
    def get_analysis(self) -> Dict[str, Any]:
        """Generuje szczegółową analizę zebranych metryk."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Obliczanie statystyk czasów odpowiedzi
        response_times = self.metrics["response_times"]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        sorted_times = sorted(response_times)
        p95_response_time = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        p99_response_time = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
        
        # Obliczanie success rate
        total_requests = self.metrics["requests"]["total"]
        success_rate = (self.metrics["requests"]["success"] / total_requests * 100) if total_requests > 0 else 0
        
        # Analiza endpointów
        endpoint_analysis = {}
        for endpoint, data in self.metrics["requests"]["by_endpoint"].items():
            endpoint_success_rate = (data["success"] / data["total"] * 100) if data["total"] > 0 else 0
            endpoint_avg_response_time = sum(data["response_times"]) / len(data["response_times"]) if data["response_times"] else 0
            
            endpoint_analysis[endpoint] = {
                "success_rate": f"{endpoint_success_rate:.1f}%",
                "avg_response_time": f"{endpoint_avg_response_time:.3f}s",
                "total_requests": data["total"],
                "errors": self.metrics["errors"]["by_endpoint"].get(endpoint, {})
            }
        
        # Analiza rate limitów
        avg_recovery_time = sum(self.metrics["rate_limits"]["recovery_times"]) / len(self.metrics["rate_limits"]["recovery_times"]) if self.metrics["rate_limits"]["recovery_times"] else 0
        
        return {
            "general": {
                "uptime": f"{uptime:.1f}s",
                "total_requests": total_requests,
                "success_rate": f"{success_rate:.1f}%",
                "avg_response_time": f"{avg_response_time:.3f}s",
                "p95_response_time": f"{p95_response_time:.3f}s",
                "p99_response_time": f"{p99_response_time:.3f}s"
            },
            "endpoints": endpoint_analysis,
            "rate_limits": {
                "total_hits": self.metrics["rate_limits"]["hits"],
                "avg_recovery_time": f"{avg_recovery_time:.1f}s"
            },
            "volume": {
                "sent_mb": self.metrics["volume"]["bytes_sent"] / 1024 / 1024,
                "received_mb": self.metrics["volume"]["bytes_received"] / 1024 / 1024
            },
            "errors": dict(self.metrics["errors"]["by_type"])
        }
        
    def save_report(self, file_path: str = None):
        """Zapisuje raport z metrykami do pliku JSON."""
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"reports/{self.api_name}_metrics_{timestamp}.json"
            
        analysis = self.get_analysis()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=4)
            self.logger.info(f"Zapisano raport metryk do {file_path}")
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania raportu: {e}")
            
    def reset(self):
        """Resetuje wszystkie zebrane metryki."""
        self.__init__(self.api_name)
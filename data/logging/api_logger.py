
import logging
import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Tworzenie katalogu dla logów, jeśli nie istnieje
os.makedirs("logs", exist_ok=True)

# Konfiguracja loggera
api_logger = logging.getLogger("api")
api_logger.setLevel(logging.INFO)

# Handler dla pliku
file_handler = logging.FileHandler("logs/api_requests.log")
file_handler.setLevel(logging.INFO)

# Handler dla konsoli
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Format
formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Dodawanie handlerów
api_logger.addHandler(file_handler)
api_logger.addHandler(console_handler)

class ApiLogger:
    """
    Klasa do logowania zapytań API z rate limiting,
    statystykami i alertami o błędach
    """
    
    def __init__(self, exchange_name: str, max_errors_before_alert: int = 5):
        """
        Inicjalizacja loggera API
        
        Args:
            exchange_name (str): Nazwa giełdy/API
            max_errors_before_alert (int): Liczba błędów przed alertem
        """
        self.exchange_name = exchange_name
        self.logger = api_logger.getChild(exchange_name)
        
        # Statystyki
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'requests_per_endpoint': {},
            'errors_per_code': {},
            'last_errors': []
        }
        
        # Rate limiting
        self.requests_timestamps = []
        self.errors_timestamps = []
        self.max_errors_before_alert = max_errors_before_alert
        
        # Mutex dla thread-safety
        self.stats_lock = threading.Lock()
        
    def log_request(self, endpoint: str, method: str, params: Dict = None,
                   response: Dict = None, response_time: float = None, 
                   error: str = None, error_code: str = None) -> None:
        """
        Loguje zapytanie API
        
        Args:
            endpoint (str): Endpoint API
            method (str): Metoda HTTP (GET, POST, DELETE)
            params (Dict, optional): Parametry zapytania
            response (Dict, optional): Odpowiedź API
            response_time (float, optional): Czas odpowiedzi w sekundach
            error (str, optional): Komunikat błędu
            error_code (str, optional): Kod błędu
        """
        with self.stats_lock:
            timestamp = datetime.now()
            self.stats['total_requests'] += 1
            
            # Dodaj endpoint do statystyk
            if endpoint not in self.stats['requests_per_endpoint']:
                self.stats['requests_per_endpoint'][endpoint] = 0
            self.stats['requests_per_endpoint'][endpoint] += 1
            
            # Zapisz timestamp zapytania
            self.requests_timestamps.append(time.time())
            
            # Sukces lub błąd
            if error is None:
                self.stats['successful_requests'] += 1
                
                # Aktualizuj średni czas odpowiedzi
                if response_time is not None:
                    current_avg = self.stats['avg_response_time']
                    current_total = current_avg * (self.stats['successful_requests'] - 1)
                    self.stats['avg_response_time'] = (current_total + response_time) / self.stats['successful_requests']
                
                log_msg = (f"{method} {endpoint} - Sukces "
                          f"({response_time:.3f}s)" if response_time else f"{method} {endpoint} - Sukces")
                self.logger.info(log_msg)
            else:
                self.stats['failed_requests'] += 1
                
                # Zapisz timestamp błędu
                self.errors_timestamps.append(time.time())
                
                # Dodaj kod błędu do statystyk
                if error_code not in self.stats['errors_per_code']:
                    self.stats['errors_per_code'][error_code] = 0
                self.stats['errors_per_code'][error_code] += 1
                
                # Zapisz błąd w historii
                error_entry = {
                    'timestamp': timestamp,
                    'endpoint': endpoint,
                    'method': method,
                    'error': error,
                    'error_code': error_code
                }
                self.stats['last_errors'].append(error_entry)
                
                # Zachowaj tylko ostatnie 100 błędów
                if len(self.stats['last_errors']) > 100:
                    self.stats['last_errors'] = self.stats['last_errors'][-100:]
                
                log_msg = f"{method} {endpoint} - BŁĄD: {error}"
                if error_code:
                    log_msg += f" (kod: {error_code})"
                self.logger.error(log_msg)
                
                # Sprawdź, czy trzeba wygenerować alert
                self._check_error_alert()
    
    def _check_error_alert(self) -> None:
        """Sprawdza, czy należy wygenerować alert o błędach"""
        # Filtruj błędy z ostatnich 5 minut
        recent_errors = [t for t in self.errors_timestamps if time.time() - t < 300]
        
        if len(recent_errors) >= self.max_errors_before_alert:
            self.logger.critical(
                f"ALERT: {len(recent_errors)} błędów API w ciągu ostatnich 5 minut! "
                f"Sprawdź logi i status API {self.exchange_name}."
            )
    
    def get_request_rate(self, time_window: int = 60) -> int:
        """
        Zwraca liczbę zapytań w danym oknie czasowym
        
        Args:
            time_window (int): Okno czasowe w sekundach
            
        Returns:
            int: Liczba zapytań w oknie czasowym
        """
        with self.stats_lock:
            # Filtruj zapytania z danego okna czasowego
            current_time = time.time()
            recent_requests = [t for t in self.requests_timestamps if current_time - t < time_window]
            return len(recent_requests)
    
    def get_error_rate(self, time_window: int = 300) -> float:
        """
        Zwraca współczynnik błędów w danym oknie czasowym
        
        Args:
            time_window (int): Okno czasowe w sekundach
            
        Returns:
            float: Współczynnik błędów (0.0 - 1.0)
        """
        with self.stats_lock:
            # Filtruj zapytania i błędy z danego okna czasowego
            current_time = time.time()
            recent_requests = [t for t in self.requests_timestamps if current_time - t < time_window]
            recent_errors = [t for t in self.errors_timestamps if current_time - t < time_window]
            
            if not recent_requests:
                return 0.0
                
            return len(recent_errors) / len(recent_requests)
    
    def get_stats(self) -> Dict:
        """
        Zwraca statystyki API
        
        Returns:
            Dict: Statystyki API
        """
        with self.stats_lock:
            stats_copy = self.stats.copy()
            
            # Dodaj aktualne współczynniki
            stats_copy['request_rate_1min'] = self.get_request_rate(60)
            stats_copy['request_rate_5min'] = self.get_request_rate(300)
            stats_copy['error_rate_5min'] = self.get_error_rate(300)
            
            return stats_copy
    
    def clear_stats(self) -> None:
        """Czyści statystyki API"""
        with self.stats_lock:
            self.stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_response_time': 0,
                'requests_per_endpoint': {},
                'errors_per_code': {},
                'last_errors': []
            }
            self.requests_timestamps = []
            self.errors_timestamps = []
            
            self.logger.info("Statystyki API wyczyszczone")

# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja loggera
    bybit_logger = ApiLogger(exchange_name="bybit")
    
    # Przykładowe logi
    bybit_logger.log_request(
        endpoint="/v5/market/tickers",
        method="GET",
        params={"category": "spot", "symbol": "BTCUSDT"},
        response_time=0.234
    )
    
    # Przykładowy błąd
    bybit_logger.log_request(
        endpoint="/v5/order/create",
        method="POST",
        params={"category": "spot", "symbol": "BTCUSDT", "side": "Buy"},
        error="Invalid quantity",
        error_code="10001"
    )
    
    # Wyświetl statystyki
    print(json.dumps(bybit_logger.get_stats(), indent=2))

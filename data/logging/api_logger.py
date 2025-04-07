
"""
API Logger - moduł odpowiedzialny za logowanie ruchu API
i analitykę komunikacji z giełdami.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Konfiguracja logowania
api_logger = logging.getLogger("APILogger")
api_logger.setLevel(logging.INFO)

# Upewnij się, że istnieje katalog logs
os.makedirs("logs", exist_ok=True)

# Dodaj obsługę pliku
file_handler = logging.FileHandler("logs/api_requests.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
api_logger.addHandler(file_handler)

# Dodaj obsługę konsoli
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
api_logger.addHandler(console_handler)


class APILogger:
    """
    Klasa odpowiedzialna za logowanie ruchu API i analitykę komunikacji z giełdami.
    
    Funkcjonalności:
    - Logowanie żądań i odpowiedzi API
    - Śledzenie limitów żądań (rate limits)
    - Analiza czasów odpowiedzi
    - Wykrywanie wzorców błędów
    """

    def __init__(
        self,
        log_to_file: bool = True,
        log_responses: bool = True,
        log_sensitive_data: bool = False,
        max_response_log_length: int = 500,
    ):
        """
        Inicjalizacja APILoggera.
        
        Args:
            log_to_file: Czy logować do pliku
            log_responses: Czy logować pełne odpowiedzi API
            log_sensitive_data: Czy logować wrażliwe dane (np. klucze API, hasła)
            max_response_log_length: Maksymalna długość logowanej odpowiedzi
        """
        self.log_to_file = log_to_file
        self.log_responses = log_responses
        self.log_sensitive_data = log_sensitive_data
        self.max_response_log_length = max_response_log_length
        
        # Historia żądań API
        self.request_history = []
        
        # Statystyki API
        self.api_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0,
            "rate_limit_hits": 0,
            "errors_by_code": {},
        }
        
        api_logger.info("APILogger zainicjalizowany")

    def log_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        exchange: str = "unknown",
    ) -> Dict:
        """
        Loguje żądanie API.
        
        Args:
            method: Metoda HTTP
            endpoint: Endpoint API
            params: Parametry URL
            data: Dane JSON
            headers: Nagłówki HTTP
            exchange: Nazwa giełdy
            
        Returns:
            Dict: Dane żądania z ID i czasem rozpoczęcia
        """
        # Maskowanie wrażliwych danych
        safe_headers = self._mask_sensitive_data(headers) if headers else {}
        
        # Tworzenie rekordu żądania
        request_data = {
            "id": f"{exchange}-{int(time.time() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "exchange": exchange,
            "method": method,
            "endpoint": endpoint,
            "params": params,
            "data": data,
            "headers": safe_headers,
            "start_time": time.time(),
        }
        
        # Logowanie
        log_message = f"API Request: {method} {endpoint}"
        if params:
            log_message += f", Params: {json.dumps(params)}"
        if data:
            log_message += f", Data: {json.dumps(data)[:self.max_response_log_length]}"
            
        api_logger.info(log_message)
        
        # Aktualizacja statystyk
        self.api_stats["total_requests"] += 1
        
        return request_data

    def log_response(
        self,
        request_data: Dict,
        status_code: int,
        response_data: Any,
        response_time: Optional[float] = None,
    ) -> None:
        """
        Loguje odpowiedź API.
        
        Args:
            request_data: Dane żądania zwrócone przez log_request
            status_code: Kod statusu HTTP
            response_data: Dane odpowiedzi
            response_time: Czas odpowiedzi w sekundach (opcjonalnie)
        """
        # Oblicz czas odpowiedzi, jeśli nie podano
        if response_time is None and "start_time" in request_data:
            response_time = time.time() - request_data["start_time"]
            
        # Aktualizuj rekord żądania o odpowiedź
        request_data.update({
            "status_code": status_code,
            "response_time": response_time,
            "end_time": time.time(),
        })
        
        # Zapisz pełną odpowiedź, jeśli włączono tę opcję
        if self.log_responses:
            if isinstance(response_data, dict) or isinstance(response_data, list):
                try:
                    response_str = json.dumps(response_data)
                    if len(response_str) > self.max_response_log_length:
                        response_str = response_str[:self.max_response_log_length] + "..."
                    request_data["response"] = response_str
                except Exception:
                    request_data["response"] = str(response_data)[:self.max_response_log_length]
            else:
                request_data["response"] = str(response_data)[:self.max_response_log_length]
                
        # Dodaj do historii
        self.request_history.append(request_data)
        
        # Ograniczenie historii do 1000 ostatnich rekordów
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
            
        # Aktualizacja statystyk
        if 200 <= status_code < 300:
            self.api_stats["successful_requests"] += 1
        else:
            self.api_stats["failed_requests"] += 1
            
            # Klasyfikacja błędów
            error_key = f"{status_code}"
            if error_key in self.api_stats["errors_by_code"]:
                self.api_stats["errors_by_code"][error_key] += 1
            else:
                self.api_stats["errors_by_code"][error_key] = 1
                
        # Aktualizuj średni czas odpowiedzi
        total_requests = self.api_stats["successful_requests"] + self.api_stats["failed_requests"]
        if total_requests > 0:
            self.api_stats["avg_response_time"] = (
                (self.api_stats["avg_response_time"] * (total_requests - 1) + response_time) / total_requests
            )
            
        # Wykrywanie przekroczenia limitów
        if status_code == 429:
            self.api_stats["rate_limit_hits"] += 1
            api_logger.warning(f"Rate limit hit for {request_data['exchange']}: {request_data['endpoint']}")
            
        # Logowanie
        log_message = (
            f"API Response: {request_data['method']} {request_data['endpoint']} "
            f"Status: {status_code}, Time: {response_time:.3f}s"
        )
        
        if status_code >= 400:
            api_logger.error(log_message)
            if self.log_responses:
                api_logger.error(f"Error details: {request_data.get('response', 'No details')}")
        else:
            api_logger.info(log_message)

    def log_error(
        self,
        request_data: Dict,
        error: Exception,
        retry_attempt: Optional[int] = None,
    ) -> None:
        """
        Loguje błąd podczas wykonywania żądania API.
        
        Args:
            request_data: Dane żądania zwrócone przez log_request
            error: Wyjątek
            retry_attempt: Numer próby ponowienia (opcjonalnie)
        """
        # Oblicz czas błędu
        error_time = time.time() - request_data.get("start_time", time.time())
        
        # Aktualizuj rekord żądania o błąd
        request_data.update({
            "error": str(error),
            "error_type": type(error).__name__,
            "response_time": error_time,
            "end_time": time.time(),
        })
        
        if retry_attempt is not None:
            request_data["retry_attempt"] = retry_attempt
            
        # Dodaj do historii
        self.request_history.append(request_data)
        
        # Ograniczenie historii do 1000 ostatnich rekordów
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
            
        # Aktualizacja statystyk
        self.api_stats["failed_requests"] += 1
        
        # Klasyfikacja błędów
        error_key = type(error).__name__
        if error_key in self.api_stats["errors_by_code"]:
            self.api_stats["errors_by_code"][error_key] += 1
        else:
            self.api_stats["errors_by_code"][error_key] = 1
            
        # Logowanie
        retry_info = f", Retry: {retry_attempt}" if retry_attempt is not None else ""
        log_message = (
            f"API Error: {request_data['method']} {request_data['endpoint']} "
            f"Error: {type(error).__name__}: {str(error)}, Time: {error_time:.3f}s{retry_info}"
        )
        
        api_logger.error(log_message)

    def log_websocket_event(
        self,
        event_type: str,
        data: Any,
        exchange: str = "unknown",
    ) -> None:
        """
        Loguje zdarzenie WebSocket.
        
        Args:
            event_type: Typ zdarzenia (np. 'open', 'message', 'error', 'close')
            data: Dane zdarzenia
            exchange: Nazwa giełdy
        """
        # Maskowanie wrażliwych danych
        safe_data = data
        if isinstance(data, dict) and not self.log_sensitive_data:
            safe_data = self._mask_sensitive_data(data)
            
        # Logowanie
        timestamp = datetime.now().isoformat()
        
        if event_type == "open":
            api_logger.info(f"WebSocket({exchange}) Connected")
        elif event_type == "close":
            code = data.get("code", "unknown")
            reason = data.get("reason", "unknown")
            api_logger.info(f"WebSocket({exchange}) Closed: {code} - {reason}")
        elif event_type == "error":
            api_logger.error(f"WebSocket({exchange}) Error: {safe_data}")
        elif event_type == "message":
            if isinstance(safe_data, dict) or isinstance(safe_data, list):
                try:
                    data_str = json.dumps(safe_data)
                    if len(data_str) > self.max_response_log_length:
                        data_str = data_str[:self.max_response_log_length] + "..."
                except Exception:
                    data_str = str(safe_data)[:self.max_response_log_length]
            else:
                data_str = str(safe_data)[:self.max_response_log_length]
                
            api_logger.debug(f"WebSocket({exchange}) Message: {data_str}")
        else:
            api_logger.info(f"WebSocket({exchange}) {event_type}: {safe_data}")
            
        # Przykładowy rekord zdarzenia (możesz go zapisać do historii)
        event_record = {
            "timestamp": timestamp,
            "exchange": exchange,
            "event_type": event_type,
            "data": safe_data if self.log_responses else None,
        }

    def get_stats(self) -> Dict:
        """
        Zwraca statystyki API.
        
        Returns:
            Dict: Statystyki API
        """
        return self.api_stats

    def get_recent_errors(self, limit: int = 10) -> List[Dict]:
        """
        Zwraca ostatnie błędy API.
        
        Args:
            limit: Maksymalna liczba błędów do zwrócenia
            
        Returns:
            List[Dict]: Lista ostatnich błędów
        """
        errors = [
            req for req in self.request_history
            if "error" in req or ("status_code" in req and req["status_code"] >= 400)
        ]
        return errors[-limit:]

    def _mask_sensitive_data(self, data: Dict) -> Dict:
        """
        Maskuje wrażliwe dane w słowniku.
        
        Args:
            data: Słownik do zamaskowania
            
        Returns:
            Dict: Zamaskowany słownik
        """
        if not isinstance(data, dict):
            return data
            
        sensitive_keys = [
            "api_key", "apikey", "key", "secret", "password", "token", "auth",
            "signature", "sign", "X-MBX-APIKEY", "X-BAPI-API-KEY", "X-BAPI-SIGN",
        ]
        
        masked_data = {}
        for key, value in data.items():
            if any(s_key in key.lower() for s_key in sensitive_keys):
                masked_data[key] = "********"
            elif isinstance(value, dict):
                masked_data[key] = self._mask_sensitive_data(value)
            else:
                masked_data[key] = value
                
        return masked_data


# Globalny singleton APILogger
api_logger_instance = APILogger()


# Przykład użycia
if __name__ == "__main__":
    import requests
    
    # Używanie loggera
    logger = APILogger()
    
    # Przykład logowania żądania i odpowiedzi
    try:
        # Symulacja żądania
        req_data = logger.log_request(
            method="GET",
            endpoint="/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
            exchange="binance",
        )
        
        # Wykonanie faktycznego żądania (dla demonstracji)
        start = time.time()
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
        )
        response_time = time.time() - start
        
        # Logowanie odpowiedzi
        logger.log_response(
            request_data=req_data,
            status_code=response.status_code,
            response_data=response.json(),
            response_time=response_time,
        )
        
        # Przykład logowania błędu
        req_data = logger.log_request(
            method="GET",
            endpoint="/api/v3/nonexistent",
            exchange="binance",
        )
        
        try:
            # Wykonanie żądania, które zawiedzie
            start = time.time()
            response = requests.get("https://api.binance.com/api/v3/nonexistent")
            response_time = time.time() - start
            
            logger.log_response(
                request_data=req_data,
                status_code=response.status_code,
                response_data=response.text,
                response_time=response_time,
            )
        except Exception as e:
            logger.log_error(
                request_data=req_data,
                error=e,
            )
            
        # Przykład logowania zdarzeń WebSocket
        logger.log_websocket_event(
            event_type="open",
            data={},
            exchange="binance",
        )
        
        logger.log_websocket_event(
            event_type="message",
            data={"e": "trade", "p": "50000.00", "q": "0.01", "T": 1631234567890},
            exchange="binance",
        )
        
        logger.log_websocket_event(
            event_type="close",
            data={"code": 1000, "reason": "Normal closure"},
            exchange="binance",
        )
        
        # Wyświetlenie statystyk
        print("\nAPI Stats:")
        for key, value in logger.get_stats().items():
            print(f"{key}: {value}")
            
        # Wyświetlenie ostatnich błędów
        errors = logger.get_recent_errors()
        if errors:
            print("\nRecent Errors:")
            for error in errors:
                print(f"- {error.get('method')} {error.get('endpoint')}: {error.get('error', error.get('status_code'))}")
                
    except Exception as e:
        print(f"Error during demo: {e}")

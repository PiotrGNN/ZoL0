"""
Dekoratory do monitorowania API
"""
import time
import functools
from typing import Callable, Any
from utils.api_metrics import APIMetricsCollector
from utils.api_alerts import AlertManager

def collect_metrics(api_name: str, endpoint: str):
    """
    Dekorator zbierający metryki dla endpoint'ów API.
    
    Przykład użycia:
    @collect_metrics('bybit', '/v5/market/orderbook')
    def get_orderbook(self, symbol: str):
        ...
    """
    metrics_collector = APIMetricsCollector(api_name)
    alert_manager = AlertManager()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            request_size = sum(len(str(arg).encode()) for arg in args)
            request_size += sum(len(str(k).encode()) + len(str(v).encode()) for k, v in kwargs.items())
            
            try:
                result = func(*args, **kwargs)
                success = True
                error_type = None
                response_size = len(str(result).encode()) if result else 0
            except Exception as e:
                success = False
                error_type = type(e).__name__
                response_size = len(str(e).encode())
                raise
            finally:
                response_time = time.time() - start_time
                
                # Zbieranie metryk
                metrics_collector.record_request(
                    endpoint=endpoint,
                    success=success,
                    response_time=response_time,
                    error_type=error_type,
                    response_size=response_size,
                    request_size=request_size
                )
                
                # Sprawdzenie alertów
                alert_manager.check_metrics(metrics_collector.metrics, api_name)
                
            return result
        return wrapper
    return decorator

def rate_limit_safe(max_retries: int = 3, delay: float = 1.0):
    """
    Dekorator zabezpieczający przed przekroczeniem limitów API.
    
    Przykład użycia:
    @rate_limit_safe(max_retries=3, delay=1.0)
    @collect_metrics('bybit', '/v5/market/orderbook')
    def get_orderbook(self, symbol: str):
        ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        last_error = e
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))  # Exponential backoff
                            continue
                    raise
                    
            if last_error:
                raise last_error
                
        return wrapper
    return decorator
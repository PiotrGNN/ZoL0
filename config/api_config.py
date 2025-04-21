"""
Konfiguracja dla API i systemu metryk
"""
from typing import Dict, Any

# Podstawowa konfiguracja API
API_CONFIG = {
    "bybit": {
        "base_url": "https://api.bybit.com",
        "timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "rate_limit": {
            "requests_per_minute": 600,
            "order_rate_limit": 100
        }
    }
}

# Konfiguracja monitorowania
METRICS_CONFIG = {
    "enabled": True,
    "save_interval": 3600,  # Zapisuj metryki co godzinę
    "retention_days": 7,    # Przechowuj metryki przez 7 dni
    "alert_thresholds": {
        "error_rate": 0.05,  # Alert przy 5% błędów
        "response_time": 2.0, # Alert przy czasie odpowiedzi > 2s
        "rate_limit_hits": 10 # Alert przy 10 przekroczeniach limitu
    }
}

def get_api_config(api_name: str) -> Dict[str, Any]:
    """Pobiera konfigurację dla danego API."""
    if api_name not in API_CONFIG:
        raise ValueError(f"Brak konfiguracji dla API: {api_name}")
    return API_CONFIG[api_name]

def get_metrics_config() -> Dict[str, Any]:
    """Pobiera konfigurację systemu metryk."""
    return METRICS_CONFIG
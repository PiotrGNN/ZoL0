"""
cache_manager.py - Moduł zarządzający buforowaniem danych API i śledzeniem limitów zapytań
"""

import os
import json
import time
import logging
from typing import Any, Dict, Tuple, Optional, List, Union
from datetime import datetime

# Konfiguracja logowania
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.FileHandler("logs/cache_manager.log")
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Katalog przechowywania cache
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Parametry dla różnych środowisk
ENVIRONMENT_CONFIG = {
    'production': {
        'min_interval': 20.0,       # Jeszcze bardziej konserwatywny odstęp między zapytaniami (20000ms)
        'max_calls_per_minute': 3,  # Radykalnie zmniejszony limit dla produkcji
        'cache_ttl_multiplier': 15.0,# Ekstremalnie długi czas cache'owania dla produkcji
        'server_time_ttl': 1800,     # 30 minut cache'owania czasu serwera (by uniknąć odpytywania)
        'rate_limit_reset_time': 1800# 30 minut po przekroczeniu limitu zanim spróbujemy ponownie
    },
    'development': {
        'min_interval': 5.0,        # Zwiększony minimalny odstęp między zapytaniami (5000ms)
        'max_calls_per_minute': 10, # Konserwatywny limit dla środowiska deweloperskiego
        'cache_ttl_multiplier': 4.0,# Dłuższy czas cache'owania
        'server_time_ttl': 300,     # 5 minut cache'owania czasu serwera
        'rate_limit_reset_time': 300# 5 minut po przekroczeniu limitu zanim spróbujemy ponownie
    }
}

# Sprawdź, czy używamy produkcyjnego API na podstawie zmiennych środowiskowych
_is_production = os.getenv('BYBIT_USE_TESTNET', 'true').lower() != 'true' or os.getenv('IS_PRODUCTION', 'false').lower() == 'true'
_env_config = ENVIRONMENT_CONFIG['production'] if _is_production else ENVIRONMENT_CONFIG['development']

# Funkcja do wykrywania błędów CloudFront
def detect_cloudfront_error(error_message: str) -> bool:
    """
    Wykrywa błędy związane z CloudFront i limitami IP w komunikatach błędów.

    Args:
        error_message: Wiadomość błędu do analizy

    Returns:
        bool: True jeśli wykryto błąd CloudFront/IP limit, False w przeciwnym razie
    """
    error_lower = error_message.lower()
    cloudfront_indicators = [
        'cloudfront', 
        'distribution', 
        '403 forbidden',
        'rate limit',
        '429 too many requests',
        'access denied',
        'ip address has been blocked',
        'throttled',
        'operation too frequent',
        'too many requests'
    ]

    for indicator in cloudfront_indicators:
        if indicator in error_lower:
            logger.warning(f"Wykryto błąd CloudFront/IP limit: '{indicator}' w '{error_message}'")
            return True

    return False

def get_cached_data(key: str) -> Tuple[Any, bool]:
    """
    Pobiera dane z cache'a.

    Args:
        key: Klucz pod którym zapisano dane

    Returns:
        Tuple zawierający (dane, czy_znaleziono)
    """
    # Sprawdź cache na dysku
    file_path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data, True
        except Exception as e:
            logger.warning(f"Błąd podczas odczytu cache z dysku dla klucza {key}: {e}")

    return None, False

def store_cached_data(key: str, data: Any) -> None:
    """
    Zapisuje dane w cache'u.

    Args:
        key: Klucz pod którym zapisać dane
        data: Dane do zapisania
    """
    try:
        file_path = os.path.join(CACHE_DIR, f"{key}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"Błąd podczas zapisu cache na dysk dla klucza {key}: {e}")

def is_cache_valid(key: str, ttl: int = 300) -> bool:
    """
    Sprawdza czy cache jest wciąż ważny.

    Args:
        key: Klucz cache'a
        ttl: Czas życia w sekundach (domyślnie 5 minut)

    Returns:
        bool: True jeśli cache jest ważny, False w przeciwnym wypadku
    """
    file_path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(file_path):
        return False

    current_time = time.time()
    last_update = os.path.getmtime(file_path)

    return current_time - last_update < ttl

def clear_cache(key: str = None) -> None:
    """
    Czyści cache.

    Args:
        key: Opcjonalny klucz. Jeśli None, czyści cały cache.
    """
    if key is None:
        # Usuń pliki cache z dysku
        for file_name in os.listdir(CACHE_DIR):
            if file_name.endswith('.json'):
                try:
                    os.remove(os.path.join(CACHE_DIR, file_name))
                except Exception as e:
                    logger.warning(f"Nie można usunąć pliku cache {file_name}: {e}")
    else:
        # Usuń plik cache z dysku
        file_path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Nie można usunąć pliku cache {key}: {e}")

def get_api_status() -> Dict[str, Any]:
    """
    Zwraca aktualny status API i parametry limitera.

    Returns:
        Dict[str, Any]: Status API
    """
    # Sprawdź czy jesteśmy w stanie rate_limit
    rate_limited, found = get_cached_data("api_rate_limited")
    is_limited = found and rate_limited

    # Sprawdź status blokady CloudFront
    cloudfront_status = get_cloudfront_status()

    return {
        "rate_limited": is_limited,
        "calls_left": 5 if not is_limited else 0,
        "environment": 'production' if _is_production else 'development',
        "cloudfront_blocked": cloudfront_status.get("blocked", False)
    }

def set_cloudfront_block_status(blocked: bool, error_message: str = ""):
    """
    Ustawia status blokady CloudFront w cache.

    Args:
        blocked (bool): Czy blokada jest aktywna
        error_message (str): Opcjonalny komunikat błędu
    """
    cloudfront_status = {
        "blocked": blocked,
        "timestamp": time.time(),
        "error": error_message
    }
    store_cached_data("cloudfront_status", cloudfront_status)
    logger.warning(f"Status blokady CloudFront ustawiony na: {blocked}")

def get_cloudfront_status() -> Dict[str, Any]:
    """
    Pobiera aktualny status blokady CloudFront.

    Returns:
        Dict[str, Any]: Status blokady CloudFront
    """
    status, found = get_cached_data("cloudfront_status")
    if not found:
        status = {"blocked": False, "timestamp": 0, "error": ""}

    # Jeśli blokada trwa dłużej niż 60 minut, automatycznie ją resetujemy
    if status.get("blocked", False) and time.time() - status.get("timestamp", 0) > 3600:
        status["blocked"] = False
        status["error"] = "Automatyczny reset blokady po 60 minutach"
        store_cached_data("cloudfront_status", status)
        logger.info("Automatyczny reset blokady CloudFront po 60 minutach")

    return status

# Funkcje do obsługi blokady CloudFront (zduplikowane w oryginale, usunięte)


# Automatyczna inicjalizacja przy imporcie
def init_cache():
    """Inicjalizuje cache i ustawia domyślne parametry."""
    # Ustaw konserwatywne limity dla Bybit API
    set_rate_limit_parameters(max_calls_per_minute=6, min_interval=10.0)

    # Ustaw w cache flagę rate_limit na False
    store_cached_data("api_rate_limited", False)

    logger.info("Cache zainicjalizowany z domyślnymi parametrami")

def set_rate_limit_parameters(max_calls_per_minute: int = None, min_interval: float = None):
    """
    Ustawia parametry rate limitera.

    Args:
        max_calls_per_minute: Maksymalna liczba zapytań na minutę
        min_interval: Minimalny odstęp między zapytaniami w sekundach
    """
    if max_calls_per_minute is not None:
        _env_config["max_calls_per_minute"] = max_calls_per_minute

    if min_interval is not None:
        _env_config["min_interval"] = min_interval

    logger.info(f"Ustawiono parametry rate limitera: max_calls_per_minute={_env_config['max_calls_per_minute']}, min_interval={_env_config['min_interval']}s")

init_cache()


if __name__ == "__main__":
    # Przykład użycia
    @cache_with_ttl(ttl=10)
    def slow_function(param):
        logger.info(f"Wywołanie slow_function({param})")
        time.sleep(2)  # Symulacja długiego przetwarzania
        return f"Wynik {param}"

    @rate_limit
    def api_function(param):
        logger.info(f"Wywołanie api_function({param})")
        return f"API Wynik {param}"

    @adaptive_cache
    def smart_api_function(param):
        logger.info(f"Wywołanie smart_api_function({param})")

        # Symulacja czasami przekraczająca limit
        if param % 3 == 0:
            raise Exception("You have breached the rate limit. (ErrCode: 403)")

        return f"Smart API Wynik {param}"

    # Test cache'owania
    for i in range(5):
        logger.info(f"Test {i+1}")
        result = slow_function(1)
        logger.info(f"Wynik: {result}")
        time.sleep(1)

    # Test rate limitingu
    for i in range(5):
        result = api_function(i)
        logger.info(f"API Wynik: {result}")

    # Test adaptacyjnego cache'a
    for i in range(10):
        try:
            result = smart_api_function(i)
            logger.info(f"Smart API Wynik: {result}")
        except Exception as e:
            logger.error(f"Błąd: {e}")
        time.sleep(0.1)

    # Sprawdź status API
    status = get_api_status()
    logger.info(f"Status API: {status}")

#Missing functions from edited snippet are added here:
def cache_with_ttl(ttl: int = 300, key_prefix: str = ""):
    """
    Dekorator do cachowania wyników funkcji.

    Args:
        ttl: Czas życia cache'a w sekundach (domyślnie 5 minut)
        key_prefix: Prefiks dla klucza cache'a

    Returns:
        Dekorowana funkcja
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generuj klucz na podstawie argumentów funkcji
            args_str = str(args) + str(sorted(kwargs.items()))
            cache_key = f"{key_prefix}_{func.__name__}_{hash(args_str)}"

            # Sprawdź czy dane są w cache'u
            if is_cache_valid(cache_key, ttl):
                data, found = get_cached_data(cache_key)
                if found:
                    logger.debug(f"Używam cache'owanych danych dla: {func.__name__}")
                    return data

            # Wywołaj funkcję i zapisz wynik w cache'u
            result = func(*args, **kwargs)
            store_cached_data(cache_key, result)
            return result
        return wrapper
    return decorator

def rate_limit(func):
    """
    Dekorator do ograniczania częstotliwości wywołań funkcji (rate limiting).
    Zapobiega przekroczeniu limitów API.

    Args:
        func: Funkcja do udekorowania

    Returns:
        Dekorowana funkcja
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        current_time = time.time()

        # Sprawdź czy nie przekroczono limitu zapytań na minutę
        if _env_config["calls_count"] >= _env_config["max_calls_per_minute"]:
            sleep_time = 60 - (current_time - _env_config["window_start"]) + 0.1
            if sleep_time > 0:
                logger.warning(f"Przekroczono limit zapytań na minutę. Oczekiwanie {sleep_time:.2f}s")
                time.sleep(sleep_time)
            _env_config["calls_count"] = 0
            _env_config["window_start"] = time.time()


        # Zachowaj minimalny odstęp między zapytaniami
        time_since_last_call = current_time - _env_config["last_call"]
        if time_since_last_call < _env_config["min_interval"]:
            sleep_time = _env_config["min_interval"] - time_since_last_call
            time.sleep(sleep_time)

        # Aktualizuj liczniki
        _env_config["last_call"] = time.time()
        _env_config["calls_count"] += 1

        # Wywołaj oryginalną funkcję
        return func(*args, **kwargs)

    return wrapper


def adaptive_cache(func):
    """
    Dekorator łączący cache i rate limiting z adaptacyjnym TTL.
    Automatycznie wydłuża TTL w przypadku błędów API.

    Args:
        func: Funkcja do udekorowania

    Returns:
        Dekorowana funkcja
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generuj klucz cache'a
        args_str = str(args) + str(sorted(kwargs.items()))
        cache_key = f"adaptive_{func.__name__}_{hash(args_str)}"

        # Sprawdź czy dane są w cache'u
        ttl = 300

        # W przypadku API statusu sprawdzamy również flagę rate_limit
        rate_limit_key = "api_rate_limited"
        rate_limited, found = get_cached_data(rate_limit_key)
        if found and rate_limited:
            ttl = 3600
            logger.info(f"Wykryto flagę rate_limit, używam dłuższego TTL: {3600}s")

        if is_cache_valid(cache_key, ttl):
            data, found = get_cached_data(cache_key)
            if found:
                logger.debug(f"Używam cache'owanych danych dla: {func.__name__} (TTL: {ttl}s)")
                return data

        try:
            # Zastosuj rate limiting
            result = rate_limit(func)(*args, **kwargs)

            # Zapisz wynik w cache'u
            store_cached_data(cache_key, result)

            # Resetuj flagę rate_limit jeśli funkcja zakończyła się sukcesem
            store_cached_data(rate_limit_key, False)

            return result
        except Exception as e:
            # Sprawdź czy błąd dotyczy przekroczenia limitu zapytań
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str or "403" in error_str:
                logger.warning(f"Wykryto przekroczenie limitu API w funkcji {func.__name__}. Ustawiam flagę rate_limit.")

                # Ustaw flagę rate_limit
                store_cached_data(rate_limit_key, True)

                # Próbuj użyć ostatnich znanych danych z cache'a (nawet jeśli TTL minęło)
                data, found = get_cached_data(cache_key)
                if found:
                    logger.info(f"Używam przeterminowanych danych z cache dla {func.__name__} z powodu limitu API.")
                    return data

            # Propaguj wyjątek dalej
            raise

    return wrapper
from functools import wraps
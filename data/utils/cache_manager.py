import logging
import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from functools import wraps
from datetime import datetime, timedelta

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# Utworzenie katalogu cache, jeśli nie istnieje
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Konfiguracja logowania
# logger = logging.getLogger("cache_manager") #This line is redundant now
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Handler do pliku
    file_handler = logging.FileHandler(os.path.join("logs", "cache_manager.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler do konsoli
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Singleton do przechowywania cache'a w pamięci
_memory_cache = {}
_cache_timestamps = {}
_cache_lock = threading.RLock()

# Konfiguracja cache'a - dłuższe czasy życia dla lepszej wydajności
DEFAULT_TTL = 120      # Domyślny czas życia cache'a w sekundach (2 minuty)
LONG_TTL = 600         # Długi czas życia dla kosztownych zapytań (10 minut)
EMERGENCY_TTL = 3600   # Awaryjny czas życia (60 minut) w przypadku przekroczenia l"""
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
            
    return FalseNVIRONMENT_CONFIG['development']

# Ratelimiter globalny - z konfiguracją dostosowaną do środowiska
_rate_limiter = {
    "last_call": 0,
    "min_interval": _env_config['min_interval'],
    "calls_count": 0,
    "window_start": 0,
    "max_calls_per_minute": _env_config['max_calls_per_minute'],
    "lock": threading.RLock(),
    "environment": 'production' if _is_production else 'development',
    "cache_ttl_multiplier": _env_config['cache_ttl_multiplier'],
    "last_rate_limit_exceeded": 0,  # Czas ostatniego przekroczenia limitu
    "rate_limit_reset_time": _env_config['rate_limit_reset_time'],  # Czas po którym próbujemy ponownie po przekroczeniu limitu
    "server_time_ttl": _env_config['server_time_ttl'],  # TTL dla czasu serwera
    "startup_time": time.time()     # Czas startu aplikacji - pomocne przy zarządzaniu limitami
}

# Zapisz informację o środowisku w logu
logger.info(f"Cache skonfigurowany dla środowiska: {_rate_limiter['environment']}")
logger.info(f"Parametry rate limitera: max_calls={_rate_limiter['max_calls_per_minute']}, " +
           f"min_interval={_rate_limiter['min_interval']}s, " +
           f"cache_ttl_multiplier={_rate_limiter['cache_ttl_multiplier']}")


def get_cached_data(key: str) -> Tuple[Any, bool]:
    """
    Pobiera dane z cache'a.

    Args:
        key: Klucz pod którym zapisano dane

    Returns:
        Tuple zawierający (dane, czy_znaleziono)
    """
    with _cache_lock:
        if key in _memory_cache:
            return _memory_cache[key], True

    # Sprawdź cache na dysku
    file_path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                with _cache_lock:
                    _memory_cache[key] = data
                    _cache_timestamps[key] = os.path.getmtime(file_path)
                return data, True
        except Exception as e:
            logger.warning(f"Błąd podczas odczytu cache z dysku dla klucza {key}: {e}")

    return None, False


def store_cached_data(key: str, data: Any, persist: bool = True) -> None:
    """
    Zapisuje dane w cache'u.

    Args:
        key: Klucz pod którym zapisać dane
        data: Dane do zapisania
        persist: Czy zapisać dane na dysku
    """
    with _cache_lock:
        _memory_cache[key] = data
        _cache_timestamps[key] = time.time()

    if persist:
        try:
            file_path = os.path.join(CACHE_DIR, f"{key}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Błąd podczas zapisu cache na dysk dla klucza {key}: {e}")


def is_cache_valid(key: str, ttl: int = DEFAULT_TTL) -> bool:
    """
    Sprawdza czy cache jest wciąż ważny.

    Args:
        key: Klucz cache'a
        ttl: Czas życia w sekundach

    Returns:
        bool: True jeśli cache jest ważny, False w przeciwnym wypadku
    """
    with _cache_lock:
        if key not in _cache_timestamps:
            return False

        current_time = time.time()
        last_update = _cache_timestamps[key]

        return current_time - last_update < ttl


def clear_cache(key: str = None) -> None:
    """
    Czyści cache.

    Args:
        key: Opcjonalny klucz. Jeśli None, czyści cały cache.
    """
    with _cache_lock:
        if key is None:
            _memory_cache.clear()
            _cache_timestamps.clear()

            # Usuń pliki cache z dysku
            for file_name in os.listdir(CACHE_DIR):
                if file_name.endswith('.json'):
                    try:
                        os.remove(os.path.join(CACHE_DIR, file_name))
                    except Exception as e:
                        logger.warning(f"Nie można usunąć pliku cache {file_name}: {e}")
        else:
            if key in _memory_cache:
                del _memory_cache[key]
            if key in _cache_timestamps:
                del _cache_timestamps[key]

            # Usuń plik cache z dysku
            file_path = os.path.join(CACHE_DIR, f"{key}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Nie można usunąć pliku cache {key}: {e}")


def cache_with_ttl(ttl: int = DEFAULT_TTL, key_prefix: str = ""):
    """
    Dekorator do cachowania wyników funkcji.

    Args:
        ttl: Czas życia cache'a w sekundach
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
        with _rate_limiter["lock"]:
            current_time = time.time()

            # Sprawdź czy należy zresetować licznik (po minucie)
            if current_time - _rate_limiter["window_start"] > 60:
                _rate_limiter["window_start"] = current_time
                _rate_limiter["calls_count"] = 0

            # Sprawdź czy nie przekroczono limitu zapytań na minutę
            if _rate_limiter["calls_count"] >= _rate_limiter["max_calls_per_minute"]:
                sleep_time = 60 - (current_time - _rate_limiter["window_start"]) + 0.1
                if sleep_time > 0:
                    logger.warning(f"Przekroczono limit zapytań na minutę. Oczekiwanie {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    current_time = time.time()
                    _rate_limiter["window_start"] = current_time
                    _rate_limiter["calls_count"] = 0

            # Zachowaj minimalny odstęp między zapytaniami
            time_since_last_call = current_time - _rate_limiter["last_call"]
            if time_since_last_call < _rate_limiter["min_interval"]:
                sleep_time = _rate_limiter["min_interval"] - time_since_last_call
                time.sleep(sleep_time)

            # Aktualizuj liczniki
            _rate_limiter["last_call"] = time.time()
            _rate_limiter["calls_count"] += 1

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
        ttl = DEFAULT_TTL

        # W przypadku API statusu sprawdzamy również flagę rate_limit
        rate_limit_key = "api_rate_limited"
        rate_limited, found = get_cached_data(rate_limit_key)
        if found and rate_limited:
            ttl = EMERGENCY_TTL
            logger.info(f"Wykryto flagę rate_limit, używam dłuższego TTL: {EMERGENCY_TTL}s")

        if is_cache_valid(cache_key, ttl):
            data, found = get_cached_data(cache_key)
            if found:
                logger.debug(f"Używam cache'owanych danych dla: {func.__name__} (TTL: {ttl}s)")
                return data

        try:
            # Zastosuj rate limiting
            with _rate_limiter["lock"]:
                current_time = time.time()

                # Sprawdź czy należy zresetować licznik (po minucie)
                if current_time - _rate_limiter["window_start"] > 60:
                    _rate_limiter["window_start"] = current_time
                    _rate_limiter["calls_count"] = 0

                # Sprawdź czy nie przekroczono limitu zapytań na minutę
                if _rate_limiter["calls_count"] >= _rate_limiter["max_calls_per_minute"]:
                    sleep_time = 60 - (current_time - _rate_limiter["window_start"]) + 0.1
                    if sleep_time > 0:
                        logger.warning(f"Przekroczono limit zapytań na minutę. Oczekiwanie {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                        current_time = time.time()
                        _rate_limiter["window_start"] = current_time
                        _rate_limiter["calls_count"] = 0

                # Zachowaj minimalny odstęp między zapytaniami
                time_since_last_call = current_time - _rate_limiter["last_call"]
                if time_since_last_call < _rate_limiter["min_interval"]:
                    sleep_time = _rate_limiter["min_interval"] - time_since_last_call
                    time.sleep(sleep_time)

                # Aktualizuj liczniki
                _rate_limiter["last_call"] = time.time()
                _rate_limiter["calls_count"] += 1

            # Wywołaj oryginalną funkcję
            result = func(*args, **kwargs)

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


def get_api_status() -> Dict[str, Any]:
    """Zwraca aktualny status API i parametry limitera."""
    with _rate_limiter["lock"]:
        current_time = time.time()
        time_in_window = current_time - _rate_limiter["window_start"]
        calls_left = _rate_limiter["max_calls_per_minute"] - _rate_limiter["calls_count"]

        # Resetuj okno jeśli minęła minuta
        if time_in_window > 60:
            calls_left = _rate_limiter["max_calls_per_minute"]
            time_in_window = 0

        # Sprawdź czy jesteśmy w stanie rate_limit
        rate_limited, found = get_cached_data("api_rate_limited")
        is_limited = found and rate_limited

        # Sprawdź czy jesteśmy w fazie startowej (30 sekund od startu)
        startup_phase = is_in_startup_phase()

        return {
            "rate_limited": is_limited,
            "calls_in_current_window": _rate_limiter["calls_count"],
            "calls_left": max(0, calls_left),
            "window_reset_in": max(0, 60 - time_in_window),
            "last_call": _rate_limiter["last_call"],
            "min_interval": _rate_limiter["min_interval"],
            "in_startup_phase": startup_phase,
            "time_since_startup": current_time - _rate_limiter["startup_time"],
            "environment": _rate_limiter["environment"]
        }


def is_in_startup_phase():
    """
    Sprawdza czy aplikacja jest w fazie startowej (pierwsze 30 sekund od uruchomienia).
    W fazie startowej szczególnie dbamy o limity API.

    Returns:
        bool: True jeśli aplikacja jest w fazie startowej
    """
    with _rate_limiter["lock"]:
        current_time = time.time()
        time_since_startup = current_time - _rate_limiter["startup_time"]
        return time_since_startup < 30  # 30 sekund to faza startowa


def set_rate_limit_parameters(max_calls_per_minute: int = None, min_interval: float = None):
    """
    Ustawia parametry rate limitera.

    Args:
        max_calls_per_minute: Maksymalna liczba zapytań na minutę
        min_interval: Minimalny odstęp między zapytaniami w sekundach
    """
    with _rate_limiter["lock"]:
        if max_calls_per_minute is not None:
            _rate_limiter["max_calls_per_minute"] = max_calls_per_minute

        if min_interval is not None:
            _rate_limiter["min_interval"] = min_interval

        logger.info(f"Ustawiono parametry rate limitera: max_calls_per_minute={_rate_limiter['max_calls_per_minute']}, min_interval={_rate_limiter['min_interval']}s")


# Inicjalizacja cache'a
def init_cache():
    """Inicjalizuje cache i ustawia domyślne parametry."""
    # Ustaw konserwatywne limity dla Bybit API
    set_rate_limit_parameters(max_calls_per_minute=6, min_interval=10.0)

    # Ustaw w cache flagę rate_limit na False
    store_cached_data("api_rate_limited", False)

    logger.info("Cache zainicjalizowany z domyślnymi parametrami")


# Funkcje do obsługi blokady CloudFront
def detect_cloudfront_error(error_message: str) -> bool:
    """
    Wykrywa czy błąd pochodzi z CloudFront (AWS) lub jest związany z limitami IP.
    
    Args:
        error_message (str): Komunikat błędu do przeanalizowania
        
    Returns:
        bool: True jeśli wykryto błąd CloudFront lub limit IP
    """
    error_message = error_message.lower()
    cloudfront_indicators = [
        'cloudfront', 
        'distribution is configured', 
        'request could not be satisfied',
        '403 forbidden',
        'access denied',
        'rate limit',
        'too many requests',
        '429',
        'ip rate limit'
    ]
    
    for indicator in cloudfront_indicators:
        if indicator in error_message:
            return True
    return False


# Funkcje do obsługi blokady CloudFront
def detect_cloudfront_error(error_message: str) -> bool:
    """
    Wykrywa czy błąd pochodzi z CloudFront (AWS) lub jest związany z limitami IP.
    
    Args:
        error_message (str): Komunikat błędu do przeanalizowania
        
    Returns:
        bool: True jeśli wykryto błąd CloudFront lub limit IP
    """
    error_message = error_message.lower()
    cloudfront_indicators = [
        'cloudfront', 
        'distribution is configured', 
        'request could not be satisfied',
        '403 forbidden',
        'access denied',
        'rate limit',
        'too many requests',
        '429',
        'ip rate limit'
    ]
    
    for indicator in cloudfront_indicators:
        if indicator in error_message:
            logger.warning(f"Wykryto błąd CloudFront/limit IP: {indicator}")
            return True
            
    return False

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

# Automatyczna inicjalizacja przy imporcie
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
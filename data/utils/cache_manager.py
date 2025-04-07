import logging
import time
import threading
import os
import json
from typing import Any, Dict, List, Tuple, Optional
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
EMERGENCY_TTL = 3600   # Awaryjny czas życia (60 minut) w przypadku przekroczenia limitów API

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

# Sprawdź, czy używamy produkcyjnego API na podstawie zmiennej środowiskowej
import os
_is_production = os.getenv('BYBIT_USE_TESTNET', 'true').lower() != 'true'
_env_config = ENVIRONMENT_CONFIG['production'] if _is_production else ENVIRONMENT_CONFIG['development']

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


def get_api_status():
    """
    Sprawdza status limitów API i zwraca informacje o dostępności.

    Returns:
        Dict zawierający informacje o statusie API
    """
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
"""
cache_manager.py
---------------
Moduł do zarządzania cachingiem danych oraz ograniczeniami API.
Zapewnia efektywne buforowanie odpowiedzi API oraz inteligentne zarządzanie limitami zapytań.
"""

import json
import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, Tuple, Optional

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cache_manager")

# Katalog gdzie będą przechowywane dane cache
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Domyślne wartości parametrów rate limitera
_MAX_CALLS_PER_MINUTE = 3  # Zmniejszona wartość ze względu na błędy 403
_MIN_INTERVAL = 10.0  # Zwiększona wartość dla bezpieczniejszych zapytań
_CACHE_TTL_MULTIPLIER = 15.0  # Mnożnik czasu ważności cache

# Domyślne wartości z pliku .env lub konfiguracji
try:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Ustawienie parametrów na podstawie zmiennych środowiskowych
    _MAX_CALLS_PER_MINUTE = int(os.getenv("API_MAX_CALLS_PER_MINUTE", _MAX_CALLS_PER_MINUTE))
    _MIN_INTERVAL = float(os.getenv("API_MIN_INTERVAL", _MIN_INTERVAL))
    _CACHE_TTL_MULTIPLIER = float(os.getenv("API_CACHE_TTL_MULTIPLIER", _CACHE_TTL_MULTIPLIER))
    
    # Sprawdź środowisko i dostosuj parametry
    env = os.getenv("APP_ENV", "development").lower()
    if env == "production":
        logger.info(f"Cache skonfigurowany dla środowiska: {env}")
        # Bardziej konserwatywne wartości dla produkcji
        _MAX_CALLS_PER_MINUTE = min(_MAX_CALLS_PER_MINUTE, 3)
        _MIN_INTERVAL = max(_MIN_INTERVAL, 20.0)
    else:
        logger.info(f"Cache skonfigurowany dla środowiska: {env}")
except Exception as e:
    logger.warning(f"Błąd podczas ładowania zmiennych środowiskowych: {e}")

# Stan rate limiter'a
_rate_limit_state = {
    "last_call_time": 0,
    "calls_in_current_minute": 0,
    "minute_start_time": 0,
    "rate_limited": False,
    "rate_limit_reset_time": 0
}

logger.info(f"Parametry rate limitera: max_calls={_MAX_CALLS_PER_MINUTE}, min_interval={_MIN_INTERVAL}s, cache_ttl_multiplier={_CACHE_TTL_MULTIPLIER}")

def set_rate_limit_parameters(max_calls_per_minute=None, min_interval=None, cache_ttl_multiplier=None):
    """Aktualizuje parametry rate limitera."""
    global _MAX_CALLS_PER_MINUTE, _MIN_INTERVAL, _CACHE_TTL_MULTIPLIER
    
    if max_calls_per_minute is not None:
        _MAX_CALLS_PER_MINUTE = max_calls_per_minute
    
    if min_interval is not None:
        _MIN_INTERVAL = min_interval
    
    if cache_ttl_multiplier is not None:
        _CACHE_TTL_MULTIPLIER = cache_ttl_multiplier
    
    logger.info(f"Ustawiono parametry rate limitera: max_calls_per_minute={_MAX_CALLS_PER_MINUTE}, min_interval={_MIN_INTERVAL}s")

def get_api_status():
    """Zwraca aktualny status API i ograniczeń."""
    global _rate_limit_state
    
    current_time = time.time()
    
    # Resetuj licznik wywołań po upływie minuty
    if current_time - _rate_limit_state["minute_start_time"] > 60:
        _rate_limit_state["calls_in_current_minute"] = 0
        _rate_limit_state["minute_start_time"] = current_time
    
    # Sprawdź czy reset limitu już upłynął
    if _rate_limit_state["rate_limited"] and current_time > _rate_limit_state["rate_limit_reset_time"]:
        _rate_limit_state["rate_limited"] = False
        logger.info("Reset stanu przekroczenia limitów API - upłynął czas oczekiwania")
    
    return {
        "rate_limited": _rate_limit_state["rate_limited"],
        "calls_in_current_minute": _rate_limit_state["calls_in_current_minute"],
        "time_since_last_call": current_time - _rate_limit_state["last_call_time"],
        "remaining_calls": max(0, _MAX_CALLS_PER_MINUTE - _rate_limit_state["calls_in_current_minute"])
    }

def update_api_call_state():
    """Aktualizuje stan wywołań API - używane przy każdym wywołaniu API."""
    global _rate_limit_state
    
    current_time = time.time()
    
    # Resetuj licznik wywołań po upływie minuty
    if current_time - _rate_limit_state["minute_start_time"] > 60:
        _rate_limit_state["calls_in_current_minute"] = 0
        _rate_limit_state["minute_start_time"] = current_time
    
    # Zwiększ licznik wywołań
    _rate_limit_state["calls_in_current_minute"] += 1
    _rate_limit_state["last_call_time"] = current_time
    
    # Sprawdź czy przekroczono limit
    if _rate_limit_state["calls_in_current_minute"] >= _MAX_CALLS_PER_MINUTE:
        if not _rate_limit_state["rate_limited"]:
            _rate_limit_state["rate_limited"] = True
            _rate_limit_state["rate_limit_reset_time"] = current_time + 60
            logger.warning(f"Przekroczono limit wywołań API ({_MAX_CALLS_PER_MINUTE}/min). Ograniczanie wywołań na {60} sekund.")
            # Zapisz informację o przekroczeniu limitu w cache
            store_cached_data("api_rate_limited", True)

def apply_rate_limit() -> float:
    """
    Stosuje ograniczenia częstotliwości wywołań API.
    
    Returns:
        float: Czas oczekiwania (sleep) w sekundach.
    """
    global _rate_limit_state
    
    current_time = time.time()
    time_since_last_call = current_time - _rate_limit_state["last_call_time"]
    
    # Jeśli jesteśmy w stanie przekroczenia limitu, stosujemy bardziej agresywne ograniczenia
    if _rate_limit_state["rate_limited"]:
        # Sprawdź czy reset limitu już upłynął
        if current_time > _rate_limit_state["rate_limit_reset_time"]:
            _rate_limit_state["rate_limited"] = False
            logger.info("Reset stanu przekroczenia limitów API - upłynął czas oczekiwania")
            # Zapisz informację o resecie limitu w cache
            store_cached_data("api_rate_limited", False)
        else:
            # Bardziej agresywne oczekiwanie, jeśli przekroczono limit
            sleep_time = max(_MIN_INTERVAL * 2, 15.0)
            logger.info(f"Limit API przekroczony - oczekiwanie {sleep_time:.1f}s")
            time.sleep(sleep_time)
            return sleep_time
    
    # Standardowe ograniczenie częstotliwości
    if time_since_last_call < _MIN_INTERVAL:
        sleep_time = _MIN_INTERVAL - time_since_last_call
        time.sleep(sleep_time)
        return sleep_time
    
    return 0.0

def _get_cache_file_path(key: str) -> str:
    """Zwraca ścieżkę do pliku cache dla danego klucza."""
    # Zastąp wszystkie znaki, które mogą być niedozwolone w nazwach plików
    safe_key = key.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    return os.path.join(CACHE_DIR, f"{safe_key}.json")

def store_cached_data(key: str, data: Any) -> bool:
    """
    Zapisuje dane w cache.
    
    Args:
        key: Klucz identyfikujący dane.
        data: Dane do zapisania (muszą być serializowalne do JSON).
        
    Returns:
        bool: True jeśli operacja się powiodła, False w przeciwnym wypadku.
    """
    try:
        cache_file = _get_cache_file_path(key)
        cache_data = {
            "timestamp": time.time(),
            "data": data
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        return True
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania danych w cache: {e}")
        return False

def get_cached_data(key: str) -> Tuple[Any, bool]:
    """
    Pobiera dane z cache.
    
    Args:
        key: Klucz identyfikujący dane.
        
    Returns:
        Tuple[Any, bool]: (dane, znaleziono) - gdzie znaleziono to True jeśli dane zostały znalezione.
    """
    try:
        cache_file = _get_cache_file_path(key)
        
        if not os.path.exists(cache_file):
            return None, False
        
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        return cache_data["data"], True
    except Exception as e:
        logger.error(f"Błąd podczas odczytu danych z cache: {e}")
        return None, False

def is_cache_valid(key: str, ttl: int = 60) -> bool:
    """
    Sprawdza czy dane w cache są wciąż ważne.
    
    Args:
        key: Klucz identyfikujący dane.
        ttl: Czas ważności w sekundach.
        
    Returns:
        bool: True jeśli cache jest ważny, False w przeciwnym wypadku.
    """
    try:
        cache_file = _get_cache_file_path(key)
        
        if not os.path.exists(cache_file):
            return False
        
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Sprawdź czy dane są wystarczająco aktualne
        current_time = time.time()
        cache_time = cache_data.get("timestamp", 0)
        
        # Jeśli stan API jest ograniczony, użyj dłuższego TTL
        api_status = get_api_status()
        effective_ttl = ttl
        
        if api_status["rate_limited"]:
            effective_ttl = ttl * _CACHE_TTL_MULTIPLIER
        
        return (current_time - cache_time) < effective_ttl
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania ważności cache: {e}")
        return False

def clear_cache(key: str = None) -> bool:
    """
    Czyści cache dla danego klucza lub cały cache.
    
    Args:
        key: Klucz do wyczyszczenia. Jeśli None, czyści wszystkie dane.
        
    Returns:
        bool: True jeśli operacja się powiodła, False w przeciwnym wypadku.
    """
    try:
        if key:
            cache_file = _get_cache_file_path(key)
            if os.path.exists(cache_file):
                os.remove(cache_file)
        else:
            for filename in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, filename)
                if os.path.isfile(file_path) and filename.endswith('.json'):
                    os.remove(file_path)
        
        return True
    except Exception as e:
        logger.error(f"Błąd podczas czyszczenia cache: {e}")
        return False

def get_cache_stats() -> Dict[str, Any]:
    """
    Zwraca statystyki dotyczące cache.
    
    Returns:
        Dict[str, Any]: Statystyki cache.
    """
    try:
        stats = {
            "total_entries": 0,
            "total_size_bytes": 0,
            "oldest_entry": None,
            "newest_entry": None,
            "entries": []
        }
        
        if not os.path.exists(CACHE_DIR):
            return stats
        
        oldest_time = float('inf')
        newest_time = 0
        
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(CACHE_DIR, filename)
                
                # Statystyki pliku
                file_size = os.path.getsize(file_path)
                stats["total_size_bytes"] += file_size
                stats["total_entries"] += 1
                
                try:
                    with open(file_path, 'r') as f:
                        cache_data = json.load(f)
                    
                    timestamp = cache_data.get("timestamp", 0)
                    
                    if timestamp < oldest_time:
                        oldest_time = timestamp
                        stats["oldest_entry"] = filename
                    
                    if timestamp > newest_time:
                        newest_time = timestamp
                        stats["newest_entry"] = filename
                    
                    stats["entries"].append({
                        "key": filename[:-5],  # Usuń rozszerzenie .json
                        "timestamp": timestamp,
                        "size_bytes": file_size,
                        "age_seconds": time.time() - timestamp
                    })
                except:
                    pass
        
        return stats
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statystyk cache: {e}")
        return {"error": str(e)}

# Inicjalizacja komponentu
logger.info("Cache zainicjalizowany z domyślnymi parametrami")

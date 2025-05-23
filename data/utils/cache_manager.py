"""
cache_manager.py
---------------
Moduł odpowiedzialny za zarządzanie cache'em danych API i limitami zapytań.
Implementuje mechanizmy buforowania oraz kontroli dostępu do API.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

# Konfiguracja logowania - Ulepszona wersja z oryginalnego kodu i edytora
logger = logging.getLogger("cache_manager")
if not logger.handlers:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "cache_manager.log"))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

# Globalne zmienne konfiguracyjne
CACHE_DIR = "data/cache"
DEFAULT_TTL = 300  # 5 minut

# Parametry limitowania zapytań API
MAX_CALLS_PER_MINUTE = 6  # Domyślna wartość
MIN_INTERVAL_BETWEEN_CALLS = 10.0  # Domyślny minimalny odstęp w sekundach
RATE_LIMIT_LAST_RESET = time.time()
RATE_LIMIT_EXCEEDED = False
CLOUDFRONT_BLOCK = {
    "blocked": False,
    "since": 0,
    "error": "",
    "reset_time": 0
}

# Statystyki API
API_STATS = {
    "last_call_time": 0,
    "call_count": 0,
    "error_count": 0,
    "last_error": None,
    "rate_limited": False
}

def init_cache_manager():
    """Inicjuje menedżer cache, tworząc niezbędne katalogi."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info("Cache zainicjalizowany z domyślnymi parametrami")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas inicjalizacji cache: {e}")
        return False

def set_rate_limit_parameters(max_calls_per_minute: int = 6, min_interval: float = 10.0):
    """
    Ustawia parametry limitowania zapytań API.

    Args:
        max_calls_per_minute (int): Maksymalna liczba wywołań na minutę
        min_interval (float): Minimalny odstęp między wywołaniami w sekundach
    """
    global MAX_CALLS_PER_MINUTE, MIN_INTERVAL_BETWEEN_CALLS

    MAX_CALLS_PER_MINUTE = max_calls_per_minute
    MIN_INTERVAL_BETWEEN_CALLS = min_interval
    logger.info(f"Ustawiono parametry rate limitera: max_calls_per_minute={max_calls_per_minute}, min_interval={min_interval}s")

def _get_cache_path(key: str) -> str:
    """Generuje ścieżkę do pliku cache dla danego klucza."""
    # Upewnij się, że katalog cache istnieje
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Upewnij się, że klucz jest bezpieczny i nie zawiera znaków niedozwolonych w ścieżkach
    safe_key = "".join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in key)

    # Użyj os.path.join dla kompatybilności między systemami
    return os.path.join(CACHE_DIR, f"{safe_key}.json")

def store_cached_data(key: str, data: Any) -> bool:
    """
    Zapisuje dane w cache.

    Args:
        key (str): Klucz identyfikujący dane
        data (Any): Dane do zapisania

    Returns:
        bool: True jeśli zapis się powiódł, False w przeciwnym wypadku
    """
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = _get_cache_path(key)

        cache_entry = {
            "data": data,
            "timestamp": time.time(),
            "ttl": DEFAULT_TTL
        }

        with open(cache_path, 'w') as f:
            json.dump(cache_entry, f)

        logger.debug(f"Zapisano dane w cache dla klucza: {key}")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania danych w cache dla klucza {key}: {e}")
        return False

def get_cached_data(key: str) -> Tuple[Any, bool]:
    """Pobiera dane z cache z poprawną obsługą typów."""
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        cache_file = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Prawidłowa obsługa typów danych
            if isinstance(data, bool):
                # Jeśli dane to boolean, zwróć go bezpośrednio
                return data, True
            elif isinstance(data, dict) and 'data' in data:
                # Jeśli dane są zagnieżdżone w kluczu 'data', wyodrębnij je
                return data['data'], True
            else:
                # W przeciwnym razie zwróć dane jako są
                return data, True
        return None, False
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych z cache dla klucza {key}: {e}")
        return None, False

def is_cache_valid(key: str, ttl: int = DEFAULT_TTL) -> bool:
    """Sprawdza czy dane w cache są ważne z rozszerzoną obsługą błędów."""
    try:
        cache_file = os.path.join(CACHE_DIR, f"{key}.json")
        if not os.path.exists(cache_file):
            logger.debug(f"Plik cache {key} nie istnieje")
            return False

        # Sprawdzenie czy plik nie jest pusty
        if os.path.getsize(cache_file) == 0:
            logger.warning(f"Plik cache {key} jest pusty")
            return False

        # Sprawdzenie czy plik można odczytać
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Plik cache {key} zawiera nieprawidłowy JSON")
            return False

        # Sprawdzenie czasu modyfikacji pliku
        mod_time = os.path.getmtime(cache_file)
        current_time = time.time()

        # Jeśli plik jest starszy niż TTL, to cache jest nieważny
        is_valid = (current_time - mod_time) < ttl
        if not is_valid:
            logger.debug(f"Cache {key} wygasł (upłynęło {current_time - mod_time:.1f}s, TTL={ttl}s)")

        return is_valid
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania ważności cache dla klucza {key}: {e}")
        return False

def invalidate_cache(key: str) -> bool:
    """
    Unieważnia wpis w cache.

    Args:
        key (str): Klucz identyfikujący dane

    Returns:
        bool: True jeśli unieważnienie się powiodło, False w przeciwnym przypadku
    """
    try:
        cache_path = _get_cache_path(key)

        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.debug(f"Unieważniono cache dla klucza: {key}")
            return True

        logger.debug(f"Brak danych w cache dla klucza: {key}")
        return False
    except Exception as e:
        logger.error(f"Błąd podczas unieważniania cache dla klucza {key}: {e}")
        return False

def clear_all_cache() -> bool:
    """
    Czyści cały cache.

    Returns:
        bool: True jeśli czyszczenie się powiodło, False w przeciwnym przypadku
    """
    try:
        if os.path.exists(CACHE_DIR):
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith('.json'):
                    os.remove(os.path.join(CACHE_DIR, filename))

        logger.info("Wyczyszczono cały cache")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas czyszczenia cache: {e}")
        return False

def get_api_status() -> Dict[str, Any]:
    """
    Zwraca aktualny status API.

    Returns:
        Dict[str, Any]: Słownik zawierający status API
    """
    global API_STATS, RATE_LIMIT_LAST_RESET, RATE_LIMIT_EXCEEDED, CLOUDFRONT_BLOCK

    # Sprawdź, czy przekroczyliśmy limit zapytań
    if API_STATS["call_count"] > MAX_CALLS_PER_MINUTE and time.time() - RATE_LIMIT_LAST_RESET < 60:
        API_STATS["rate_limited"] = True
    else:
        # Reset licznika co minutę
        if time.time() - RATE_LIMIT_LAST_RESET >= 60:
            API_STATS["call_count"] = 0
            API_STATS["rate_limited"] = False

    # Bezpieczne pobieranie wartości z CLOUDFRONT_BLOCK
    cloudfront_blocked = False
    if isinstance(CLOUDFRONT_BLOCK, dict) and "blocked" in CLOUDFRONT_BLOCK:
        cloudfront_blocked = CLOUDFRONT_BLOCK["blocked"]

    return {
        "last_call_time": API_STATS["last_call_time"],
        "call_count": API_STATS["call_count"],
        "error_count": API_STATS["error_count"],
        "last_error": API_STATS["last_error"],
        "rate_limited": API_STATS["rate_limited"] or RATE_LIMIT_EXCEEDED or cloudfront_blocked,
        "cloudfront_blocked": cloudfront_blocked
    }

def record_api_call(success: bool = True, error: Optional[str] = None) -> None:
    """
    Rejestruje wywołanie API.

    Args:
        success (bool): Czy wywołanie się powiodło
        error (Optional[str]): Komunikat błędu, jeśli wystąpił
    """
    global API_STATS, RATE_LIMIT_LAST_RESET

    API_STATS["last_call_time"] = time.time()
    API_STATS["call_count"] += 1

    if not success:
        API_STATS["error_count"] += 1
        API_STATS["last_error"] = error

        # Sprawdź, czy błąd dotyczy przekroczenia limitu
        if error and ("rate limit" in error.lower() or "429" in error or "403" in error):
            API_STATS["rate_limited"] = True

    # Reset licznika co minutę
    if time.time() - RATE_LIMIT_LAST_RESET >= 60:
        RATE_LIMIT_LAST_RESET = time.time()
        API_STATS["call_count"] = 1  # Zaczynamy od 1, bo właśnie wykonaliśmy wywołanie

def detect_cloudfront_error(error_message: str) -> bool:
    """
    Wykrywa błędy związane z CloudFront i limitami IP w komunikacie błędu.

    Args:
        error_message (str): Komunikat błędu do analizy

    Returns:
        bool: True jeśli wykryto błąd CloudFront lub limit IP, False w przeciwnym przypadku
    """
    error_lower = error_message.lower()
    cloudfront_indicators = [
        'cloudfront', 
        'distribution', 
        '403', 
        'rate limit', 
        '429', 
        'too many requests',
        'access denied',
        'quota exceeded',
        'ip has been blocked'
    ]

    return any(indicator in error_lower for indicator in cloudfront_indicators)

def set_cloudfront_block_status(blocked: bool, error_message: str = "") -> None:
    """
    Ustawia status blokady CloudFront.

    Args:
        blocked (bool): Czy blokada jest aktywna
        error_message (str): Komunikat błędu związany z blokadą
    """
    global CLOUDFRONT_BLOCK

    if blocked:
        CLOUDFRONT_BLOCK["blocked"] = True
        CLOUDFRONT_BLOCK["since"] = time.time()
        CLOUDFRONT_BLOCK["error"] = error_message
        CLOUDFRONT_BLOCK["reset_time"] = time.time() + 1800  # 30 minut blokady
        logger.warning(f"Ustawiono status blokady CloudFront: {error_message}")
    else:
        # Resetuj status blokady
        CLOUDFRONT_BLOCK["blocked"] = False
        CLOUDFRONT_BLOCK["error"] = ""
        logger.info("Zresetowano status blokady CloudFront")

def get_cloudfront_status() -> Dict[str, Any]:
    """
    Zwraca aktualny status blokady CloudFront.

    Returns:
        Dict[str, Any]: Słownik zawierający status blokady CloudFront
    """
    global CLOUDFRONT_BLOCK

    # Sprawdź, czy minął czas blokady
    if CLOUDFRONT_BLOCK["blocked"] and time.time() > CLOUDFRONT_BLOCK["reset_time"]:
        CLOUDFRONT_BLOCK["blocked"] = False
        CLOUDFRONT_BLOCK["error"] = ""
        logger.info("Automatyczny reset statusu blokady CloudFront po upływie czasu blokady")

    return {
        "blocked": CLOUDFRONT_BLOCK["blocked"],
        "since": CLOUDFRONT_BLOCK["since"],
        "error": CLOUDFRONT_BLOCK["error"],
        "reset_time": CLOUDFRONT_BLOCK["reset_time"],
        "time_left": max(0, CLOUDFRONT_BLOCK["reset_time"] - time.time()) if CLOUDFRONT_BLOCK["blocked"] else 0
    }

# Singleton pattern dla inicjalizacji cache_manager
_instance_initialized = False

def get_instance():
    """Uzyskaj instancję cache_manager (singleton pattern)"""
    global _instance_initialized
    if not _instance_initialized:
        init_cache_manager()
        _instance_initialized = True
    return _instance_initialized

# Automatyczna inicjalizacja przy imporcie, ale tylko raz
if not _instance_initialized:
    init_cache_manager()
    _instance_initialized = True

from functools import wraps

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
        global API_STATS, RATE_LIMIT_LAST_RESET, RATE_LIMIT_EXCEEDED
        current_time = time.time()

        # Sprawdź czy nie przekroczono limitu zapytań na minutę
        if API_STATS["call_count"] >= MAX_CALLS_PER_MINUTE:
            sleep_time = 60 - (current_time - RATE_LIMIT_LAST_RESET) + 0.1
            if sleep_time > 0:
                logger.warning(f"Przekroczono limit zapytań na minutę. Oczekiwanie {sleep_time:.2f}s")
                time.sleep(sleep_time)
            RATE_LIMIT_LAST_RESET = current_time
            API_STATS["call_count"] = 0
            RATE_LIMIT_EXCEEDED = True

        # Zachowaj minimalny odstęp między zapytaniami
        time_since_last_call = current_time - API_STATS["last_call_time"]
        if time_since_last_call < MIN_INTERVAL_BETWEEN_CALLS:
            sleep_time = MIN_INTERVAL_BETWEEN_CALLS - time_since_last_call
            time.sleep(sleep_time)

        # Aktualizuj liczniki
        API_STATS["last_call_time"] = current_time
        API_STATS["call_count"] += 1
        RATE_LIMIT_EXCEEDED = False

        # Wywołaj oryginalną funkcję
        try:
            result = func(*args, **kwargs)
            record_api_call() #Record successful call
            return result
        except Exception as e:
            record_api_call(False, str(e)) #Record failed call with error
            raise

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
        if get_api_status()["rate_limited"]:
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
            return result
        except Exception as e:
            # Sprawdź czy błąd dotyczy przekroczenia limitu zapytań
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str or "403" in error_str:
                logger.warning(f"Wykryto przekroczenie limitu API w funkcji {func.__name__}.")

                # Próbuj użyć ostatnich znanych danych z cache'a (nawet jeśli TTL minęło)
                data, found = get_cached_data(cache_key)
                if found:
                    logger.info(f"Używam przeterminowanych danych z cache dla {func.__name__} z powodu limitu API.")
                    return data

            # Propaguj wyjątek dalej
            raise

    return wrapper

# Przykład użycia (z oryginalnego kodu) - Zostawione dla testów
if __name__ == "__main__":
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

def safe_cache_get(key: str, default_value=None, expected_keys: List[str] = None):
    """
    Safely retrieves data from cache with validation and fallback.

    Parameters:
        key (str): Cache key to retrieve
        default_value: Value to return if cache miss or validation fails
        expected_keys (List[str]): Optional list of keys that must exist in cached dict

    Returns:
        Any: The cached value if found and valid, otherwise default_value
    """
    try:
        data, found = get_cached_data(key)

        # Check if data was found
        if not found or data is None:
            return default_value

        # Konwersja boolean na dict dla zachowania kompatybilności
        if isinstance(data, bool):
            data = {"value": data}
            logger.warning(f"Konwertowano boolean na dict dla klucza {key}")

        # If expected_keys provided, verify all keys exist in data
        if expected_keys and isinstance(data, dict):
            if not all(k in data for k in expected_keys):
                missing_keys = [k for k in expected_keys if k not in data]
                logging.warning(f"Cache data for {key} missing required keys: {missing_keys}")
                return default_value

        return data
    except Exception as e:
        logging.error(f"Error in safe_cache_get for key {key}: {e}")
        return default_value

def clean_old_data(max_age_hours=24, max_cache_size_mb=500):
    """
    Czyści stare dane z cache na podstawie wieku i limituje całkowity rozmiar cache.

    Args:
        max_age_hours (int): Maksymalny wiek danych w godzinach
        max_cache_size_mb (int): Maksymalny rozmiar cache w MB
    """
    now = datetime.now()
    files_removed = 0
    size_before = 0
    size_after = 0

    try:
        cache_folder = os.path.join('data', 'cache')

        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder, exist_ok=True)
            return

        # Zbierz informacje o plikach (ścieżka, rozmiar, wiek)
        files_info = []
        for filename in os.listdir(cache_folder):
            file_path = os.path.join(cache_folder, filename)

            # Sprawdź, czy to plik (nie katalog)
            if not os.path.isfile(file_path):
                continue

            # Pobierz rozmiar pliku w bajtach
            file_size = os.path.getsize(file_path)
            size_before += file_size

            # Sprawdź wiek pliku
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            age_hours = (now - file_mod_time).total_seconds() / 3600

            files_info.append((file_path, file_size, age_hours))

        # Usuń stare pliki
        for file_path, file_size, age_hours in files_info:
            if age_hours > max_age_hours:
                os.remove(file_path)
                files_removed += 1
            else:
                size_after += file_size

        # Jeśli rozmiar cache nadal przekracza limit
        if size_after > max_cache_size_mb * 1024 * 1024:
            # Posortuj pozostałe pliki według wieku (od najstarszego)
            remaining_files = [(p, s, a) for p, s, a in files_info if a <= max_age_hours]
            remaining_files.sort(key=lambda x: x[2], reverse=True)

            # Usuń najstarsze pliki aż do osiągnięcia limitu rozmiaru
            for file_path, file_size, _ in remaining_files:
                if size_after <= max_cache_size_mb * 1024 * 1024:
                    break

                if os.path.exists(file_path):  # Sprawdź czy plik nadal istnieje
                    os.remove(file_path)
                    size_after -= file_size
                    files_removed += 1

        # Raportuj wyniki czyszczenia
        mb_before = size_before / (1024 * 1024)
        mb_after = size_after / (1024 * 1024)
        logging.info(f"Cache czyszczenie: usunięto {files_removed} plików")
        logging.info(f"Rozmiar cache przed: {mb_before:.2f} MB, po: {mb_after:.2f} MB")

        return files_removed
    except Exception as e:
        logging.error(f"Błąd podczas czyszczenia cache: {e}")
        return 0

# Funkcja do automatycznego czyszczenia cache co określony czas
def setup_auto_cleanup(interval_hours=6):
    """
    Konfiguruje automatyczne czyszczenie cache w regularnych odstępach czasu.

    Args:
        interval_hours (int): Częstotliwość czyszczenia w godzinach
    """
    import threading
    import time

    def cleanup_thread():
        while True:
            # Czyszczenie cache
            try:
                files_removed = clean_old_data()
                logging.info(f"Automatyczne czyszczenie cache: usunięto {files_removed} plików")
            except Exception as e:
                logging.error(f"Błąd podczas automatycznego czyszczenia cache: {e}")

            # Czekaj określony czas
            time.sleep(interval_hours * 3600)

    # Uruchom wątek czyszczenia w tle
    cleanup_thread = threading.Thread(target=cleanup_thread, daemon=True)
    cleanup_thread.start()
    logging.info(f"Skonfigurowano automatyczne czyszczenie cache co {interval_hours} godzin")

# Uruchom automatyczne czyszczenie przy imporcie
try:
    setup_auto_cleanup()
except Exception as e:
    logging.error(f"Nie udało się skonfigurować automatycznego czyszczenia cache: {e}")

class CacheManager:
    """Enhanced cache management system with memory optimization."""
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        max_memory_mb: int = 512,
        cleanup_interval: int = 3600,
        max_age_hours: int = 24
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
            max_memory_mb: Maximum memory usage in MB
            cleanup_interval: Cleanup interval in seconds
            max_age_hours: Maximum age of cache entries in hours
        """
        self.cache_dir = cache_dir
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.cleanup_interval = cleanup_interval
        self.max_age = timedelta(hours=max_age_hours)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Memory cache with weak references
        self._memory_cache = weakref.WeakValueDictionary()
        
        # Start cleanup thread
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"Cache manager initialized: {cache_dir} (max: {max_memory_mb}MB)")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Try memory cache first
            if key in self._memory_cache:
                return self._memory_cache[key]
            
            # Try file cache
            cache_file = os.path.join(self.cache_dir, f"{self._hash_key(key)}.json")
            if os.path.exists(cache_file):
                modification_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - modification_time <= self.max_age:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        self._memory_cache[key] = data
                        return data
                else:
                    os.remove(cache_file)
            
            return None
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def set(self, key: str, value: Any) -> bool:
        """Set value in cache."""
        try:
            # Store in memory cache
            self._memory_cache[key] = value
            
            # Store in file cache
            cache_file = os.path.join(self.cache_dir, f"{self._hash_key(key)}.json")
            with open(cache_file, 'w') as f:
                json.dump(value, f)
            
            # Check memory usage
            self._check_memory_usage()
            
            return True
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            # Remove from memory cache
            self._memory_cache.pop(key, None)
            
            # Remove from file cache
            cache_file = os.path.join(self.cache_dir, f"{self._hash_key(key)}.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache."""
        try:
            # Clear memory cache
            self._memory_cache.clear()
            
            # Clear file cache
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._stop_cleanup.is_set():
            try:
                self._cleanup_expired()
                self._check_memory_usage()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
            
            self._stop_cleanup.wait(self.cleanup_interval)

    def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        now = datetime.now()
        for file in os.listdir(self.cache_dir):
            try:
                file_path = os.path.join(self.cache_dir, file)
                if now - datetime.fromtimestamp(os.path.getmtime(file_path)) > self.max_age:
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error cleaning up file {file}: {e}")

    def _check_memory_usage(self) -> None:
        """Check and optimize memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_use = process.memory_info().rss
            
            if memory_use > self.max_memory:
                # Clear memory cache if usage is too high
                self._memory_cache.clear()
                logger.warning(f"Memory usage too high ({memory_use/1024/1024:.1f}MB), cleared memory cache")
        except ImportError:
            logger.warning("psutil not available, cannot check memory usage")
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")

    def _hash_key(self, key: str) -> str:
        """Create hash from cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_cleanup.set()
        self._cleanup_thread.join()

    def __del__(self):
        self._stop_cleanup.set()

# Global cache manager instance
cache_manager = CacheManager()

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    return cache_manager
"""
cache_manager.py
----------------
Moduł zarządzający pamięcią podręczną (cache).

Funkcjonalności:
- Implementacja strategii wygasania: time-based oraz LRU.
- Konfigurowalny rozmiar cache i czas wygaśnięcia wpisów.
- Mechanizmy monitoringu rozmiaru cache oraz automatycznego oczyszczania przy przekroczeniu limitów.
- Testy wydajnościowe i obsługa scenariuszy brzegowych (np. brak miejsca na dysku).
"""

import logging
import time
from collections import OrderedDict

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class CacheEntry:
    def __init__(self, value, timestamp, ttl):
        self.value = value
        self.timestamp = timestamp
        self.ttl = ttl  # Time to live in seconds

    def is_expired(self):
        return (time.time() - self.timestamp) > self.ttl


class CacheManager:
    def __init__(
        self, max_size: int = 128, default_ttl: int = 300, strategy: str = "LRU"
    ):
        """
        Inicjalizuje CacheManager.

        Parameters:
            max_size (int): Maksymalna liczba wpisów w cache.
            default_ttl (int): Domyślny czas wygaśnięcia wpisu (w sekundach).
            strategy (str): Strategia zarządzania cache, np. 'LRU'.
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy.upper()
        if self.strategy != "LRU":
            raise ValueError("Obecnie obsługiwana jest tylko strategia LRU.")
        self.cache = OrderedDict()
        logging.info(
            "CacheManager zainicjalizowany (max_size=%d, default_ttl=%d, strategy=%s).",
            self.max_size,
            self.default_ttl,
            self.strategy,
        )

    def set(self, key, value, ttl: int = None):
        """
        Dodaje lub aktualizuje wpis w cache.

        Parameters:
            key: Klucz identyfikujący wpis.
            value: Wartość do zapamiętania.
            ttl (int): Czas wygaśnięcia wpisu w sekundach. Jeśli None, używany jest default_ttl.
        """
        ttl = ttl if ttl is not None else self.default_ttl
        current_time = time.time()
        if key in self.cache:
            logging.debug("Aktualizacja istniejącego wpisu dla klucza: %s", key)
            del self.cache[key]  # Usuwamy, aby dodać wpis na końcu (najświeższy)
        self.cache[key] = CacheEntry(value, current_time, ttl)
        logging.debug("Wpis dodany: %s -> %s", key, value)
        self._ensure_capacity()

    def get(self, key):
        """
        Pobiera wartość z cache, jeśli wpis nie wygasł.

        Parameters:
            key: Klucz wpisu.

        Returns:
            Wartość wpisu lub None, jeśli wpis nie istnieje lub wygasł.
        """
        if key not in self.cache:
            logging.debug("Klucz %s nie znaleziony w cache.", key)
            return None
        entry = self.cache[key]
        if entry.is_expired():
            logging.debug("Wpis dla klucza %s wygasł.", key)
            del self.cache[key]
            return None
        # Przenieś wpis na koniec, aby oznaczyć go jako ostatnio używany (LRU)
        self.cache.move_to_end(key)
        logging.debug("Pobrano wpis z cache dla klucza %s.", key)
        return entry.value

    def delete(self, key):
        """
        Usuwa wpis z cache.

        Parameters:
            key: Klucz wpisu.
        """
        if key in self.cache:
            del self.cache[key]
            logging.debug("Usunięto wpis z cache dla klucza %s.", key)

    def _ensure_capacity(self):
        """
        Sprawdza, czy rozmiar cache nie przekracza limitu i usuwa najstarsze wpisy, jeśli to konieczne.
        """
        while len(self.cache) > self.max_size:
            removed_key, removed_entry = self.cache.popitem(last=False)
            logging.info(
                "Cache przekroczony limit. Usunięto najstarszy wpis: %s", removed_key
            )

    def clear_expired(self):
        """
        Usuwa wszystkie wygasłe wpisy z cache.
        """
        keys_to_delete = [
            key for key, entry in self.cache.items() if entry.is_expired()
        ]
        for key in keys_to_delete:
            del self.cache[key]
            logging.debug("Usunięto wygasły wpis dla klucza: %s", key)

    def cache_size(self):
        """
        Zwraca aktualny rozmiar cache.
        """
        return len(self.cache)


# -------------------- Testy jednostkowe --------------------
if __name__ == "__main__":
    # Proste testy wydajnościowe i funkcjonalne dla CacheManager
    import unittest

    class TestCacheManager(unittest.TestCase):
        def setUp(self):
            self.cache = CacheManager(
                max_size=5, default_ttl=2
            )  # Krótki TTL dla testów

        def test_set_and_get(self):
            self.cache.set("a", 1)
            self.assertEqual(
                self.cache.get("a"), 1, "Wartość dla klucza 'a' powinna być 1."
            )

        def test_expiration(self):
            self.cache.set("b", 2, ttl=1)  # TTL = 1 sekunda
            time.sleep(1.1)
            self.assertIsNone(
                self.cache.get("b"), "Wpis dla klucza 'b' powinien wygasnąć."
            )

        def test_capacity(self):
            # Dodajemy 6 wpisów, max_size = 5, więc najstarszy powinien zostać usunięty.
            for i in range(6):
                self.cache.set(f"key{i}", i)
            self.assertEqual(
                self.cache.cache_size(), 5, "Rozmiar cache powinien wynosić 5."
            )

        def test_delete(self):
            self.cache.set("c", 3)
            self.cache.delete("c")
            self.assertIsNone(
                self.cache.get("c"), "Wpis dla klucza 'c' powinien zostać usunięty."
            )

        def test_clear_expired(self):
            self.cache.set("d", 4, ttl=1)
            self.cache.set("e", 5, ttl=5)
            time.sleep(1.2)
            self.cache.clear_expired()
            self.assertIsNone(
                self.cache.get("d"),
                "Wpis dla klucza 'd' powinien zostać usunięty po wygaśnięciu.",
            )
            self.assertEqual(
                self.cache.get("e"), 5, "Wpis dla klucza 'e' powinien nadal istnieć."
            )

    unittest.main()
"""
cache_manager.py
-----------------
Moduł obsługujący cache'owanie danych z API, aby zmniejszyć liczbę zapytań.
"""

import time
import logging
from functools import wraps
from typing import Dict, Any, Callable, Optional, Tuple

# Konfiguracja logowania
logger = logging.getLogger("cache_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Główny cache przechowujący dane
_CACHE: Dict[str, Dict[str, Any]] = {}

def cache_data(ttl: int = 30) -> Callable:
    """
    Dekorator do cache'owania wyników funkcji.
    
    Args:
        ttl: Czas życia cache'u w sekundach (domyślnie 30s)
    
    Returns:
        Callable: Dekorator
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generowanie klucza cache'u
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Sprawdzenie czy dane są w cache'u i aktualne
            if cache_key in _CACHE:
                entry = _CACHE[cache_key]
                if time.time() - entry["timestamp"] < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return entry["data"]
            
            # Wywołanie oryginalnej funkcji
            result = func(*args, **kwargs)
            
            # Zapisanie wyniku w cache'u
            _CACHE[cache_key] = {
                "data": result,
                "timestamp": time.time()
            }
            logger.debug(f"Cache store for {func.__name__}")
            
            return result
        return wrapper
    return decorator

def get_cached_data(key: str) -> Optional[Tuple[Any, float]]:
    """
    Pobiera dane z cache'u.
    
    Args:
        key: Klucz cache'u
    
    Returns:
        Optional[Tuple[Any, float]]: Dane i timestamp lub None jeśli brak danych
    """
    if key in _CACHE:
        entry = _CACHE[key]
        return entry["data"], entry["timestamp"]
    return None

def store_cached_data(key: str, data: Any) -> None:
    """
    Zapisuje dane w cache'u.
    
    Args:
        key: Klucz cache'u
        data: Dane do zapisania
    """
    _CACHE[key] = {
        "data": data,
        "timestamp": time.time()
    }

def is_cache_valid(key: str, ttl: int = 30) -> bool:
    """
    Sprawdza czy dane w cache'u są aktualne.
    
    Args:
        key: Klucz cache'u
        ttl: Czas życia cache'u w sekundach
    
    Returns:
        bool: True jeśli dane są aktualne, False w przeciwnym przypadku
    """
    if key in _CACHE:
        return time.time() - _CACHE[key]["timestamp"] < ttl
    return False

def clear_cache() -> None:
    """Czyści cały cache."""
    global _CACHE
    _CACHE = {}
    logger.info("Cache cleared")

def clear_expired_cache(ttl: int = 30) -> int:
    """
    Czyści wygasłe dane z cache'u.
    
    Args:
        ttl: Czas życia cache'u w sekundach
    
    Returns:
        int: Liczba usuniętych wpisów
    """
    global _CACHE
    current_time = time.time()
    expired_keys = [
        key for key, entry in _CACHE.items() 
        if current_time - entry["timestamp"] > ttl
    ]
    
    for key in expired_keys:
        del _CACHE[key]
    
    if expired_keys:
        logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    return len(expired_keys)

"""
import_manager.py
-----------------
Moduł odpowiedzialny za dynamiczny import innych modułów w projekcie.
Funkcjonalności:
- Dynamiczne importowanie modułów na podstawie nazwy.
- Zarządzanie zależnościami i ładowaniem modułów w czasie wykonania.
- Implementacja mechanizmu cache, aby uniknąć wielokrotnego ładowania tych samych modułów.
- Sprawdzanie wersji pakietów i zgodności z wymaganiami projektu.
- Logowanie operacji importu oraz obsługa błędów (np. brak pakietu, konflikt wersji).
"""

import importlib
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ImportManager:
    def __init__(self):
        # Cache dla zaimportowanych modułów
        self.module_cache = {}

    def import_module(self, module_name: str):
        """
        Dynamicznie importuje moduł o podanej nazwie i zapisuje go w cache.

        Parameters:
            module_name (str): Nazwa modułu do zaimportowania.

        Returns:
            module: Zaimportowany moduł.

        Raises:
            ImportError: Jeśli moduł nie może zostać zaimportowany.
        """
        if module_name in self.module_cache:
            logging.info("Moduł '%s' pobrany z cache.", module_name)
            return self.module_cache[module_name]

        try:
            module = importlib.import_module(module_name)
            self.module_cache[module_name] = module
            logging.info("Moduł '%s' zaimportowany pomyślnie.", module_name)
            return module
        except ImportError as e:
            logging.error("Nie udało się zaimportować modułu '%s': %s", module_name, e)
            raise

    def check_version(self, module_name: str, required_version: str) -> bool:
        """
        Sprawdza, czy zaimportowany moduł spełnia wymagania wersji.

        Parameters:
            module_name (str): Nazwa modułu.
            required_version (str): Wymagana wersja.

        Returns:
            bool: True, jeśli wersja modułu jest zgodna z wymaganą, False w przeciwnym razie.
        """
        try:
            module = self.import_module(module_name)
            module_version = getattr(module, "__version__", None)
            if module_version is None:
                logging.warning("Moduł '%s' nie posiada atrybutu __version__.", module_name)
                return False
            if module_version == required_version:
                logging.info(
                    "Moduł '%s' spełnia wymagania wersji: %s.",
                    module_name,
                    required_version,
                )
                return True
            else:
                logging.warning(
                    "Wersja modułu '%s' (%s) nie jest zgodna z wymaganą (%s).",
                    module_name,
                    module_version,
                    required_version,
                )
                return False
        except ImportError:
            return False

    def clear_cache(self):
        """
        Czyści cache zaimportowanych modułów.
        """
        self.module_cache.clear()
        logging.info("Cache modułów został wyczyszczony.")


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        import_manager = ImportManager()
        # Przykładowy import modułu "json" (który jest częścią standardowej biblioteki)
        json_module = import_manager.import_module("json")
        logging.info("Przykładowy import modułu 'json' zakończony: %s", json_module)

        # Sprawdzenie wersji dla modułu "requests" (jeśli dostępne)
        try:
            version_ok = import_manager.check_version("requests", "2.25.1")
            logging.info("Wersja modułu 'requests' zgodna: %s", version_ok)
        except Exception as e:
            logging.warning("Nie udało się sprawdzić wersji modułu 'requests': %s", e)

        # Czyszczenie cache
        import_manager.clear_cache()
    except Exception as e:
        logging.error("Błąd w module import_manager.py: %s", e)
        raise

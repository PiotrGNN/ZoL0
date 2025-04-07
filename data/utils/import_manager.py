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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


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
                logging.warning(
                    "Moduł '%s' nie posiada atrybutu __version__.", module_name
                )
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
"""
import_manager.py
---------------
Moduł zarządzający importami i kontrolujący zależności w systemie.
"""

import logging
import sys
import importlib
import inspect
import os
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

# Lista modułów, które są wykluczone z automatycznego importu
EXCLUDED_MODULES = []

def exclude_modules_from_auto_import(modules: List[str]) -> None:
    """
    Dodaje moduły do listy wykluczonych z automatycznego importu.
    
    Args:
        modules (list): Lista nazw modułów do wykluczenia.
    """
    global EXCLUDED_MODULES
    EXCLUDED_MODULES.extend(modules)
    logger.debug(f"Dodano moduły do wykluczenia z auto-importu: {modules}")

def is_module_excluded(module_name: str) -> bool:
    """
    Sprawdza, czy moduł jest wykluczony z automatycznego importu.
    
    Args:
        module_name (str): Nazwa modułu do sprawdzenia.
        
    Returns:
        bool: True jeśli moduł jest wykluczony, False w przeciwnym razie.
    """
    return any(module_name.startswith(excluded) for excluded in EXCLUDED_MODULES)

def import_submodules(package_name: str) -> Dict[str, Any]:
    """
    Importuje wszystkie podmoduły pakietu rekursywnie.
    
    Args:
        package_name (str): Nazwa pakietu, z którego importujemy podmoduły.
        
    Returns:
        dict: Słownik zaimportowanych modułów.
    """
    package = importlib.import_module(package_name)
    results = {}
    
    # Sprawdzamy, czy pakiet ma atrybut __path__ (jest pakietem)
    if hasattr(package, '__path__'):
        # Uzyskanie pełnej ścieżki do pakietu
        package_path = package.__path__[0]
        
        # Iteracja po plikach w katalogu pakietu
        for loader, module_name, is_pkg in importlib.util.find_loader(package_name).get_loader_module_details():
            # Pomijamy __pycache__ i wykluczane moduły
            if module_name.startswith('__') or is_module_excluded(f"{package_name}.{module_name}"):
                continue
            
            # Budujemy pełną nazwę modułu
            full_name = f"{package_name}.{module_name}"
            
            try:
                # Importujemy moduł
                module = importlib.import_module(full_name)
                results[full_name] = module
                
                # Jeśli to pakiet, rekurencyjnie importujemy jego podmoduły
                if is_pkg:
                    submodules = import_submodules(full_name)
                    results.update(submodules)
            except Exception as e:
                logger.warning(f"Błąd podczas importowania modułu {full_name}: {str(e)}")
    
    return results

def find_implementations(base_class, package_name: str = "data") -> Dict[str, Any]:
    """
    Znajduje wszystkie implementacje określonej klasy bazowej w pakiecie.
    
    Args:
        base_class: Klasa bazowa, której implementacji szukamy.
        package_name (str): Nazwa pakietu, w którym szukamy.
        
    Returns:
        dict: Słownik znalezionych implementacji.
    """
    implementations = {}
    modules = import_submodules(package_name)
    
    for module_name, module in modules.items():
        # Sprawdzamy każdą klasę w module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Sprawdzamy, czy to subklasa naszej klasy bazowej i czy to nie jest sama klasa bazowa
            if issubclass(obj, base_class) and obj != base_class:
                implementations[name] = obj
    
    return implementations

def lazy_import(module_name: str) -> Callable:
    """
    Realizuje leniwe importowanie modułów.
    
    Args:
        module_name (str): Nazwa modułu do zaimportowania.
        
    Returns:
        callable: Funkcja importująca moduł na żądanie.
    """
    class LazyModule:
        def __init__(self, module_name):
            self.module_name = module_name
            self._module = None
            
        def __getattr__(self, name):
            if self._module is None:
                self._module = importlib.import_module(self.module_name)
            return getattr(self._module, name)
    
    return LazyModule(module_name)

def safe_import(module_name: str, error_handler: Optional[Callable] = None) -> Any:
    """
    Bezpiecznie importuje moduł z obsługą błędów.
    
    Args:
        module_name (str): Nazwa modułu do zaimportowania.
        error_handler (callable, optional): Funkcja obsługi błędów.
        
    Returns:
        Module or None: Zaimportowany moduł lub None w przypadku błędu.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        if error_handler:
            return error_handler(e, module_name)
        else:
            logger.error(f"Błąd importowania modułu {module_name}: {str(e)}")
            return None

def module_exists(module_name: str) -> bool:
    """
    Sprawdza, czy moduł istnieje bez jego importowania.
    
    Args:
        module_name (str): Nazwa modułu do sprawdzenia.
        
    Returns:
        bool: True jeśli moduł istnieje, False w przeciwnym razie.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError):
        return False
"""
import_manager.py
----------------
Moduł do zarządzania importami.
"""

import importlib
import logging
import os
import sys
from typing import List, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Zbiór modułów wykluczonych z automatycznego importu
excluded_modules: Set[str] = set()

def exclude_modules_from_auto_import(modules: List[str]) -> None:
    """
    Wyklucza moduły z automatycznego importu.
    
    Parameters:
        modules (List[str]): Lista modułów do wykluczenia.
    """
    global excluded_modules
    for module in modules:
        excluded_modules.add(module)
    logger.info(f"Wykluczono moduły z automatycznego importu: {modules}")

def is_module_excluded(module_name: str) -> bool:
    """
    Sprawdza, czy moduł jest wykluczony z automatycznego importu.
    
    Parameters:
        module_name (str): Nazwa modułu.
        
    Returns:
        bool: True jeśli moduł jest wykluczony, False w przeciwnym razie.
    """
    for excluded in excluded_modules:
        if module_name.startswith(excluded) or module_name == excluded:
            return True
    return False

def import_module_safely(module_name: str) -> None:
    """
    Bezpiecznie importuje moduł, obsługując wyjątki.
    
    Parameters:
        module_name (str): Nazwa modułu do importu.
    """
    if is_module_excluded(module_name):
        logger.debug(f"Pominięto import modułu '{module_name}' (wykluczony)")
        return
    
    try:
        importlib.import_module(module_name)
        logger.debug(f"Zaimportowano moduł '{module_name}'")
    except ImportError as e:
        logger.warning(f"Nie można zaimportować modułu '{module_name}': {e}")
    except Exception as e:
        logger.error(f"Błąd podczas importu modułu '{module_name}': {e}")

def auto_import_modules(package_name: str) -> None:
    """
    Automatycznie importuje wszystkie moduły z pakietu.
    
    Parameters:
        package_name (str): Nazwa pakietu.
    """
    try:
        package = importlib.import_module(package_name)
        package_path = os.path.dirname(package.__file__)
        
        for item in os.listdir(package_path):
            if item.startswith("__") or not item.endswith(".py"):
                continue
            
            module_name = f"{package_name}.{item[:-3]}"
            if not is_module_excluded(module_name):
                import_module_safely(module_name)
        
        logger.info(f"Automatycznie zaimportowano moduły z pakietu '{package_name}'")
    except Exception as e:
        logger.error(f"Błąd podczas automatycznego importu modułów z pakietu '{package_name}': {e}")

if __name__ == "__main__":
    # Przykład użycia
    exclude_modules_from_auto_import(["tests"])
    auto_import_modules("data.utils")
"""
import_manager.py
---------------
Moduł do zarządzania importami w projekcie.
"""

import importlib
import logging
import os
import pkgutil
import sys
from typing import List, Dict, Any, Set

# Lista modułów wykluczonych z automatycznego importowania
excluded_modules: Set[str] = set()

def exclude_modules_from_auto_import(modules: List[str]) -> None:
    """
    Dodaje moduły do listy wykluczonych z automatycznego importowania.

    Args:
        modules (List[str]): Lista nazw modułów do wykluczenia
    """
    global excluded_modules
    for module in modules:
        excluded_modules.add(module)
    logging.info(f"Moduły wykluczone z automatycznego importu: {excluded_modules}")

def import_submodules(package_name: str) -> Dict[str, Any]:
    """
    Importuje wszystkie podmoduły pakietu.

    Args:
        package_name (str): Nazwa pakietu

    Returns:
        Dict[str, Any]: Słownik zaimportowanych modułów
    """
    results = {}
    
    try:
        package = importlib.import_module(package_name)
        
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            # Pomijamy wykluczone moduły
            module_short_name = name.split('.')[-1]
            if module_short_name in excluded_modules:
                continue
                
            try:
                results[name] = importlib.import_module(name)
                if is_pkg:
                    submodules = import_submodules(name)
                    results.update(submodules)
            except Exception as e:
                logging.warning(f"Błąd podczas importowania {name}: {e}")
                
    except Exception as e:
        logging.error(f"Błąd podczas importowania pakietu {package_name}: {e}")
        
    return results

def reload_module(module_name: str) -> bool:
    """
    Przeładowuje wskazany moduł.

    Args:
        module_name (str): Nazwa modułu do przeładowania

    Returns:
        bool: True jeśli przeładowanie się powiodło, False w przeciwnym wypadku
    """
    try:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            logging.info(f"Przeładowano moduł {module_name}")
            return True
        else:
            importlib.import_module(module_name)
            logging.info(f"Zaimportowano moduł {module_name}")
            return True
    except Exception as e:
        logging.error(f"Błąd podczas przeładowywania modułu {module_name}: {e}")
        return False

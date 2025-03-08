"""
Inicjalizacja pakietu 'data'.

Ten plik __init__.py automatycznie importuje wszystkie moduły (.py) znajdujące się w bieżącym katalogu
oraz wszystkie podpakiety (foldery zawierające __init__.py), z wyłączeniem tych przeznaczonych wyłącznie do testów.
Dodatkowo re-eksportuje wybrane moduły z podpakietu 'data' (czyli z folderu data/data),
umożliwiając import:
    from data import HistoricalDataManager, data_preprocessing
bez konieczności znajomości wewnętrznej struktury katalogów.
"""

import os
import importlib
from typing import List

# Wymuszenie importu standardowego modułu logowania jako 'py_logging'
py_logging = importlib.import_module("logging")

__all__: List[str] = []

# Lista podpakietów do wykluczenia z automatycznego importu
EXCLUDE_SUBPACKAGES = {"tests"}


def _import_all_modules_from_directory(directory: str, package: str) -> None:
    """
    Importuje wszystkie moduły .py z danego katalogu (pomijając __init__.py)
    i dodaje ich nazwy do listy __all__.
    
    Args:
        directory: Ścieżka do katalogu, w którym szukamy plików .py.
        package: Nazwa pakietu używana przy importach względnych.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name: str = filename[:-3]
            try:
                module = importlib.import_module(f".{module_name}", package=package)
            except Exception as error:
                py_logging.warning(
                    "Pomijam moduł '%s' z powodu błędu: %s", module_name, error
                )
                continue
            globals()[module_name] = module
            __all__.append(module_name)


def _import_all_subpackages(directory: str, package: str) -> None:
    """
    Importuje wszystkie podkatalogi, które są pakietami (zawierają __init__.py),
    i dodaje ich nazwy do listy __all__, z wyłączeniem określonych folderów.
    
    Args:
        directory: Ścieżka do katalogu, w którym szukamy podpakietów.
        package: Nazwa pakietu używana przy importach względnych.
    """
    for item in os.listdir(directory):
        if item in EXCLUDE_SUBPACKAGES:
            py_logging.info("Wykluczam podpakiet '%s' z automatycznego importu.", item)
            continue
        subdir: str = os.path.join(directory, item)
        if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "__init__.py")):
            try:
                subpackage = importlib.import_module(f".{item}", package=package)
            except Exception as error:
                py_logging.warning(
                    "Pomijam podpakiet '%s' z powodu błędu: %s", item, error
                )
                continue
            globals()[item] = subpackage
            __all__.append(item)


def _import_all(directory: str, package: str) -> None:
    """
    Importuje wszystkie moduły i podpakiety z danego katalogu.
    
    Args:
        directory: Ścieżka do katalogu, z którego mają być importowane składniki.
        package: Nazwa pakietu używana przy importach względnych.
    """
    _import_all_modules_from_directory(directory, package)
    _import_all_subpackages(directory, package)


_current_dir: str = os.path.dirname(os.path.abspath(__file__))
_import_all(_current_dir, __name__)

# Re-export wybranych modułów z podpakietu "data" (czyli folderu data/data)
# Pozwala to na importowanie np. HistoricalDataManager oraz data_preprocessing
try:
    from .data.historical_data import HistoricalDataManager
    globals()["HistoricalDataManager"] = HistoricalDataManager
    if "HistoricalDataManager" not in __all__:
        __all__.append("HistoricalDataManager")
except ImportError as error:
    py_logging.warning("Nie udało się re-eksportować HistoricalDataManager: %s", error)

try:
    from .data import data_preprocessing
    globals()["data_preprocessing"] = data_preprocessing
    if "data_preprocessing" not in __all__:
        __all__.append("data_preprocessing")
except ImportError as error:
    py_logging.warning("Nie udało się re-eksportować data_preprocessing: %s", error)

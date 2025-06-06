"""
Inicjalizacja pakietu.

Ten plik __init__.py automatycznie importuje wszystkie moduły (.py) znajdujące się w bieżącym katalogu
(bez __init__.py) oraz wszystkie podpakiety (czyli katalogi zawierające plik __init__.py).
Dzięki temu, importując pakiet, masz dostęp do wszystkich jego składników poprzez:
    from package_name import *
oraz bez konieczności znajomości dokładnej struktury katalogów.
"""

import os
import importlib
from typing import List

__all__: List[str] = []


def _import_all_modules_from_directory(directory: str, package: str) -> None:
    """
    Importuje wszystkie moduły .py z danego katalogu (z wyłączeniem __init__.py)
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
                raise ImportError(
                    f"Nie udało się zaimportować modułu '{module_name}' z pakietu '{package}'."
                ) from error
            globals()[module_name] = module
            __all__.append(module_name)


def _import_all_subpackages(directory: str, package: str) -> None:
    """
    Importuje wszystkie podkatalogi, które są pakietami (zawierają __init__.py),
    i dodaje ich nazwy do listy __all__.

    Args:
        directory: Ścieżka do katalogu, w którym szukamy podpakietów.
        package: Nazwa pakietu używana przy importach względnych.
    """
    for item in os.listdir(directory):
        subdir: str = os.path.join(directory, item)
        if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "__init__.py")):
            try:
                subpackage = importlib.import_module(f".{item}", package=package)
            except Exception as error:
                raise ImportError(
                    f"Nie udało się zaimportować podpakietu '{item}' z pakietu '{package}'."
                ) from error
            globals()[item] = subpackage
            __all__.append(item)


def _import_all(directory: str, package: str) -> None:
    """
    Importuje wszystkie moduły i podpakiety z podanego katalogu.

    Args:
        directory: Ścieżka do katalogu, z którego mają być importowane składniki.
        package: Nazwa pakietu używana przy importach względnych.
    """
    _import_all_modules_from_directory(directory, package)
    _import_all_subpackages(directory, package)


_current_dir: str = os.path.dirname(os.path.abspath(__file__))
_import_all(_current_dir, __name__)

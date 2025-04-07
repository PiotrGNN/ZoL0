"""
import_manager.py
----------------
Moduł zarządzający importami pakietów i modułów w projekcie.
"""

import importlib
import logging
import os
import sys
from typing import List, Dict, Any, Optional, Set

def import_module_safe(module_path: str) -> Optional[Any]:
    """
    Bezpiecznie importuje moduł, obsługując ewentualne błędy.

    Parameters:
        module_path (str): Ścieżka do modułu.

    Returns:
        Optional[Any]: Zaimportowany moduł lub None w przypadku błędu.
    """
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        logging.warning(f"Nie udało się zaimportować modułu {module_path}: {e}")
        return None

def import_package_modules(package_path: str, exclude_modules: List[str] = None) -> Dict[str, Any]:
    """
    Importuje wszystkie moduły z pakietu.

    Parameters:
        package_path (str): Ścieżka do pakietu.
        exclude_modules (List[str], optional): Lista modułów do wykluczenia.

    Returns:
        Dict[str, Any]: Słownik zaimportowanych modułów.
    """
    if exclude_modules is None:
        exclude_modules = []

    modules = {}

    try:
        package = importlib.import_module(package_path)
        package_dir = os.path.dirname(package.__file__)

        for file in os.listdir(package_dir):
            if file.endswith(".py") and not file.startswith("__"):
                module_name = file[:-3]

                if module_name in exclude_modules:
                    logging.info(f"Pomijam moduł {module_name} (na liście wykluczeń)")
                    continue

                full_module_path = f"{package_path}.{module_name}"
                module = import_module_safe(full_module_path)

                if module:
                    modules[module_name] = module

        return modules
    except Exception as e:
        logging.error(f"Błąd podczas importowania modułów z pakietu {package_path}: {e}")
        return {}

def import_subpackages(base_package: str, exclude_packages: List[str] = None) -> Dict[str, Any]:
    """
    Importuje wszystkie podpakiety z pakietu bazowego.

    Parameters:
        base_package (str): Nazwa pakietu bazowego.
        exclude_packages (List[str], optional): Lista pakietów do wykluczenia.

    Returns:
        Dict[str, Any]: Słownik zaimportowanych pakietów.
    """
    if exclude_packages is None:
        exclude_packages = []

    packages = {}

    try:
        package = importlib.import_module(base_package)
        base_dir = os.path.dirname(package.__file__)

        for item in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, item)
            init_file = os.path.join(dir_path, "__init__.py")

            if os.path.isdir(dir_path) and os.path.exists(init_file) and item not in exclude_packages:
                try:
                    full_package_path = f"{base_package}.{item}"
                    subpackage = importlib.import_module(full_package_path)
                    packages[item] = subpackage
                except Exception as e:
                    logging.warning(f"Pomijam podpakiet '{item}' z powodu błędu: {e}")

        return packages
    except Exception as e:
        logging.error(f"Błąd podczas importowania podpakietów z {base_package}: {e}")
        return {}

def exclude_modules_from_auto_import(modules: List[str]) -> None:
    """
    Dodaje moduły do listy wykluczeń z automatycznego importu.

    Parameters:
        modules (List[str]): Lista modułów do wykluczenia.
    """
    for module in modules:
        logging.info(f"Wykluczam podpakiet '{module}' z automatycznego importu.")

def reload_module(module_path: str) -> bool:
    """
    Przeładowuje moduł.

    Parameters:
        module_path (str): Ścieżka do modułu.

    Returns:
        bool: True jeśli przeładowanie się powiodło, False w przeciwnym razie.
    """
    try:
        module = importlib.import_module(module_path)
        importlib.reload(module)
        logging.info(f"Moduł {module_path} został przeładowany")
        return True
    except Exception as e:
        logging.error(f"Błąd podczas przeładowywania modułu {module_path}: {e}")
        return False

def add_to_path(path: str) -> None:
    """
    Dodaje ścieżkę do sys.path.

    Parameters:
        path (str): Ścieżka do dodania.
    """
    if path not in sys.path:
        sys.path.append(path)
        logging.info(f"Dodano ścieżkę {path} do sys.path")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Zbiór modułów wykluczonych z automatycznego importu
excluded_modules: Set[str] = set()

def exclude_modules(modules: List[str]) -> None:
    global excluded_modules
    excluded_modules.update(modules)
    logging.info(f"Wykluczone moduły: {excluded_modules}")


def is_module_excluded(module_name: str) -> bool:
    return any(module_name.startswith(excluded) for excluded in excluded_modules)


if __name__ == "__main__":
    # Przykład użycia
    exclude_modules(["tests"])
    imported_modules = import_package_modules("data.utils", excluded_modules)
    print(f"Zaimportowane moduły: {imported_modules}")
    imported_subpackages = import_subpackages("data", ["tests"])
    print(f"Zaimportowane podpakiety: {imported_subpackages}")
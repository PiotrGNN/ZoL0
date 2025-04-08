"""
Inicjalizacja głównego pakietu data.
Zapewnia prawidłowe importowanie wszystkich podpakietów.
"""

import os
import importlib
import logging
from typing import List, Dict, Any

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# Lista wszystkich podpakietów
__all__: List[str] = []

# Mapowanie pakietów do ich aliasów do użycia w importach
PACKAGE_ALIASES: Dict[str, str] = {
    "data": "data_package",  # Unikamy konfliktu nazw data.data
    "execution": "execution",
    "indicators": "indicators",
    "logging": "log_package",  # Unikamy konfliktu z modułem logging
    "optimization": "optimization",
    "risk_management": "risk_management",
    "strategies": "strategies",
    "tests": "tests",
    "utils": "utils"
}

def import_subpackage(subpackage_name: str) -> Any:
    """
    Importuje podpakiet i obsługuje błędy.

    Args:
        subpackage_name: Nazwa podpakietu do zaimportowania

    Returns:
        Zaimportowany moduł lub None w przypadku błędu
    """
    try:
        # Użyj aliasu, aby uniknąć konfliktów nazw
        alias = PACKAGE_ALIASES.get(subpackage_name, subpackage_name)

        # Pełna ścieżka importu
        import_path = f"data.{subpackage_name}"

        # Spróbuj zaimportować podpakiet
        module = importlib.import_module(import_path)

        # Dodaj do globals() pod aliasem
        globals()[alias] = module

        # Dodaj alias do __all__
        if alias not in __all__:
            __all__.append(alias)

        logger.info(f"Zaimportowano podpakiet {import_path} jako {alias}")
        return module

    except ImportError as e:
        logger.warning(f"Pomijam podpakiet {subpackage_name} z powodu błędu: {e}")
        return None
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas importowania {subpackage_name}: {e}")
        return None

# Importuj wszystkie podpakiety
current_dir = os.path.dirname(os.path.abspath(__file__))
for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)

    # Sprawdź czy to katalog i czy zawiera __init__.py (podpakiet)
    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "__init__.py")):
        import_subpackage(item)

# Dodaj bezpośrednio dostępne funkcje i klasy
def get_available_subpackages() -> List[str]:
    """Zwraca listę dostępnych podpakietów."""
    return __all__

# Re-export wybranych modułów z podpakietu "data" (czyli folderu data/data)
# Pozwala to na importowanie np. HistoricalDataManager oraz data_preprocessing
try:
    from .data.historical_data import HistoricalDataManager
    globals()["HistoricalDataManager"] = HistoricalDataManager
    if "HistoricalDataManager" not in __all__:
        __all__.append("HistoricalDataManager")
except ImportError as error:
    logger.warning("Nie udało się re-eksportować HistoricalDataManager: %s", error)

try:
    from .data import data_preprocessing
    globals()["data_preprocessing"] = data_preprocessing
    if "data_preprocessing" not in __all__:
        __all__.append("data_preprocessing")
except ImportError as error:
    logger.warning("Nie udało się re-eksportować data_preprocessing: %s", error)

"""
Config Loader - moduł do ładowania konfiguracji z różnych źródeł:
- plik .env
- Replit Secrets
- pliki JSON/YAML
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import load_dotenv

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/config.log"), logging.StreamHandler()],
)
logger = logging.getLogger("ConfigLoader")


class ConfigLoader:
    """
    Klasa do ładowania i zarządzania konfiguracją z różnych źródeł.
    
    Obsługuje:
    - .env
    - Replit Secrets
    - pliki YAML/JSON
    - zmienne środowiskowe
    """

    def __init__(
        self,
        env_file: str = ".env",
        config_file: Optional[str] = None,
        use_replit_secrets: bool = True,
        default_config: Optional[Dict] = None,
    ):
        """
        Inicjalizacja loadera konfiguracji.
        
        Args:
            env_file: Ścieżka do pliku .env
            config_file: Ścieżka do pliku konfiguracyjnego (YAML/JSON)
            use_replit_secrets: Czy używać Replit Secrets
            default_config: Domyślna konfiguracja (używana jeśli nie znaleziono pliku)
        """
        self.env_file = env_file
        self.config_file = config_file
        self.use_replit_secrets = use_replit_secrets
        self.default_config = default_config or {}
        
        # Słownik konfiguracji
        self.config = {}
        
        # Ładujemy konfigurację
        self._load_config()
        
        logger.info("ConfigLoader zainicjalizowany")

    def _load_config(self) -> None:
        """
        Ładuje konfigurację z wszystkich źródeł.
        """
        # 1. Zacznij od domyślnej konfiguracji
        self.config = self.default_config.copy()
        
        # 2. Załaduj zmienne z .env
        self._load_env_file()
        
        # 3. Załaduj plik konfiguracyjny (JSON/YAML)
        if self.config_file:
            self._load_config_file()
            
        # 4. Załaduj Replit Secrets (nadpisuje poprzednie)
        if self.use_replit_secrets:
            self._load_replit_secrets()
            
        # 5. Załaduj zmienne środowiskowe (nadpisują poprzednie)
        self._load_environment_variables()
        
        logger.info(f"Załadowano konfigurację: {len(self.config)} zmiennych")

    def _load_env_file(self) -> None:
        """
        Ładuje zmienne z pliku .env do konfiguracji.
        """
        try:
            if os.path.exists(self.env_file):
                load_dotenv(self.env_file)
                logger.info(f"Załadowano zmienne z pliku {self.env_file}")
            else:
                logger.warning(f"Plik .env nie istnieje: {self.env_file}")
        except Exception as e:
            logger.error(f"Błąd podczas ładowania pliku .env: {e}")

    def _load_config_file(self) -> None:
        """
        Ładuje konfigurację z pliku JSON/YAML.
        """
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"Plik konfiguracyjny nie istnieje: {self.config_file}")
                return
                
            # Określ format na podstawie rozszerzenia
            file_ext = os.path.splitext(self.config_file)[1].lower()
            
            with open(self.config_file, "r") as f:
                if file_ext in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                elif file_ext == ".json":
                    config_data = json.load(f)
                else:
                    logger.warning(f"Nieobsługiwany format pliku konfiguracyjnego: {file_ext}")
                    return
                    
                # Aktualizuj konfigurację
                if isinstance(config_data, dict):
                    self.config.update(self._flatten_dict(config_data))
                    logger.info(f"Załadowano konfigurację z pliku {self.config_file}")
                else:
                    logger.warning(f"Plik konfiguracyjny nie zawiera słownika: {self.config_file}")
                    
        except Exception as e:
            logger.error(f"Błąd podczas ładowania pliku konfiguracyjnego: {e}")

    def _load_replit_secrets(self) -> None:
        """
        Ładuje zmienne z Replit Secrets do konfiguracji.
        """
        try:
            # Sprawdź, czy jesteśmy w środowisku Replit
            replit_db_url = os.environ.get("REPLIT_DB_URL")
            if not replit_db_url:
                logger.info("Nie wykryto środowiska Replit, pomijam ładowanie Replit Secrets")
                return
                
            # Wykryto środowisko Replit, spróbuj załadować sekrety
            try:
                from replit import db as replit_db
                
                # Pobierz wszystkie sekrety zaczynające się od przedrostka 'SECRET_'
                for key in replit_db.keys():
                    if key.startswith("SECRET_"):
                        # Usuń przedrostek 'SECRET_' i zapisz pod oryginalną nazwą
                        config_key = key[7:]
                        self.config[config_key] = replit_db[key]
                        
                logger.info("Załadowano zmienne z Replit Secrets")
                
            except ImportError:
                logger.warning("Moduł 'replit' nie jest zainstalowany, pomijam ładowanie Replit Secrets")
                
        except Exception as e:
            logger.error(f"Błąd podczas ładowania Replit Secrets: {e}")

    def _load_environment_variables(self) -> None:
        """
        Ładuje zmienne środowiskowe do konfiguracji.
        """
        # Iteruj po wszystkich zmiennych środowiskowych
        for key, value in os.environ.items():
            # Dodaj do konfiguracji
            self.config[key] = value
            
        logger.debug("Załadowano zmienne środowiskowe")

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """
        Spłaszcza zagnieżdżony słownik.
        
        Args:
            d: Słownik do spłaszczenia
            parent_key: Klucz rodzica
            sep: Separator między kluczami
            
        Returns:
            Dict: Spłaszczony słownik
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Pobiera wartość konfiguracji.
        
        Args:
            key: Klucz konfiguracji
            default: Domyślna wartość jeśli klucz nie istnieje
            
        Returns:
            Any: Wartość konfiguracji lub domyślna wartość
        """
        # Najpierw sprawdź, czy klucz istnieje w konfiguracji
        if key in self.config:
            return self.config[key]
            
        # Następnie sprawdź zmienne środowiskowe
        if key in os.environ:
            return os.environ[key]
            
        # Zwróć wartość domyślną
        return default

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Pobiera wartość konfiguracji jako int.
        
        Args:
            key: Klucz konfiguracji
            default: Domyślna wartość jeśli klucz nie istnieje lub nie można skonwertować
            
        Returns:
            int or None: Wartość konfiguracji jako int lub domyślna wartość
        """
        value = self.get(key, default)
        if value is None:
            return default
            
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Nie można skonwertować '{key}' do int: {value}")
            return default

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        Pobiera wartość konfiguracji jako float.
        
        Args:
            key: Klucz konfiguracji
            default: Domyślna wartość jeśli klucz nie istnieje lub nie można skonwertować
            
        Returns:
            float or None: Wartość konfiguracji jako float lub domyślna wartość
        """
        value = self.get(key, default)
        if value is None:
            return default
            
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Nie można skonwertować '{key}' do float: {value}")
            return default

    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        Pobiera wartość konfiguracji jako bool.
        
        Args:
            key: Klucz konfiguracji
            default: Domyślna wartość jeśli klucz nie istnieje
            
        Returns:
            bool or None: Wartość konfiguracji jako bool lub domyślna wartość
        """
        value = self.get(key, default)
        if value is None:
            return default
            
        if isinstance(value, bool):
            return value
            
        if isinstance(value, (int, float)):
            return bool(value)
            
        if isinstance(value, str):
            return value.lower() in ["true", "yes", "y", "1", "t"]
            
        return default

    def get_list(
        self, key: str, default: Optional[list] = None, separator: str = ","
    ) -> Optional[list]:
        """
        Pobiera wartość konfiguracji jako listę.
        
        Args:
            key: Klucz konfiguracji
            default: Domyślna wartość jeśli klucz nie istnieje
            separator: Separator dla wartości tekstowych
            
        Returns:
            list or None: Wartość konfiguracji jako lista lub domyślna wartość
        """
        value = self.get(key, default)
        if value is None:
            return default
            
        if isinstance(value, list):
            return value
            
        if isinstance(value, str):
            return [item.strip() for item in value.split(separator)]
            
        return [value]

    def get_dict(self, key: str, default: Optional[Dict] = None) -> Optional[Dict]:
        """
        Pobiera wartość konfiguracji jako słownik.
        
        Args:
            key: Klucz konfiguracji
            default: Domyślna wartość jeśli klucz nie istnieje
            
        Returns:
            dict or None: Wartość konfiguracji jako słownik lub domyślna wartość
        """
        value = self.get(key, default)
        if value is None:
            return default
            
        if isinstance(value, dict):
            return value
            
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Nie można sparsować '{key}' jako JSON: {value}")
                return default
                
        return default

    def set(self, key: str, value: Any) -> None:
        """
        Ustawia wartość konfiguracji.
        
        Args:
            key: Klucz konfiguracji
            value: Wartość konfiguracji
        """
        self.config[key] = value
        logger.debug(f"Ustawiono wartość konfiguracji: {key}={value}")

    def to_dict(self) -> Dict:
        """
        Zwraca całą konfigurację jako słownik.
        
        Returns:
            Dict: Słownik konfiguracji
        """
        return self.config.copy()

    def save_to_file(self, file_path: Optional[str] = None, format: str = "yaml") -> bool:
        """
        Zapisuje konfigurację do pliku.
        
        Args:
            file_path: Ścieżka do pliku (domyślnie self.config_file)
            format: Format pliku ('yaml' lub 'json')
            
        Returns:
            bool: True jeśli zapisano pomyślnie, False w przeciwnym razie
        """
        file_path = file_path or self.config_file
        if not file_path:
            logger.error("Nie podano ścieżki do pliku konfiguracyjnego")
            return False
            
        try:
            with open(file_path, "w") as f:
                if format.lower() == "yaml":
                    yaml.dump(self.config, f, default_flow_style=False)
                elif format.lower() == "json":
                    json.dump(self.config, f, indent=2)
                else:
                    logger.error(f"Nieobsługiwany format pliku: {format}")
                    return False
                    
            logger.info(f"Zapisano konfigurację do pliku {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania konfiguracji do pliku: {e}")
            return False


# Inicjalizacja globalnej instancji ConfigLoader
config = ConfigLoader()


# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja lodera
    config_loader = ConfigLoader(
        env_file=".env",
        config_file="config/settings.yml",
        use_replit_secrets=True,
    )
    
    # Przykłady pobierania wartości
    api_key = config_loader.get("BYBIT_API_KEY", "default_key")
    test_mode = config_loader.get_bool("TEST_MODE", True)
    risk_level = config_loader.get("RISK_LEVEL", "low")
    
    print(f"API Key: {api_key}")
    print(f"Test Mode: {test_mode}")
    print(f"Risk Level: {risk_level}")
    
    # Przykład zapisywania wartości
    config_loader.set("CUSTOM_SETTING", "test_value")
    
    # Pobieranie całej konfiguracji
    all_config = config_loader.to_dict()
    print(f"Cała konfiguracja: {all_config}")

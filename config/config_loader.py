"""
config_loader.py
----------------
Moduł do ładowania konfiguracji z różnych formatów (JSON, YAML, INI) oraz zmiennych środowiskowych.
Obsługuje cache, walidację i priorytet ładowania.
"""

import os
import json
import yaml
import configparser
import logging
from dotenv import load_dotenv, dotenv_values
from typing import Any, Dict, List, Optional

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ConfigLoader:
    """
    Klasa do ładowania konfiguracji z plików (JSON, YAML, INI) i zmiennych środowiskowych.
    Obsługuje priorytetowe ładowanie danych oraz cache.
    """

    def __init__(self, config_files: Optional[List[str]] = None, env_prefix: str = "", cache_enabled: bool = True):
        self.config_files = config_files or []
        self.env_prefix = env_prefix
        self.cache_enabled = cache_enabled
        self._config_cache: Optional[Dict[str, Any]] = None

    def load(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Ładuje konfigurację z plików i zmiennych środowiskowych, scala je i waliduje."""
        if self.cache_enabled and self._config_cache is not None and not force_refresh:
            logging.info("Ładowanie konfiguracji z cache.")
            return self._config_cache

        config: Dict[str, Any] = {}

        # Ładowanie plików konfiguracyjnych
        for filepath in self.config_files:
            try:
                file_config = self._load_from_file(filepath)
                logging.info("Załadowano konfigurację z pliku: %s", filepath)
                config = self._merge_configs(config, file_config)
            except Exception as e:
                logging.error("Błąd przy ładowaniu pliku %s: %s", filepath, e)

        # Ładowanie zmiennych środowiskowych
        env_config = self._load_from_env()
        if env_config:
            logging.info("Załadowano konfigurację ze zmiennych środowiskowych.")
            config = self._merge_configs(config, env_config)

        # Normalizacja kluczy (małe litery)
        config = {key.lower(): value for key, value in config.items()}

        # Usunięcie duplikatów
        config = self._remove_duplicates(config)

        self.validate_config(config)

        if self.cache_enabled:
            self._config_cache = config

        return config

    def _load_from_file(self, filepath: str) -> Dict[str, Any]:
        """Ładuje konfigurację z plików JSON, YAML lub INI."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Plik konfiguracyjny nie istnieje: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                if filepath.lower().endswith((".yaml", ".yml")):
                    return yaml.safe_load(f) or {}
                elif filepath.lower().endswith(".json"):
                    return json.load(f) or {}
                elif filepath.lower().endswith(".ini"):
                    config = configparser.ConfigParser()
                    config.read_file(f)
                    return {section: dict(config.items(section)) for section in config.sections()}
                else:
                    raise ValueError("Nieobsługiwany format pliku. Obsługiwane formaty: JSON, YAML, INI.")
        except (json.JSONDecodeError, yaml.YAMLError, configparser.Error) as e:
            logging.error("Błąd parsowania pliku konfiguracyjnego %s: %s", filepath, e)
            return {}

    def _load_from_env(self) -> Dict[str, Any]:
        """Ładuje zmienne środowiskowe zaczynające się od podanego prefiksu."""
        config: Dict[str, Any] = {}

        # Próba załadowania .env (jeśli istnieje)
        if os.path.exists(".env"):
            try:
                dotenv_values(".env")
                load_dotenv()
                logging.info("Załadowano plik .env")
            except Exception as e:
                logging.warning("Nie udało się poprawnie sparsować .env: %s", e)

        prefix = self.env_prefix.upper()
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lstrip("_").lower()
                config[config_key] = self._cast_value(value)
        return config

    def _cast_value(self, value: str) -> Any:
        """Konwertuje wartość zmiennej środowiskowej na int, float lub bool."""
        if not value:
            return None
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                continue
        return value

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Scala dwie konfiguracje, gdzie wartości z 'override' nadpisują te z 'base'."""
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _remove_duplicates(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Usuwa duplikaty kluczy (np. DATABASE i database)."""
        keys_to_remove = [key for key in config if key.lower() in config and key.upper() in config]
        for key in keys_to_remove:
            logging.warning(f"Usuwanie duplikatu konfiguracji: {key}")
            del config[key]
        return config

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Waliduje konfigurację i dodaje brakujące sekcje z wartościami domyślnymi."""
        required_defaults = {
            "database": {"host": "localhost", "port": 5432, "user": "user", "password": "password", "name": "default_db"},
            "trading": {"commission": 0.001, "spread": 0.0005, "slippage": 0.0005},
        }

        for key, default_value in required_defaults.items():
            if key not in config:
                logging.warning("Brak sekcji '%s'. Dodajemy domyślną konfigurację.", key)
                config[key] = default_value

        logging.info("Walidacja konfiguracji zakończona sukcesem.")


# -------------------- Testy jednostkowe --------------------
if __name__ == "__main__":
    loader = ConfigLoader(config_files=["config.json", "config.yaml"], env_prefix="MYAPP")
    config = loader.load()
    print(json.dumps(config, indent=4, ensure_ascii=False))

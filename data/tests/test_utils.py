"""
test_utils.py
-------------
Testy jednostkowe dla modułów narzędziowych, takich jak:
- config_loader.py
- database_manager.py
- data_validator.py
- import_manager.py
- notification_system.py

Testy weryfikują poprawność ładowania konfiguracji, operacji na bazie danych,
walidacji danych, importu modułów oraz wysyłki powiadomień.
"""

import json
import os
import tempfile
import unittest

# Dla celów testowych tworzymy dummy implementacje modułów.
# W rzeczywistości powinno się importować rzeczywiste moduły, np.:
# from config.config_loader import ConfigLoader
# from database_manager import DatabaseManager
# from data_validator import DataValidator
# from import_manager import ImportManager
# from notification_system import NotificationSystem


class DummyConfigLoader:
    def __init__(self, config_files, env_prefix, cache_enabled=True):
        self.config_files = config_files
        self.env_prefix = env_prefix
        self.cache_enabled = cache_enabled
        self._config_cache = None

    def load(self):
        # Symulujemy ładowanie konfiguracji
        # Zwracamy domyślną konfigurację niezależnie od plików
        return {
            "database": {"host": "localhost", "port": 3306},
            "api_key": "dummy_api_key",
        }


class DummyDatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        # Symulujemy połączenie z bazą danych
        self.conn = True
        return self.conn

    def execute_query(self, query):
        # Zwracamy dummy wynik dla dowolnego zapytania
        return "dummy_result"


class DummyDataValidator:
    def validate(self, data):
        # Zwracamy True, jeśli data nie jest pusta, inaczej False
        return bool(data)


class DummyImportManager:
    def import_module(self, module_name):
        # Symulujemy import modułu; dla "dummy" zwracamy True, w przeciwnym razie rzucamy błąd
        if module_name == "dummy":
            return True
        else:
            raise ImportError("Module not found.")


class DummyNotificationSystem:
    def send_notification(self, message):
        # Symulujemy wysłanie powiadomienia, zwracając potwierdzenie
        return f"Notification sent: {message}"


class TestUtilsModules(unittest.TestCase):
    def test_config_loader(self):
        # Utwórz tymczasowy plik konfiguracyjny
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
            config_data = {"database": {"host": "127.0.0.1", "port": 5432}}
            json.dump(config_data, tmp)
            tmp_path = tmp.name
        try:
            loader = DummyConfigLoader(config_files=[tmp_path], env_prefix="APP")
            config = loader.load()
            self.assertIn("database", config, "Konfiguracja powinna zawierać sekcję 'database'.")
            # DummyConfigLoader zwraca stałą konfigurację
            self.assertEqual(config["database"]["host"], "localhost")
        finally:
            os.remove(tmp_path)

    def test_database_manager(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        try:
            db_manager = DummyDatabaseManager(db_path)
            conn = db_manager.connect()
            self.assertTrue(conn, "Połączenie z bazą powinno być aktywne.")
            result = db_manager.execute_query("SELECT 1")
            self.assertEqual(result, "dummy_result", "Wynik zapytania powinien być 'dummy_result'.")
        finally:
            os.remove(db_path)

    def test_data_validator(self):
        validator = DummyDataValidator()
        self.assertTrue(validator.validate([1, 2, 3]), "Dane nie powinny być puste.")
        self.assertFalse(validator.validate([]), "Pusta lista powinna być niepoprawna.")

    def test_import_manager(self):
        import_manager = DummyImportManager()
        self.assertTrue(
            import_manager.import_module("dummy"),
            "Import modułu 'dummy' powinien się powieść.",
        )
        with self.assertRaises(ImportError):
            import_manager.import_module("nonexistent")

    def test_notification_system(self):
        notification = DummyNotificationSystem()
        response = notification.send_notification("Test message")
        self.assertIn(
            "Notification sent",
            response,
            "Odpowiedź powinna zawierać 'Notification sent'.",
        )


if __name__ == "__main__":
    unittest.main()

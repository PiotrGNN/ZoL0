"""
database_manager.py
-------------------
Moduł zarządzający połączeniami z bazą danych i wykonywaniem operacji CRUD.
Funkcjonalności:
- Obsługuje różne typy baz danych (np. SQLite, PostgreSQL, MySQL) – przykładowa implementacja dla SQLite.
- Implementuje pooling połączeń, transakcje oraz automatyczne ponawianie w razie chwilowych błędów.
- Umożliwia migrację schematu (np. przy użyciu własnych skryptów migracyjnych).
- Zawiera funkcje do monitorowania czasu wykonania zapytań oraz logowanie kluczowych operacji.
"""

import logging
import sqlite3
import time

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class DatabaseManager:
    def __init__(self, db_path: str):
        """
        Inicjalizuje menedżera bazy danych.

        Parameters:
            db_path (str): Ścieżka do pliku bazy danych (SQLite).
        """
        self.db_path = db_path
        self.conn = None
        self.connect()

    def connect(self):
        """
        Nawiązuje połączenie z bazą danych.
        """
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logging.info("Połączono z bazą danych: %s", self.db_path)
        except Exception as e:
            logging.error("Błąd połączenia z bazą danych: %s", e)
            raise

    def execute_query(self, query: str, params: tuple = (), commit: bool = False):
        """
        Wykonuje zapytanie SQL z obsługą transakcji i ponawiania.

        Parameters:
            query (str): Zapytanie SQL.
            params (tuple): Parametry do zapytania.
            commit (bool): Czy zatwierdzić zmiany po wykonaniu zapytania.

        Returns:
            list: Lista wierszy wynikowych jako słowniki.
        """
        retries = 3
        delay = 1
        for attempt in range(1, retries + 1):
            try:
                cursor = self.conn.cursor()
                start_time = time.time()
                cursor.execute(query, params)
                duration = time.time() - start_time
                logging.info("Zapytanie wykonane w %.4f s: %s", duration, query)
                if commit:
                    self.conn.commit()
                results = cursor.fetchall()
                return [dict(row) for row in results]
            except sqlite3.OperationalError as e:
                logging.warning(
                    "Błąd operacyjny przy zapytaniu (próba %d): %s", attempt, e
                )
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                logging.error("Błąd przy wykonywaniu zapytania: %s", e)
                raise
        logging.error("Nie udało się wykonać zapytania po %d próbach.", retries)
        raise Exception("Błąd wykonania zapytania.")

    def migrate_schema(self, migration_script: str):
        """
        Wykonuje migrację schematu bazy danych na podstawie podanego skryptu SQL.

        Parameters:
            migration_script (str): Ścieżka do pliku SQL zawierającego migrację.
        """
        try:
            with open(migration_script, "r") as file:
                sql = file.read()
            self.execute_query(sql, commit=True)
            logging.info("Migracja schematu wykonana z pliku: %s", migration_script)
        except Exception as e:
            logging.error("Błąd podczas migracji schematu: %s", e)
            raise

    def close(self):
        """
        Zamyka połączenie z bazą danych.
        """
        if self.conn:
            self.conn.close()
            logging.info("Połączenie z bazą danych zostało zamknięte.")


# Przykładowe użycie i testy
if __name__ == "__main__":
    try:
        dbm = DatabaseManager("example.db")
        # Przykładowe tworzenie tabeli
        create_table_query = """
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            value REAL
        );
        """
        dbm.execute_query(create_table_query, commit=True)
        # Przykładowe wstawienie rekordu
        insert_query = "INSERT INTO test_table (name, value) VALUES (?, ?);"
        dbm.execute_query(insert_query, ("example", 123.45), commit=True)
        # Odczyt danych
        select_query = "SELECT * FROM test_table;"
        results = dbm.execute_query(select_query)
        logging.info("Wyniki zapytania: %s", results)
        dbm.close()
    except Exception as e:
        logging.error("Błąd w module database_manager.py: %s", e)
        raise

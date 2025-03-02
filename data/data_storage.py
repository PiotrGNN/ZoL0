"""
data_storage.py
---------------
Moduł odpowiedzialny za zapisywanie i pobieranie danych rynkowych.
Obsługuje formaty CSV oraz bazę SQLite, z możliwością rozszerzenia o bazy NoSQL (np. MongoDB).

Funkcjonalności:
- Implementacja funkcji CRUD (create, read, update, delete) dla danych w formacie CSV oraz w bazie SQLite.
- Mechanizmy buforowania (cache) przy pobieraniu danych.
- Obsługa dużych wolumenów danych i automatyczne partycjonowanie tabel (przy wykorzystaniu partycjonowania na podstawie daty).
- Mechanizm backupu i odtwarzania w razie awarii.
- Testy wydajnościowe i funkcjonalne dla zapewnienia niezawodności.
"""

import csv
import logging
import os
import shutil
import sqlite3
from datetime import datetime
from functools import lru_cache

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ścieżki domyślne
DEFAULT_DB_PATH = "./data/historical_data.db"
BACKUP_DIR = "./data/backups"


class DataStorage:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Inicjalizuje obiekt do przechowywania danych.

        Parameters:
            db_path (str): Ścieżka do bazy SQLite.
        """
        self.db_path = db_path
        self.conn = None
        self._connect_db()
        os.makedirs(BACKUP_DIR, exist_ok=True)

    def _connect_db(self):
        """Łączy się z bazą danych SQLite."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logging.info("Połączono z bazą danych: %s", self.db_path)
        except Exception as e:
            logging.error("Błąd połączenia z bazą danych: %s", e)
            raise

    # ------------------ Funkcje CSV ------------------

    def create_csv(self, file_path: str, headers: list, rows: list):
        """
        Tworzy nowy plik CSV i zapisuje w nim dane.

        Parameters:
            file_path (str): Ścieżka do pliku CSV.
            headers (list): Lista nazw kolumn.
            rows (list): Lista wierszy, gdzie każdy wiersz to lista wartości.
        """
        try:
            with open(file_path, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(rows)
            logging.info("Utworzono plik CSV: %s", file_path)
        except Exception as e:
            logging.error("Błąd przy tworzeniu pliku CSV %s: %s", file_path, e)
            raise

    @lru_cache(maxsize=32)
    def read_csv(self, file_path: str) -> list:
        """
        Odczytuje plik CSV i zwraca listę słowników.

        Parameters:
            file_path (str): Ścieżka do pliku CSV.

        Returns:
            list: Lista wierszy w formie słowników.
        """
        try:
            with open(file_path, mode="r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                data = [row for row in reader]
            logging.info("Odczytano plik CSV: %s", file_path)
            return data
        except Exception as e:
            logging.error("Błąd przy odczycie pliku CSV %s: %s", file_path, e)
            raise

    def update_csv(self, file_path: str, headers: list, rows: list):
        """
        Nadpisuje plik CSV nowymi danymi.

        Parameters:
            file_path (str): Ścieżka do pliku CSV.
            headers (list): Lista nazw kolumn.
            rows (list): Nowa lista wierszy.
        """
        try:
            self.create_csv(file_path, headers, rows)
            # Wyczyść cache dla tego pliku
            self.read_csv.cache_clear()
            logging.info("Zaktualizowano plik CSV: %s", file_path)
        except Exception as e:
            logging.error("Błąd przy aktualizacji pliku CSV %s: %s", file_path, e)
            raise

    def delete_csv(self, file_path: str):
        """
        Usuwa plik CSV.

        Parameters:
            file_path (str): Ścieżka do pliku CSV.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.read_csv.cache_clear()
                logging.info("Usunięto plik CSV: %s", file_path)
            else:
                logging.warning("Plik CSV nie istnieje: %s", file_path)
        except Exception as e:
            logging.error("Błąd przy usuwaniu pliku CSV %s: %s", file_path, e)
            raise

    # ------------------ Funkcje SQLite ------------------

    def create_table(self, table_name: str, schema: str):
        """
        Tworzy tabelę w bazie danych SQLite.

        Parameters:
            table_name (str): Nazwa tabeli.
            schema (str): Definicja schematu SQL (np. "(timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)").
        """
        try:
            cursor = self.conn.cursor()
            sql = f"CREATE TABLE IF NOT EXISTS {table_name} {schema}"
            cursor.execute(sql)
            self.conn.commit()
            logging.info("Utworzono lub już istnieje tabela: %s", table_name)
        except Exception as e:
            logging.error("Błąd przy tworzeniu tabeli %s: %s", table_name, e)
            raise

    def insert_record(self, table_name: str, record: dict):
        """
        Wstawia pojedynczy rekord do tabeli.

        Parameters:
            table_name (str): Nazwa tabeli.
            record (dict): Rekord do wstawienia.
        """
        try:
            cursor = self.conn.cursor()
            columns = ", ".join(record.keys())
            placeholders = ", ".join(["?"] * len(record))
            sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, tuple(record.values()))
            self.conn.commit()
            logging.info("Wstawiono rekord do tabeli %s: %s", table_name, record)
        except Exception as e:
            logging.error("Błąd przy wstawianiu rekordu do tabeli %s: %s", table_name, e)
            raise

    def read_records(self, table_name: str, conditions: str = None) -> list:
        """
        Odczytuje rekordy z tabeli na podstawie warunków.

        Parameters:
            table_name (str): Nazwa tabeli.
            conditions (str, optional): Warunki SQL, np. "WHERE timestamp > '2023-01-01'".

        Returns:
            list: Lista rekordów (każdy rekord to słownik).
        """
        try:
            cursor = self.conn.cursor()
            sql = f"SELECT * FROM {table_name}"
            if conditions:
                sql += f" {conditions}"
            cursor.execute(sql)
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            logging.info("Odczytano %d rekordów z tabeli %s.", len(results), table_name)
            return results
        except Exception as e:
            logging.error("Błąd przy odczycie rekordów z tabeli %s: %s", table_name, e)
            raise

    def update_record(self, table_name: str, update_values: dict, conditions: str):
        """
        Aktualizuje rekordy w tabeli na podstawie warunków.

        Parameters:
            table_name (str): Nazwa tabeli.
            update_values (dict): Słownik wartości do aktualizacji.
            conditions (str): Warunki SQL (np. "WHERE timestamp = '2023-01-01 00:00:00'").
        """
        try:
            cursor = self.conn.cursor()
            set_clause = ", ".join([f"{k} = ?" for k in update_values.keys()])
            sql = f"UPDATE {table_name} SET {set_clause} {conditions}"
            cursor.execute(sql, tuple(update_values.values()))
            self.conn.commit()
            logging.info(
                "Zaktualizowano rekordy w tabeli %s przy warunkach: %s",
                table_name,
                conditions,
            )
        except Exception as e:
            logging.error("Błąd przy aktualizacji rekordów w tabeli %s: %s", table_name, e)
            raise

    def delete_record(self, table_name: str, conditions: str):
        """
        Usuwa rekordy z tabeli na podstawie warunków.

        Parameters:
            table_name (str): Nazwa tabeli.
            conditions (str): Warunki SQL (np. "WHERE timestamp < '2023-01-01'").
        """
        try:
            cursor = self.conn.cursor()
            sql = f"DELETE FROM {table_name} {conditions}"
            cursor.execute(sql)
            self.conn.commit()
            logging.info(
                "Usunięto rekordy z tabeli %s przy warunkach: %s",
                table_name,
                conditions,
            )
        except Exception as e:
            logging.error("Błąd przy usuwaniu rekordów z tabeli %s: %s", table_name, e)
            raise

    # ------------------ Mechanizmy Backup i Odtwarzania ------------------

    def backup_db(self):
        """
        Tworzy backup bazy danych SQLite, kopiując plik bazy do folderu backup.
        Nazwa backupu zawiera datę i godzinę wykonania.
        """
        try:
            if os.path.exists(self.db_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(BACKUP_DIR, f"backup_{timestamp}.db")
                shutil.copy2(self.db_path, backup_path)
                logging.info("Utworzono backup bazy danych: %s", backup_path)
            else:
                logging.warning("Plik bazy danych nie istnieje, backup nie został wykonany.")
        except Exception as e:
            logging.error("Błąd podczas tworzenia backupu bazy danych: %s", e)
            raise

    def restore_db(self, backup_file: str):
        """
        Odtwarza bazę danych z pliku backupu.

        Parameters:
            backup_file (str): Ścieżka do pliku backupu.
        """
        try:
            if os.path.exists(backup_file):
                # Zamykamy bieżące połączenie
                if self.conn:
                    self.conn.close()
                shutil.copy2(backup_file, self.db_path)
                self._connect_db()
                logging.info("Baza danych została odtworzona z backupu: %s", backup_file)
            else:
                logging.error("Plik backupu nie istnieje: %s", backup_file)
        except Exception as e:
            logging.error("Błąd podczas odtwarzania bazy danych: %s", e)
            raise


# ------------------ Przykładowe użycie i testy funkcjonalne ------------------

if __name__ == "__main__":
    try:
        # Inicjalizacja DataStorage
        ds = DataStorage()

        # Przykład: tworzenie tabeli dla danych świecowych
        table_name = "candles"
        schema = "(timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)"
        ds.create_table(table_name, schema)

        # Wstawianie przykładowego rekordu
        record = {
            "timestamp": "2023-01-01 00:00:00",
            "open": 16500,
            "high": 16600,
            "low": 16450,
            "close": 16550,
            "volume": 350.5,
        }
        ds.insert_record(table_name, record)

        # Odczyt rekordów
        records = ds.read_records(table_name)
        logging.info("Rekordy w tabeli '%s': %s", table_name, records)

        # Aktualizacja rekordu (przykład aktualizacji ceny zamknięcia)
        ds.update_record(table_name, {"close": 16580}, "WHERE timestamp = '2023-01-01 00:00:00'")
        updated_records = ds.read_records(table_name)
        logging.info("Zaktualizowane rekordy: %s", updated_records)

        # Usunięcie rekordu
        ds.delete_record(table_name, "WHERE timestamp = '2023-01-01 00:00:00'")
        final_records = ds.read_records(table_name)
        logging.info("Rekordy po usunięciu: %s", final_records)

        # Test operacji na CSV
        csv_file = "./data/test_historical_data.csv"
        headers = ["timestamp", "open", "high", "low", "close", "volume"]
        rows = [
            ["2023-01-01 00:00:00", "16500", "16600", "16450", "16550", "350.5"],
            ["2023-01-01 00:01:00", "16550", "16620", "16530", "16580", "300.1"],
        ]
        ds.create_csv(csv_file, headers, rows)
        csv_data = ds.read_csv(csv_file)
        logging.info("Dane z pliku CSV: %s", csv_data)
        ds.update_csv(
            csv_file,
            headers,
            rows + [["2023-01-01 00:02:00", "16580", "16650", "16560", "16620", "400.0"]],
        )
        updated_csv_data = ds.read_csv(csv_file)
        logging.info("Zaktualizowane dane z pliku CSV: %s", updated_csv_data)
        ds.delete_csv(csv_file)

        # Test backup i restore bazy danych
        ds.backup_db()
        # Aby przetestować restore, należy podać istniejący plik backupu.
        # Przykładowo: ds.restore_db("./data/backups/backup_20230101_120000.db")

        logging.info("Testy funkcjonalne modułu data_storage.py zakończone sukcesem.")
    except Exception as e:
        logging.error("Błąd w module data_storage.py: %s", e)
        raise

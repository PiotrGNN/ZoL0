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
from typing import List, Dict, Any
from datetime import datetime
import json

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
        self.cursor = None
        self.connect()
        self._initialize_db()

    def connect(self):
        """
        Nawiązuje połączenie z bazą danych.
        """
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            logging.info("Połączono z bazą danych: %s", self.db_path)
        except Exception as e:
            logging.error("Błąd połączenia z bazą danych: %s", e)
            raise

    def _initialize_db(self):
        """
        Inicjalizuje bazę danych i tworzy wymagane tabele, jeśli nie istnieją.
        """
        try:
            # Create AI models metadata table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    training_date TIMESTAMP,
                    metrics TEXT,
                    status TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create trades table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    status TEXT,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    pnl REAL,
                    side TEXT
                )
            """)
            
            self.conn.commit()
            
            # Add some initial test data if table is empty
            self.cursor.execute("SELECT COUNT(*) FROM ai_models")
            if self.cursor.fetchone()[0] == 0:
                self._populate_test_models()
                
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise

    def _populate_test_models(self):
        """Populate test AI model data."""
        test_models = [
            {
                'model_name': 'RandomForest',
                'model_type': 'Classification',
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': json.dumps({'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87}),
                'status': 'Active'
            },
            {
                'model_name': 'SentimentAnalyzer',
                'model_type': 'NLP',
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': json.dumps({'accuracy': 0.78, 'f1_score': 0.76}),
                'status': 'Active'
            },
            {
                'model_name': 'AnomalyDetector',
                'model_type': 'Anomaly',
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': json.dumps({'precision': 0.92, 'recall': 0.88}),
                'status': 'Active'
            }
        ]
        
        for model in test_models:
            self.cursor.execute("""
                INSERT INTO ai_models (model_name, model_type, training_date, metrics, status)
                VALUES (:model_name, :model_type, :training_date, :metrics, :status)
            """, model)
        
        self.conn.commit()
        logging.info("Populated test AI model data")

    def get_model_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all AI models.

        Returns:
            List[Dict[str, Any]]: List of model metadata records
        """
        try:
            self.cursor.execute("""
                SELECT model_name, model_type, training_date, metrics, status, last_updated 
                FROM ai_models 
                ORDER BY last_updated DESC
            """)

            columns = ['model_name', 'model_type', 'training_date', 'metrics', 'status', 'last_updated']
            results = []

            for row in self.cursor.fetchall():
                model_data = {}
                for i, column in enumerate(columns):
                    if column == 'training_date' or column == 'last_updated':
                        model_data[column] = datetime.strptime(row[i], '%Y-%m-%d %H:%M:%S') if row[i] else None
                    else:
                        model_data[column] = row[i]
                results.append(model_data)

            return results

        except Exception as e:
            logging.error(f"Error fetching model metadata: {e}")
            return []

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

    def get_trades(self, status: str = None) -> List[Dict[str, Any]]:
        """
        Get trades from the database, optionally filtered by status.
        
        Args:
            status (str, optional): Filter trades by status (e.g., 'OPEN', 'CLOSED')
            
        Returns:
            List[Dict[str, Any]]: List of trade dictionaries with fields:
                - id: Trade ID
                - symbol: Trading pair symbol
                - direction: Trade direction (e.g. 'LONG', 'SHORT')
                - entry_price: Entry price
                - exit_price: Exit price (if trade is closed)
                - quantity: Trade size
                - status: Trade status
                - entry_time: Entry timestamp
                - exit_time: Exit timestamp (if trade is closed)
                - profit_loss: Realized P&L (if trade is closed)
                - commission: Trading fees
                - strategy_id: Associated strategy ID
        """
        query = "SELECT * FROM trades"
        params = ()
        
        if status:
            query += " WHERE status = ?"
            params = (status,)
            
        return self.execute_query(query, params)


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

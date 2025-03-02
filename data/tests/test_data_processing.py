"""
test_data_processing.py
-----------------------
Testy jednostkowe dla modułów:
- data_preprocessing.py (czyszczenie danych, transformacja, detekcja outlierów)
- data_storage.py (operacje CRUD na plikach CSV oraz bazie SQLite)
- historical_data.py (wczytywanie, aktualizacja i walidacja danych historycznych)
Testy mają na celu weryfikację poprawności przetwarzania danych oraz integracji między modułami.
"""

import logging
import os
import unittest

import numpy as np
import pandas as pd

# Zakładamy, że moduły znajdują się w folderze data/data oraz data
from data.data.data_preprocessing import (
    clean_data,
    compute_log_returns,
    detect_outliers,
    preprocess_pipeline,
    winsorize_series,
)
from data.data.historical_data import HistoricalDataManager

# Konfiguracja logowania do testów
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Przygotowanie przykładowego DataFrame z danymi rynkowymi
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(110, 120, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(100, 115, 100),
                "volume": np.random.randint(1000, 2000, 100),
            }
        )
        # Wprowadzenie braków i outlierów
        self.df.loc[10, "close"] = np.nan
        self.df.loc[20, "close"] = self.df["close"].max() * 1.5

    def test_clean_data(self):
        # Test czyszczenia danych
        df_clean = clean_data(self.df, fill_method="median")
        self.assertFalse(df_clean.isnull().values.any(), "Dane powinny być oczyszczone z braków.")

    def test_compute_log_returns(self):
        # Test obliczania logarytmicznych zwrotów
        df_clean = clean_data(self.df, fill_method="median")
        log_returns = compute_log_returns(df_clean, price_col="close")
        self.assertTrue(isinstance(log_returns, pd.Series), "Wynik powinien być pd.Series.")
        self.assertGreater(len(log_returns), 0, "Logarytmiczne zwroty nie powinny być puste.")

    def test_detect_outliers(self):
        # Test detekcji outlierów
        df_out = detect_outliers(self.df, column="close", threshold=2.5)
        self.assertIn(
            "is_outlier",
            df_out.columns,
            "DataFrame powinien zawierać kolumnę 'is_outlier'.",
        )

    def test_winsorize_series(self):
        # Test winsoryzacji serii
        series = self.df["close"]
        winsorized = winsorize_series(series, limits=(0.05, 0.05))
        self.assertTrue(
            (winsorized >= series.quantile(0.05)).all() and (winsorized <= series.quantile(0.95)).all(),
            "Wartości winsoryzowane powinny być w określonych granicach.",
        )

    def test_preprocess_pipeline(self):
        # Test kompleksowego pipeline'u przetwarzania danych
        df_processed = preprocess_pipeline(
            self.df,
            price_col="close",
            fill_method="median",
            outlier_threshold=2.5,
            winsorize_limits=(0.05, 0.05),
        )
        self.assertIn(
            "log_return",
            df_processed.columns,
            "Przetworzone dane powinny zawierać kolumnę 'log_return'.",
        )


class TestDataStorageAndHistoricalData(unittest.TestCase):
    def setUp(self):
        # Przygotowanie przykładowych danych do testowania operacji na CSV
        self.test_csv = "temp_test_data.csv"
        self.df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="D"),
                "open": np.linspace(100, 110, 10),
                "high": np.linspace(105, 115, 10),
                "low": np.linspace(95, 105, 10),
                "close": np.linspace(100, 110, 10),
                "volume": np.random.randint(1000, 1500, 10),
            }
        )
        self.df.to_csv(self.test_csv, index=False)
        # Ścieżka do tymczasowej bazy SQLite
        self.test_db = "temp_historical_data.db"
        # Inicjalizacja managera danych historycznych
        self.historical_manager = HistoricalDataManager(csv_path=self.test_csv, db_path=self.test_db)

    def tearDown(self):
        # Usuwanie tymczasowych plików
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_load_from_csv(self):
        df_loaded = self.historical_manager.load_from_csv()
        self.assertFalse(df_loaded.empty, "Wczytane dane z CSV nie powinny być puste.")
        self.assertIn("timestamp", df_loaded.columns, "CSV powinien zawierać kolumnę 'timestamp'.")

    def test_update_csv(self):
        # Dodanie nowego rekordu i sprawdzenie aktualizacji
        new_record = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2023-01-11")],
                "open": [111],
                "high": [116],
                "low": [106],
                "close": [112],
                "volume": [1300],
            }
        )
        self.historical_manager.update_csv(new_record)
        df_updated = pd.read_csv(self.test_csv)
        self.assertIn(
            "2023-01-11",
            df_updated["timestamp"].astype(str).iloc[-1],
            "Nowy rekord nie został dodany.",
        )

    def test_load_from_db(self):
        # Testujemy operacje na bazie SQLite
        ds = self.historical_manager.data_storage
        table_name = "candles"
        schema = "(timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)"
        ds.create_table(table_name, schema)
        record = {
            "timestamp": "2023-01-01 00:00:00",
            "open": 100,
            "high": 105,
            "low": 95,
            "close": 102,
            "volume": 1200,
        }
        ds.insert_record(table_name, record)
        records = ds.read_records(table_name)
        self.assertGreaterEqual(
            len(records),
            1,
            "Tabela w bazie SQLite powinna zawierać co najmniej jeden rekord.",
        )


if __name__ == "__main__":
    unittest.main()

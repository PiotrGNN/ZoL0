"""
test_data_processing.py
-----------------------
Testy jednostkowe dla modułów:
  - data_preprocessing.py (czyszczenie danych, transformacja, detekcja outlierów)
  - historical_data.py (wczytywanie, aktualizacja i walidacja danych historycznych)

Testy mają na celu weryfikację poprawności przetwarzania danych oraz integracji między modułami.
"""

import logging
import os
import sqlite3
import unittest
from typing import Any

import numpy as np
import pandas as pd

from data.data.data_preprocessing import (
    clean_data,
    compute_log_returns,
    detect_outliers,
    preprocess_pipeline,
    winsorize_series,
)
from data.data.historical_data import HistoricalDataManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class TestDataProcessing(unittest.TestCase):
    """Testy modułów przetwarzania danych."""

    def setUp(self) -> None:
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(100, 110, 100),
            "high": np.random.uniform(110, 120, 100),
            "low": np.random.uniform(90, 100, 100),
            "close": np.random.uniform(100, 115, 100),
            "volume": np.random.randint(1000, 2000, 100),
        })
        self.df.loc[10, "close"] = np.nan
        self.df.loc[20, "close"] = self.df["close"].max() * 2.0  # Bardzo duży outlier

    def test_clean_data(self) -> None:
        df_clean = clean_data(self.df, fill_method="median")
        self.assertFalse(
            df_clean.isnull().values.any(),
            "Dane powinny być oczyszczone z braków (fill_method='median')."
        )

    def test_compute_log_returns(self) -> None:
        df_clean = clean_data(self.df, fill_method="median")
        log_returns = compute_log_returns(df_clean, price_col="close")
        self.assertIsInstance(log_returns, pd.Series)
        self.assertGreater(len(log_returns), 0)

    def test_detect_outliers(self) -> None:
        df_out = detect_outliers(self.df, column="close", threshold=2.5)
        self.assertIn("is_outlier", df_out.columns)
        self.assertTrue(df_out["is_outlier"].iloc[20])

    def test_winsorize_series(self) -> None:
        series = self.df["close"]
        winsorized = winsorize_series(series, limits=(0.05, 0.05))

        lower_limit, upper_limit = series.quantile([0.05, 0.95])
        tol = (upper_limit - lower_limit) * 0.1  # tolerancja na poziomie 10% zakresu
        self.assertTrue(
            winsorized.between(lower_limit - tol, upper_limit + tol).all(),
            f"Wartości winsoryzowane poza zakresem [{lower_limit - tol}, {upper_limit + tol}]."
        )

    def test_preprocess_pipeline(self) -> None:
        df_processed = preprocess_pipeline(
            self.df,
            price_col="close",
            fill_method="median",
            outlier_threshold=2.5,
            winsorize_limits=(0.05, 0.05)
        )
        self.assertIn("log_return", df_processed.columns)


class TestDataStorageAndHistoricalData(unittest.TestCase):
    """Testy operacji na plikach CSV i bazie SQLite."""

    def setUp(self) -> None:
        self.test_csv = "temp_test_data.csv"
        self.test_db = "temp_historical_data.db"
        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
            "open": np.linspace(100, 110, 10),
            "high": np.linspace(105, 115, 10),
            "low": np.linspace(95, 105, 10),
            "close": np.linspace(100, 110, 10),
            "volume": np.random.randint(1000, 1500, 10),
        })
        self.df.to_csv(self.test_csv, index=False)

        self.historical_manager = HistoricalDataManager(
            csv_path=self.test_csv,
            db_path=self.test_db
        )

    def tearDown(self) -> None:
        if hasattr(self.historical_manager, 'close_connection'):
            self.historical_manager.close_connection()
        for file in [self.test_db, self.test_csv]:
            if os.path.exists(file):
                os.remove(file)

    def test_load_from_csv(self) -> None:
        df_loaded = self.historical_manager.load_from_csv()
        self.assertFalse(df_loaded.empty)
        self.assertIn("timestamp", df_loaded.columns)

    def test_update_csv(self) -> None:
        new_record = pd.DataFrame({
            "timestamp": [pd.Timestamp("2023-01-11")],
            "open": [111.0],
            "high": [116.0],
            "low": [106.0],
            "close": [112.0],
            "volume": [1300],
        })
        self.historical_manager.update_csv(new_record)
        df_updated = pd.read_csv(self.test_csv)
        self.assertIn("2023-01-11", df_updated["timestamp"].values)


if __name__ == "__main__":
    unittest.main()

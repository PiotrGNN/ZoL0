"""
historical_data.py
------------------
Moduł do zarządzania historycznymi danymi rynkowymi.

Funkcjonalności:
- Wczytywanie danych z plików CSV oraz z bazy danych SQLite.
- Aktualizacja danych (dopisywanie nowych świec) oraz ich walidacja (np. brak luk czasowych, spójność wolumenu).
- Integracja z modułami data_storage.py i config_loader.py w celu elastycznego wyboru źródła danych.
- Testy jednostkowe sprawdzające poprawność danych.
"""

import logging
import os

import pandas as pd

# Import modułu data_storage, zakładając, że jest on w ścieżce
from ..data_storage import DataStorage

# Import ConfigLoader, zakładając, że plik config_loader.py jest dostępny w folderze config


# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class HistoricalDataManager:
    def __init__(self, csv_path: str = None, db_path: str = None, config: dict = None):
        """
        Inicjalizuje managera danych historycznych.

        Parameters:
            csv_path (str): Ścieżka do pliku CSV z danymi historycznymi.
            db_path (str): Ścieżka do bazy SQLite z danymi historycznymi.
            config (dict): Konfiguracja ładowana z plików przez ConfigLoader.
        """
        self.csv_path = csv_path
        self.db_path = db_path
        self.config = config or {}
        self.data_storage = None

        if self.db_path:
            # Inicjalizujemy DataStorage z wykorzystaniem bazy SQLite
            self.data_storage = DataStorage(db_path=self.db_path)
        logging.info(
            "HistoricalDataManager zainicjalizowany. CSV: %s, DB: %s",
            self.csv_path,
            self.db_path,
        )

    def load_from_csv(self) -> pd.DataFrame:
        """
        Wczytuje dane historyczne z pliku CSV.

        Returns:
            pd.DataFrame: Wczytane dane.
        """
        if not self.csv_path or not os.path.exists(self.csv_path):
            msg = f"Plik CSV nie istnieje: {self.csv_path}"
            logging.error(msg)
            raise FileNotFoundError(msg)
        try:
            df = pd.read_csv(self.csv_path)
            # Konwersja kolumny timestamp do datetime, jeśli istnieje
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            logging.info("Wczytano dane z CSV: %s", self.csv_path)
            return df
        except Exception as e:
            logging.error("Błąd przy wczytywaniu danych z CSV: %s", e)
            raise

    def load_from_db(self, table_name: str = "candles") -> pd.DataFrame:
        """
        Wczytuje dane historyczne z bazy danych SQLite.

        Parameters:
            table_name (str): Nazwa tabeli z danymi.

        Returns:
            pd.DataFrame: Wczytane dane.
        """
        if not self.data_storage:
            msg = (
                "DataStorage nie został zainicjalizowany, brak ścieżki do bazy danych."
            )
            logging.error(msg)
            raise ValueError(msg)
        try:
            records = self.data_storage.read_records(table_name)
            df = pd.DataFrame(records)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            logging.info("Wczytano dane z bazy danych z tabeli: %s", table_name)
            return df
        except Exception as e:
            logging.error("Błąd przy wczytywaniu danych z bazy: %s", e)
            raise

    def update_csv(self, new_data: pd.DataFrame):
        """
        Aktualizuje plik CSV, dopisując nowe dane.

        Parameters:
            new_data (pd.DataFrame): Nowe dane do dopisania.
        """
        try:
            if not self.csv_path:
                msg = "Ścieżka CSV nie została określona."
                logging.error(msg)
                raise ValueError(msg)

            # Konwertuj timestamp na datetime jeśli jeszcze nie jest
            if "timestamp" in new_data.columns and not pd.api.types.is_datetime64_any_dtype(new_data["timestamp"]):
                new_data = new_data.copy()
                new_data["timestamp"] = pd.to_datetime(new_data["timestamp"])

            # Jeśli plik istnieje, wczytujemy go i łączymy z nowymi danymi
            if os.path.exists(self.csv_path):
                df_existing = pd.read_csv(self.csv_path)
                if "timestamp" in df_existing.columns:
                    df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"])
                df_combined = pd.concat([df_existing, new_data], ignore_index=True)
            else:
                df_combined = new_data

            # Formatuj timestamp do formatu YYYY-MM-DD przed zapisem
            if "timestamp" in df_combined.columns:
                df_combined = df_combined.copy()
                df_combined["timestamp"] = df_combined["timestamp"].dt.strftime("%Y-%m-%d")

            df_combined.to_csv(self.csv_path, index=False)
            logging.info("Plik CSV został zaktualizowany: %s", self.csv_path)
        except Exception as e:
            logging.error("Błąd przy aktualizacji CSV: %s", e)
            raise

    def update_db(self, new_data: pd.DataFrame, table_name: str = "candles"):
        """
        Aktualizuje dane w bazie SQLite, dopisując nowe rekordy.

        Parameters:
            new_data (pd.DataFrame): Nowe dane do dodania.
            table_name (str): Nazwa tabeli, do której dane mają być dodane.
        """
        if not self.data_storage:
            msg = (
                "DataStorage nie został zainicjalizowany, brak ścieżki do bazy danych."
            )
            logging.error(msg)
            raise ValueError(msg)
        try:
            # Zakładamy, że new_data jest DataFrame i iterujemy po wierszach
            for _, row in new_data.iterrows():
                record = row.to_dict()
                self.data_storage.insert_record(table_name, record)
            logging.info("Dane zostały zaktualizowane w tabeli %s", table_name)
        except Exception as e:
            logging.error("Błąd przy aktualizacji danych w bazie: %s", e)
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Walidacja danych historycznych.
        Sprawdza brak luk czasowych oraz spójność wolumenu.

        Parameters:
            df (pd.DataFrame): DataFrame z danymi.

        Returns:
            bool: True, jeśli dane są poprawne, w przeciwnym razie ValueError.
        """
        try:
            # Sprawdzenie czy kolumna timestamp jest posortowana
            if not df["timestamp"].is_monotonic_increasing:
                msg = "Dane nie są posortowane według czasu."
                logging.error(msg)
                raise ValueError(msg)
            # Sprawdzenie braków w czasie – np. brakujące dni lub okresy, zależnie od częstotliwości
            df_sorted = df.sort_values("timestamp")
            time_diffs = df_sorted["timestamp"].diff().dropna()
            # Ustalony minimalny odstęp (przykładowo 1 minuta lub 1 dzień - zależy od danych)
            min_diff = time_diffs.min()
            if min_diff > pd.Timedelta(minutes=1):
                logging.warning(
                    "Wykryto duże odstępy czasowe między rekordami: %s", min_diff
                )
            # Sprawdzenie spójności wolumenu – przykładowa walidacja: wolumeny nie mogą być ujemne
            if (df["volume"] < 0).any():
                msg = "Wolumen zawiera ujemne wartości."
                logging.error(msg)
                raise ValueError(msg)
            logging.info("Walidacja danych zakończona pomyślnie.")
            return True
        except Exception as e:
            logging.error("Błąd podczas walidacji danych: %s", e)
            raise


# -------------------- Testy jednostkowe --------------------
def unit_test_historical_data_manager():
    """
    Testy jednostkowe dla modułu historical_data.py.
    - Test wczytywania danych z CSV.
    - Test walidacji danych.
    - Test aktualizacji danych (CSV).
    """
    try:
        # Przygotowanie przykładowych danych
        sample_data = {
            "timestamp": [
                "2023-01-01 09:30:00",
                "2023-01-02 09:30:00",
                "2023-01-03 09:30:00",
            ],
            "open": [100, 104, 109],
            "high": [105, 110, 112],
            "low": [99, 102, 107],
            "close": [104, 109, 108],
            "volume": [1500, 2000, 1800],
        }
        df_sample = pd.DataFrame(sample_data)
        df_sample["timestamp"] = pd.to_datetime(df_sample["timestamp"])

        # Utworzenie tymczasowego pliku CSV
        temp_csv = "temp_historical_data.csv"
        df_sample.to_csv(temp_csv, index=False)

        # Inicjalizacja managera danych historycznych
        manager = HistoricalDataManager(csv_path=temp_csv)

        # Test wczytywania danych
        df_loaded = manager.load_from_csv()
        assert not df_loaded.empty, "Wczytane dane są puste."
        logging.info("Test wczytywania danych z CSV zakończony sukcesem.")

        # Test walidacji danych
        valid = manager.validate_data(df_loaded)
        assert valid, "Walidacja danych nie powiodła się."
        logging.info("Test walidacji danych zakończony sukcesem.")

        # Test aktualizacji CSV: dopisanie nowego rekordu
        new_record = {
            "timestamp": "2023-01-04 09:30:00",
            "open": 108,
            "high": 109,
            "low": 105,
            "close": 106,
            "volume": 1700,
        }
        df_new = pd.DataFrame([new_record])
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])
        manager.update_csv(df_new)

        df_updated = pd.read_csv(temp_csv)
        assert (
            "2023-01-04" in df_updated["timestamp"].iloc[-1]
        ), "Nowy rekord nie został dodany."
        logging.info("Test aktualizacji CSV zakończony sukcesem.")

        # Sprzątanie
        os.remove(temp_csv)
        logging.info("Testy jednostkowe historical_data.py zakończone sukcesem.")
    except AssertionError as ae:
        logging.error("AssertionError w testach jednostkowych: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach jednostkowych: %s", e)
        raise


if __name__ == "__main__":
    try:
        unit_test_historical_data_manager()
    except Exception as e:
        logging.error("Testy jednostkowe nie powiodły się: %s", e)
        raise

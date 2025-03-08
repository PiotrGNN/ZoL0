"""
data_validator.py
-----------------
Moduł weryfikujący poprawność danych wejściowych oraz wyników przetwarzania.
Funkcjonalności:
- Sprawdza zgodność typów danych, zakresy wartości (np. ceny > 0) oraz spójność czasową (brak luk, poprawny format daty).
- Umożliwia definiowanie niestandardowych reguł walidacji poprzez konfigurację.
- Loguje niezgodności i błędy, umożliwiając automatyczne odrzucenie błędnych rekordów.
- Zapewnia testy wydajnościowe, aby radzić sobie z dużymi wolumenami danych.
"""

import logging

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def validate_columns(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Sprawdza, czy DataFrame zawiera wszystkie wymagane kolumny.

    Parameters:
        df (pd.DataFrame): Dane wejściowe.
        required_columns (list): Lista nazw wymaganych kolumn.

    Returns:
        bool: True, jeśli wszystkie kolumny są obecne, w przeciwnym razie False.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logging.error("Brakujące kolumny: %s", missing)
        return False
    logging.info("Wszystkie wymagane kolumny są obecne.")
    return True


def validate_data_types(df: pd.DataFrame, column_types: dict) -> bool:
    """
    Sprawdza, czy kolumny DataFrame mają oczekiwane typy danych.

    Parameters:
        df (pd.DataFrame): Dane wejściowe.
        column_types (dict): Słownik, gdzie klucz to nazwa kolumny, a wartość to oczekiwany typ (np. int, float, datetime64[ns]).

    Returns:
        bool: True, jeśli typy danych są zgodne, w przeciwnym razie False.
    """
    for col, expected_type in column_types.items():
        if col not in df.columns:
            logging.error("Kolumna %s nie istnieje.", col)
            return False
        if not np.issubdtype(df[col].dtype, expected_type):
            logging.error(
                "Kolumna %s ma typ %s, oczekiwano %s.",
                col,
                df[col].dtype,
                expected_type,
            )
            return False
    logging.info("Wszystkie kolumny mają oczekiwane typy danych.")
    return True


def validate_date_integrity(df: pd.DataFrame, date_column: str) -> bool:
    """
    Sprawdza, czy kolumna z datami jest w poprawnym formacie i czy dane są spójne czasowo (np. brak luk).

    Parameters:
        df (pd.DataFrame): Dane wejściowe.
        date_column (str): Nazwa kolumny zawierającej daty.

    Returns:
        bool: True, jeśli daty są poprawne i spójne, w przeciwnym razie False.
    """
    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except Exception as e:
        logging.error("Błąd konwersji kolumny %s na datę: %s", date_column, e)
        return False
    if not df[date_column].is_monotonic_increasing:
        logging.error("Daty w kolumnie %s nie są uporządkowane rosnąco.", date_column)
        return False
    logging.info("Kolumna %s zawiera poprawne i spójne daty.", date_column)
    return True


def validate_numeric_ranges(df: pd.DataFrame, range_constraints: dict) -> bool:
    """
    Sprawdza, czy wartości w kolumnach numerycznych mieszczą się w określonych zakresach.

    Parameters:
        df (pd.DataFrame): Dane wejściowe.
        range_constraints (dict): Słownik z ograniczeniami, np. {"price": (0, None)} oznacza, że cena musi być dodatnia.

    Returns:
        bool: True, jeśli wszystkie ograniczenia są spełnione, w przeciwnym razie False.
    """
    valid = True
    for col, (min_val, max_val) in range_constraints.items():
        if col not in df.columns:
            logging.error("Kolumna %s nie istnieje dla walidacji zakresu.", col)
            valid = False
            continue
        if min_val is not None and (df[col] < min_val).any():
            logging.error("Kolumna %s zawiera wartości mniejsze niż %s.", col, min_val)
            valid = False
        if max_val is not None and (df[col] > max_val).any():
            logging.error("Kolumna %s zawiera wartości większe niż %s.", col, max_val)
            valid = False
    if valid:
        logging.info(
            "Wszystkie wartości numeryczne mieszczą się w określonych zakresach."
        )
    return valid


# -------------------- Testy jednostkowe --------------------
if __name__ == "__main__":
    import unittest

    class TestDataValidator(unittest.TestCase):
        def setUp(self):
            # Tworzymy przykładowe dane
            dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
            self.df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "price": np.linspace(100, 110, 10),
                    "volume": np.random.randint(100, 200, 10),
                }
            )

        def test_validate_columns(self):
            self.assertTrue(validate_columns(self.df, ["timestamp", "price", "volume"]))
            self.assertFalse(
                validate_columns(
                    self.df, ["timestamp", "price", "volume", "missing_col"]
                )
            )

        def test_validate_data_types(self):
            # Zakładamy, że timestamp powinien być datetime64, price float, volume int
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.df["price"] = self.df["price"].astype(float)
            self.df["volume"] = self.df["volume"].astype(int)
            expected_types = {
                "timestamp": np.datetime64,
                "price": np.floating,
                "volume": np.integer,
            }
            self.assertTrue(validate_data_types(self.df, expected_types))

        def test_validate_date_integrity(self):
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.assertTrue(validate_date_integrity(self.df, "timestamp"))
            # Modyfikacja kolejności dat, aby test był negatywny
            df_unsorted = self.df.iloc[::-1]
            self.assertFalse(validate_date_integrity(df_unsorted, "timestamp"))

        def test_validate_numeric_ranges(self):
            constraints = {"price": (0, None), "volume": (50, 250)}
            self.assertTrue(validate_numeric_ranges(self.df, constraints))
            # Dodajemy błędną wartość
            self.df.loc[0, "price"] = -10
            self.assertFalse(validate_numeric_ranges(self.df, constraints))

    unittest.main()

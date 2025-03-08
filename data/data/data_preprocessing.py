"""
data_preprocessing.py
---------------------
Moduł do wstępnego przetwarzania danych rynkowych.
Funkcjonalności:
- Czyszczenie danych: usuwanie lub uzupełnianie braków.
- Normalizacja/standaryzacja danych oraz transformacja cech (np. log-return).
- Automatyczna detekcja outlierów i ich traktowanie (np. winsoryzacja, odrzucanie).
- Pipeline przetwarzania, który można zintegrować w procesie treningu modeli AI.
- Możliwość konfigurowania parametrów z plików JSON/YAML i integracji z modułem config_loader.py.
- Testy jednostkowe oraz logowanie najważniejszych etapów przetwarzania.
"""

import logging

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def clean_data(df: pd.DataFrame, fill_method: str = "median") -> pd.DataFrame:
    """
    Czyści dane usuwając lub uzupełniając brakujące wartości.

    Parameters:
        df (pd.DataFrame): DataFrame z danymi rynkowymi.
        fill_method (str): Metoda uzupełniania braków ('median' lub 'mean').
                           Jeśli 'drop', usuwa wiersze z brakującymi wartościami.

    Returns:
        pd.DataFrame: Oczyszczony DataFrame.
    """
    df_clean = df.copy()
    if fill_method == "median":
        fill_value = df_clean.median(numeric_only=True)
        df_clean = df_clean.fillna(fill_value)
        logging.info("Uzupełniono brakujące wartości medianą.")
    elif fill_method == "mean":
        fill_value = df_clean.mean(numeric_only=True)
        df_clean = df_clean.fillna(fill_value)
        logging.info("Uzupełniono brakujące wartości średnią.")
    elif fill_method == "drop":
        df_clean = df_clean.dropna()
        logging.info("Usunięto wiersze z brakującymi wartościami.")
    else:
        logging.warning(
            "Nieznana metoda uzupełniania braków: %s. Domyślnie użyto mediany.",
            fill_method,
        )
        fill_value = df_clean.median(numeric_only=True)
        df_clean = df_clean.fillna(fill_value)
    return df_clean


def compute_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """
    Oblicza logarytmiczne zwroty na podstawie kolumny z cenami.

    Parameters:
        df (pd.DataFrame): DataFrame z danymi.
        price_col (str): Nazwa kolumny zawierającej ceny.

    Returns:
        pd.Series: Logarytmiczne zwroty.
    """
    try:
        prices = df[price_col]
        log_returns = np.log(prices).diff().dropna()
        logging.info("Obliczono logarytmiczne zwroty.")
        return log_returns
    except Exception as e:
        logging.error("Błąd przy obliczaniu logarytmicznych zwrotów: %s", e)
        raise


def detect_outliers(
    df: pd.DataFrame, column: str, threshold: float = 3.0
) -> pd.DataFrame:
    """
    Wykrywa outliery w określonej kolumnie, wykorzystując metodę z-score.

    Parameters:
        df (pd.DataFrame): DataFrame z danymi.
        column (str): Nazwa kolumny do analizy.
        threshold (float): Próg z-score, powyżej którego wartość jest uznawana za outlier.

    Returns:
        pd.DataFrame: DataFrame z dodatkową kolumną 'is_outlier' (True, jeśli wartość jest outlierem).
    """
    df_out = df.copy()
    try:
        col_values = df_out[column]
        mean_val = col_values.mean()
        std_val = col_values.std()
        z_scores = (col_values - mean_val) / std_val
        df_out["is_outlier"] = np.abs(z_scores) > threshold
        num_outliers = df_out["is_outlier"].sum()
        logging.info(
            "Wykryto %d outlierów w kolumnie %s przy progu z-score %.2f.",
            num_outliers,
            column,
            threshold,
        )
        return df_out
    except Exception as e:
        logging.error("Błąd przy wykrywaniu outlierów w kolumnie %s: %s", column, e)
        raise


def winsorize_series(series: pd.Series, limits: tuple = (0.05, 0.05)) -> pd.Series:
    """
    Winsoryzuje serię danych, ograniczając wartości ekstremalne.

    Parameters:
        series (pd.Series): Seria danych.
        limits (tuple): Krotka określająca dolne i górne limity (np. 5%).

    Returns:
        pd.Series: Winsoryzowana seria.
    """
    try:
        lower_limit, upper_limit = series.quantile([limits[0], 1 - limits[1]])
        winsorized = series.clip(lower=lower_limit, upper=upper_limit)
        logging.info(
            "Winsoryzacja danych zakończona. Dolny limit: %.2f, Górny limit: %.2f.",
            lower_limit,
            upper_limit,
        )
        return winsorized
    except Exception as e:
        logging.error("Błąd przy winsoryzacji serii: %s", e)
        raise


def preprocess_pipeline(
    df: pd.DataFrame,
    price_col: str = "close",
    fill_method: str = "median",
    outlier_threshold: float = 3.0,
    winsorize_limits: tuple = (0.05, 0.05),
) -> pd.DataFrame:
    """
    Kompleksowy pipeline przetwarzania danych:
    - Czyszczenie danych.
    - Detekcja i winsoryzacja outlierów.
    - Obliczanie logarytmicznych zwrotów.

    Parameters:
        df (pd.DataFrame): Dane wejściowe.
        price_col (str): Nazwa kolumny z cenami do obliczania log-return.
        fill_method (str): Metoda uzupełniania braków ('median', 'mean', 'drop').
        outlier_threshold (float): Próg z-score do wykrywania outlierów.
        winsorize_limits (tuple): Limity winsoryzacji (dolny, górny).

    Returns:
        pd.DataFrame: Przetworzone dane, zawierające oryginalne kolumny oraz nową kolumnę 'log_return'.
    """
    try:
        logging.info("Rozpoczynam pipeline przetwarzania danych.")
        # Czyszczenie danych
        df_clean = clean_data(df, fill_method=fill_method)

        # Detekcja outlierów w kolumnie z cenami
        df_outliers = detect_outliers(df_clean, price_col, threshold=outlier_threshold)
        # Winsoryzacja cen, jeśli są outliery
        df_clean[price_col] = winsorize_series(
            df_clean[price_col], limits=winsorize_limits
        )

        # Obliczenie logarytmicznych zwrotów
        log_returns = compute_log_returns(df_clean, price_col=price_col)
        df_clean = df_clean.iloc[
            1:
        ].copy()  # Usuwamy pierwszy wiersz, dla którego nie ma log-return
        df_clean["log_return"] = log_returns.values

        logging.info("Pipeline przetwarzania danych zakończony pomyślnie.")
        return df_clean
    except Exception as e:
        logging.error("Błąd w pipeline przetwarzania danych: %s", e)
        raise


# -------------------- Testy jednostkowe --------------------
def unit_test_preprocessing():
    """
    Testy jednostkowe dla modułu data_preprocessing.
    Tworzy przykładowy DataFrame, przetwarza go i sprawdza obecność oczekiwanych kolumn.
    """
    try:
        # Generujemy przykładowe dane
        data = {
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="D"),
            "open": np.random.uniform(100, 110, 10),
            "high": np.random.uniform(110, 120, 10),
            "low": np.random.uniform(90, 100, 10),
            "close": np.random.uniform(100, 115, 10),
            "volume": np.random.randint(1000, 2000, 10),
        }
        df = pd.DataFrame(data)
        # Wprowadź sztuczne braki danych
        df.loc[3, "close"] = None
        # Pipeline przetwarzania
        df_processed = preprocess_pipeline(
            df,
            price_col="close",
            fill_method="median",
            outlier_threshold=3.0,
            winsorize_limits=(0.05, 0.05),
        )
        # Sprawdzenie, czy kolumna 'log_return' jest obecna
        assert (
            "log_return" in df_processed.columns
        ), "Brak kolumny 'log_return' po przetwarzaniu."
        logging.info("Testy jednostkowe dla data_preprocessing.py zakończone sukcesem.")
    except AssertionError as ae:
        logging.error("Błąd w testach jednostkowych: %s", ae)
    except Exception as e:
        logging.error("Nieoczekiwany błąd podczas testów jednostkowych: %s", e)
        raise


if __name__ == "__main__":
    try:
        unit_test_preprocessing()
    except Exception as e:
        logging.error("Testy jednostkowe nie powiodły się: %s", e)
        raise

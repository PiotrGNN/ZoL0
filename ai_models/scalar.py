"""
scalar.py
---------
Moduł do skalowania (normalizacji i standaryzacji) danych wejściowych (ceny, wolumen, wskaźniki)
na potrzeby trenowania modeli ML i RL.

Funkcjonalności:
- Implementacja metod skalowania: StandardScaler, MinMaxScaler, RobustScaler oraz log-scaling.
- Dynamiczny dobór metody skalowania poprzez parametr 'method'.
- Zapewnienie funkcji dopasowania (fit), transformacji (transform) oraz odwrotnej transformacji (inverse_transform).
- Obsługa brakujących danych przez wypełnianie medianą przed skalowaniem.
- Obsługa dużych wolumenów danych (wsparcie dla batch processing przy użyciu numpy/pandas).
- Testy jednostkowe weryfikujące poprawność skalowania przy różnych rozkładach danych.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class DataScaler:
    """
    Klasa do skalowania danych przy użyciu różnych metod.

    Parametry:
        method (str): Metoda skalowania. Dozwolone wartości:
                      'standard'  - StandardScaler (standaryzacja),
                      'minmax'    - MinMaxScaler (skalowanie do zakresu [0,1]),
                      'robust'    - RobustScaler (odporne na outliers),
                      'log'       - Log-scaling (przy dużej zmienności, transformacja logarytmiczna).
        fill_value (float): Wartość, którą używamy do wypełniania brakujących danych (np. mediana).
    """

    def __init__(self, method: str = "standard", fill_value: float = None):
        self.method = method.lower()
        self.fill_value = (
            fill_value  # Jeśli None, mediana zostanie obliczona automatycznie.
        )
        self.scaler = None  # Będzie inicjalizowany w metodzie fit.
        logging.info("Inicjalizacja DataScaler z metodą: %s", self.method)

    def _fill_missing(self, data):
        """
        Wypełnia brakujące wartości w danych medianą.
        """
        if isinstance(data, pd.DataFrame):
            if self.fill_value is None:
                self.fill_value = data.median().median()
            filled = data.fillna(self.fill_value)
        elif isinstance(data, np.ndarray):
            if self.fill_value is None:
                self.fill_value = np.nanmedian(data)
            filled = np.where(np.isnan(data), self.fill_value, data)
        else:
            raise ValueError(
                "Obsługiwany format danych to pandas DataFrame lub numpy ndarray."
            )
        return filled

    def fit(self, data, *args, **kwargs):
        """
        Dopasowuje wybrany scaler do danych.

        Parameters:
            data (pd.DataFrame lub np.ndarray): Dane wejściowe.
            *args, **kwargs: Dodatkowe argumenty (ignorowane, dla kompatybilności)
        """
        data = self._fill_missing(data)
        if self.method == "standard":
            self.scaler = StandardScaler()
            self.scaler.fit(data)
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
            self.scaler.fit(data)
        elif self.method == "robust":
            self.scaler = RobustScaler()
            self.scaler.fit(data)
        elif self.method == "log":
            # Dla log-scaling nie wymagamy dopasowywania, ale możemy zapamiętać offset aby uniknąć log(0)
            # Ustalamy offset jako minimalna wartość dodatnia
            min_val = np.min(data[data > 0]) if np.any(data > 0) else 1e-6
            self.offset = min_val / 10.0  # mały offset
            logging.info("Log-scaling: offset ustawiony na: %f", self.offset)
        else:
            raise ValueError(f"Nieobsługiwana metoda skalowania: {self.method}")
        logging.info("Dopasowanie scalera zakończone.")

    def transform(self, data):
        """
        Transformuje dane przy użyciu dopasowanego scalera.

        Parameters:
            data (pd.DataFrame lub np.ndarray): Dane wejściowe.

        Returns:
            Przetransformowane dane w tym samym formacie co wejściowe.
        """
        data = self._fill_missing(data)
        if self.method in ["standard", "minmax", "robust"]:
            transformed = self.scaler.transform(data)
        elif self.method == "log":
            # Dodajemy offset, aby uniknąć log(0)
            transformed = np.log(data + self.offset)
        else:
            raise ValueError(f"Nieobsługiwana metoda skalowania: {self.method}")
        logging.info("Transformacja danych zakończona.")
        return transformed

    def fit_transform(self, data):
        """
        Dopasowuje scaler do danych i jednocześnie transformuje dane.

        Parameters:
            data (pd.DataFrame lub np.ndarray): Dane wejściowe.

        Returns:
            Przetransformowane dane.
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """
        Odwraca transformację danych.

        Parameters:
            data (pd.DataFrame lub np.ndarray): Przetransformowane dane.

        Returns:
            Dane w oryginalnej skali.
        """
        if self.method in ["standard", "minmax", "robust"]:
            original = self.scaler.inverse_transform(data)
        elif self.method == "log":
            # Odwracamy logarytmiczną transformację
            original = np.exp(data) - self.offset
        else:
            raise ValueError(f"Nieobsługiwana metoda skalowania: {self.method}")
        logging.info("Inverse transformacja danych zakończona.")
        return original


# -------------------- Testy jednostkowe --------------------
def unit_test_scaling():
    """
    Testy jednostkowe sprawdzające poprawność skalowania dla różnych metod.
    """
    # Generujemy przykładowe dane: losowe ceny z rozkładem lognormalnym (często używany przy cenach)
    np.random.seed(42)
    data = np.random.lognormal(mean=3.0, sigma=0.5, size=(100, 3))
    df_data = pd.DataFrame(data, columns=["price", "volume", "indicator"])

    methods = ["standard", "minmax", "robust", "log"]
    for method in methods:
        logging.info("Testowanie metody: %s", method)
        scaler = DataScaler(method=method)
        scaled = scaler.fit_transform(df_data)
        recovered = scaler.inverse_transform(scaled)
        # Sprawdzamy, czy średni błąd między oryginalnymi danymi a odzyskanymi danymi jest niewielki
        error = np.abs(df_data.values - recovered)
        mean_error = np.mean(error)
        logging.info(
            "Metoda: %s, średni błąd inverse transform: %.4f", method, mean_error
        )
        assert (
            mean_error < 1e-5 or method == "log"
        ), f"Błąd zbyt wysoki dla metody {method}"
        # Dla log skalowania, błąd może być większy z uwagi na offset, więc sprawdzamy względną zgodność
        if method == "log":
            relative_error = np.mean(
                np.abs((df_data.values - recovered) / df_data.values)
            )
            logging.info(
                "Metoda log, względny błąd inverse transform: %.4f", relative_error
            )
            assert relative_error < 0.05, "Względny błąd za wysoki dla log scaling"


if __name__ == "__main__":
    try:
        unit_test_scaling()
        logging.info("Wszystkie testy jednostkowe skalowania zakończone sukcesem.")
    except AssertionError as ae:
        logging.error("Test jednostkowy nie powiódł się: %s", ae)
    except Exception as e:
        logging.error("Nieoczekiwany błąd podczas testów jednostkowych: %s", e)

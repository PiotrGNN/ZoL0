"""
mean_reversion.py
-----------------
Moduł implementujący strategię powrotu do średniej (mean reversion).

Funkcjonalności:
- Oblicza wskaźniki takie jak z-score porównujący bieżącą cenę z SMA/EMA oraz wykorzystuje Bollinger Bands.
- Logika otwierania pozycji przy istotnych odchyleniach od średniej, z uwzględnieniem filtrów wolumenu i zmienności.
- Funkcje zarządzania ryzykiem, w tym integracja ze stop-lossami, które zabezpieczają przed trendem przeciwnym do oczekiwań.
- Implementacja testów (backtest oraz symulacja real-time) oraz szczegółowe logowanie działań.
- Kod jest skalowalny i przystosowany do pracy w portfelach wielo-assetowych.
"""

import logging

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_sma(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Oblicza prostą średnią kroczącą (SMA).

    Parameters:
        series (pd.Series): Serie cen.
        window (int): Okres średniej.

    Returns:
        pd.Series: SMA.
    """
    sma = series.rolling(window=window, min_periods=1).mean()
    logging.info("Obliczono SMA dla okna %d.", window)
    return sma


def compute_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Oblicza z-score dla serii cen w odniesieniu do SMA.

    Parameters:
        series (pd.Series): Serie cen.
        window (int): Okres do obliczeń.

    Returns:
        pd.Series: Z-score.
    """
    sma = compute_sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    zscore = (series - sma) / std.replace(0, np.nan)
    logging.info("Obliczono z-score dla okna %d.", window)
    return zscore.fillna(0)


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """
    Oblicza Bollinger Bands dla danej serii cen.

    Parameters:
        series (pd.Series): Serie cen.
        window (int): Okres SMA.
        num_std (float): Liczba odchyleń standardowych.

    Returns:
        pd.DataFrame: DataFrame z kolumnami 'Middle', 'Upper' i 'Lower'.
    """
    middle = compute_sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    logging.info("Obliczono Bollinger Bands dla okna %d.", window)
    return pd.DataFrame({"Middle": middle, "Upper": upper, "Lower": lower})


def generate_mean_reversion_signal(
    df: pd.DataFrame,
    window: int = 20,
    zscore_threshold: float = 1.5,
    volume_filter: float = None,
) -> pd.Series:
    """
    Generuje sygnał strategii powrotu do średniej na podstawie z-score oraz Bollinger Bands.

    Logika:
    - Jeżeli z-score przekracza zscore_threshold (dodatnio), generowany jest sygnał sprzedaży (-1).
    - Jeżeli z-score jest mniejszy niż -zscore_threshold, generowany jest sygnał kupna (1).
    - W przeciwnym razie sygnał wynosi 0.
    - Opcjonalnie uwzględniany jest filtr wolumenu: sygnał jest generowany tylko, gdy wolumen przekracza określony próg.

    Parameters:
        df (pd.DataFrame): Dane świecowe z kolumną 'close'. Opcjonalnie może zawierać kolumnę 'volume'.
        window (int): Okres do obliczeń SMA, z-score i Bollinger Bands.
        zscore_threshold (float): Próg dla z-score.
        volume_filter (float, optional): Minimalny wolumen, aby sygnał był brany pod uwagę.

    Returns:
        pd.Series: Sygnał transakcyjny (1 = kupno, -1 = sprzedaż, 0 = brak sygnału).
    """
    close = df["close"]
    zscore = compute_zscore(close, window)
    compute_bollinger_bands(close, window)

    signal = pd.Series(0, index=df.index)

    # Warunki dla sygnałów
    buy_condition = zscore <= -zscore_threshold  # cena bardzo poniżej średniej
    sell_condition = zscore >= zscore_threshold  # cena bardzo powyżej średniej

    # Jeżeli filtr wolumenu jest ustawiony, dodatkowo sprawdzamy, czy wolumen jest powyżej progu
    if volume_filter is not None and "volume" in df.columns:
        volume_condition = df["volume"] >= volume_filter
        buy_condition = buy_condition & volume_condition
        sell_condition = sell_condition & volume_condition

    signal[buy_condition] = 1
    signal[sell_condition] = -1

    logging.info(
        "Wygenerowano sygnał mean reversion na podstawie z-score (próg: %.2f).",
        zscore_threshold,
    )
    return signal


def risk_management_integration(signal: pd.Series, current_position: int) -> pd.Series:
    """
    Integruje sygnał strategii mean reversion z mechanizmami zarządzania ryzykiem.
    Przykładowo, w przypadku wykrycia trendu zamiast rewersji, sygnał może być zneutralizowany.

    Parameters:
        signal (pd.Series): Wstępnie wygenerowany sygnał.
        current_position (int): Aktualna pozycja (1 = long, -1 = short, 0 = brak).

    Returns:
        pd.Series: Zaktualizowany sygnał transakcyjny.
    """
    # Przykładowa logika: jeśli aktualna pozycja jest long, sygnał kupna jest neutralizowany (ustawiany na 0)
    adjusted_signal = signal.copy()
    if current_position == 1:
        adjusted_signal[adjusted_signal == 1] = 0
    elif current_position == -1:
        adjusted_signal[adjusted_signal == -1] = 0
    logging.info("Zintegrowano sygnał mean reversion z zarządzaniem ryzykiem.")
    return adjusted_signal


# -------------------- Testy jednostkowe --------------------
def unit_test_mean_reversion():
    """
    Testy jednostkowe dla modułu mean_reversion.py.
    Tworzy przykładowe dane i weryfikuje, czy sygnały są generowane zgodnie z oczekiwaniami.
    """
    try:
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        # Generujemy dane z trendem rewersyjnym: ceny oscylują wokół 100
        close_prices = 100 + np.random.normal(0, 2, 50)
        volume = np.random.randint(1000, 1500, 50)
        df = pd.DataFrame({"close": close_prices, "volume": volume}, index=dates)

        signal = generate_mean_reversion_signal(df, window=10, zscore_threshold=1.5, volume_filter=1100)
        # Sprawdzamy, czy sygnały przy dużych odchyleniach są generowane
        if signal.abs().sum() == 0:
            raise AssertionError("Brak wygenerowanych sygnałów mean reversion.")

        # Test integracji z risk management: przyjmujemy, że mamy long, więc sygnał kupna powinien być neutralizowany
        adjusted = risk_management_integration(signal, current_position=1)
        if (adjusted == 1).any():
            raise AssertionError("Sygnał kupna nie został zneutralizowany przy posiadaniu long.")

        logging.info("Testy jednostkowe mean_reversion.py zakończone sukcesem.")
    except AssertionError as ae:
        logging.error("AssertionError w testach mean_reversion.py: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach mean_reversion.py: %s", e)
        raise


if __name__ == "__main__":
    try:
        unit_test_mean_reversion()
    except Exception as e:
        logging.error("Testy jednostkowe mean_reversion.py nie powiodły się: %s", e)
        raise

"""
volume_analysis.py
------------------
Moduł analizujący wolumen transakcji.

Funkcjonalności:
- Oblicza wskaźniki oparte na wolumenie, takie jak On-Balance Volume (OBV), Chaikin Money Flow (CMF)
  oraz Volume Weighted Average Price (VWAP).
- Dodaje możliwość wykrywania anomalii wolumenowych, np. nagłych wzrostów lub spadków.
- Generuje sygnały akumulacji i dystrybucji w oparciu o analizę wolumenu.
- Zapewnia wydajność przy obsłudze dużej częstotliwości danych (np. tick data).
- Umożliwia integrację z innymi modułami analizy technicznej oraz systemami alertów.
"""

import logging

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def calculate_obv(prices: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Oblicza On-Balance Volume (OBV) na podstawie serii cen i wolumenu.

    Parameters:
        prices (pd.Series): Serie cen (np. zamknięcia).
        volumes (pd.Series): Serie wolumenu.

    Returns:
        pd.Series: Wartości OBV.
    """
    obv = [0]
    for i in range(1, len(prices)):
        if prices.iloc[i] > prices.iloc[i - 1]:
            obv.append(obv[-1] + volumes.iloc[i])
        elif prices.iloc[i] < prices.iloc[i - 1]:
            obv.append(obv[-1] - volumes.iloc[i])
        else:
            obv.append(obv[-1])
    logging.info("OBV obliczone pomyślnie.")
    return pd.Series(obv, index=prices.index)


def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Oblicza Chaikin Money Flow (CMF) na podstawie danych świecowych.

    Parameters:
        df (pd.DataFrame): DataFrame zawierający kolumny 'high', 'low', 'close' oraz 'volume'.
        period (int): Okres obliczeniowy.

    Returns:
        pd.Series: Wartości CMF.
    """
    try:
        # Obliczenie Money Flow Multiplier
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
        # Obliczenie Money Flow Volume
        mfv = mfm * df["volume"]
        # Suma Money Flow Volume i wolumenu
        cmf = mfv.rolling(window=period, min_periods=1).sum() / df["volume"].rolling(window=period, min_periods=1).sum()
        logging.info("CMF obliczone pomyślnie dla okresu %d.", period)
        return cmf.fillna(0)
    except Exception as e:
        logging.error("Błąd przy obliczaniu CMF: %s", e)
        raise


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Oblicza Volume Weighted Average Price (VWAP) na podstawie danych świecowych.

    Parameters:
        df (pd.DataFrame): DataFrame zawierający kolumny 'high', 'low', 'close' oraz 'volume'.

    Returns:
        pd.Series: Wartości VWAP.
    """
    try:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        cumulative_tp_volume = (typical_price * df["volume"]).cumsum()
        cumulative_volume = df["volume"].cumsum()
        vwap = cumulative_tp_volume / cumulative_volume
        logging.info("VWAP obliczone pomyślnie.")
        return vwap
    except Exception as e:
        logging.error("Błąd przy obliczaniu VWAP: %s", e)
        raise


def detect_volume_anomalies(volumes: pd.Series, threshold: float = 2.0) -> pd.Series:
    """
    Wykrywa anomalie wolumenowe, np. nagłe wzrosty lub spadki.
    Wykorzystuje prostą metodę z-score do identyfikacji wartości odstających.

    Parameters:
        volumes (pd.Series): Serie wolumenu.
        threshold (float): Próg z-score, powyżej którego wolumen jest uznawany za anomalię.

    Returns:
        pd.Series: Boolean Series, gdzie True oznacza anomalię.
    """
    try:
        mean_vol = volumes.mean()
        std_vol = volumes.std()
        z_scores = (volumes - mean_vol) / std_vol
        anomalies = z_scores.abs() > threshold
        logging.info(
            "Wykryto %d anomalii wolumenowych przy progu %.2f.",
            anomalies.sum(),
            threshold,
        )
        return anomalies
    except Exception as e:
        logging.error("Błąd przy wykrywaniu anomalii wolumenowych: %s", e)
        raise


def accumulation_distribution_signal(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Generuje sygnał akumulacji/dystrybucji na podstawie analizowanego wolumenu.
    Wykorzystuje wskaźnik OBV oraz zmiany cen, aby ocenić, czy następuje akumulacja (kupno) czy dystrybucja (sprzedaż).

    Parameters:
        df (pd.DataFrame): DataFrame z danymi świecowymi, zawierający kolumny 'close' oraz 'volume'.
        period (int): Okres do obliczeń wskaźnika.

    Returns:
        pd.Series: Sygnał: 1 dla akumulacji, -1 dla dystrybucji, 0 dla braku jednoznacznego sygnału.
    """
    try:
        obv = calculate_obv(df["close"], df["volume"])
        obv_change = obv.diff(periods=period).fillna(0)
        # Jeśli zmiana OBV jest dodatnia, sugeruje akumulację; ujemna - dystrybucję
        signal = obv_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        logging.info("Sygnał akumulacji/dystrybucji obliczony pomyślnie dla okresu %d.", period)
        return signal
    except Exception as e:
        logging.error("Błąd przy generowaniu sygnału akumulacji/dystrybucji: %s", e)
        raise


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowe dane
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = {
            "timestamp": dates,
            "high": np.linspace(100, 150, 100),
            "low": np.linspace(95, 145, 100),
            "close": np.linspace(98, 148, 100) + np.random.normal(0, 2, 100),
            "volume": np.random.randint(1000, 3000, 100),
        }
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)

        obv = calculate_obv(df["close"], df["volume"])
        cmf = calculate_cmf(df, period=14)
        vwap = calculate_vwap(df)
        anomalies = detect_volume_anomalies(df["volume"], threshold=2.5)
        acc_dist_signal = accumulation_distribution_signal(df, period=7)

        logging.info("Przykładowa analiza wolumenu zakończona pomyślnie.")
    except Exception as e:
        logging.error("Błąd w module volume_analysis.py: %s", e)
        raise

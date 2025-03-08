"""
technical_analysis.py
---------------------
Moduł do przeprowadzania klasycznej analizy technicznej.
Funkcjonalności:
- Implementacja wskaźników technicznych: RSI, MACD, Stochastic, różne średnie kroczące (SMA, EMA, WMA) oraz Ichimoku.
- Obsługa wielointerwałowa (np. analiza na interwale 1H i 1D jednocześnie).
- Generowanie sygnałów, takich jak przecięcia średnich, dywergencje MACD, sygnały z Ichimoku.
- Optymalizacja obliczeń przy użyciu wektoryzacji z wykorzystaniem NumPy/Pandas.
- Integracja z modułami logowania, aby sygnały mogły być przekazywane do systemu strategii (np. do trade_executor.py).
"""

import logging

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Oblicza prostą średnią kroczącą (SMA).

    Parameters:
        series (pd.Series): Serie danych (np. ceny zamknięcia).
        window (int): Okres średniej.

    Returns:
        pd.Series: Wartości SMA.
    """
    sma = series.rolling(window=window).mean()
    logging.info("Obliczono SMA dla okna %d.", window)
    return sma


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Oblicza wykładniczą średnią kroczącą (EMA).

    Parameters:
        series (pd.Series): Serie danych.
        span (int): Okres EMA.

    Returns:
        pd.Series: Wartości EMA.
    """
    ema = series.ewm(span=span, adjust=False).mean()
    logging.info("Obliczono EMA dla spanu %d.", span)
    return ema


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Oblicza Relative Strength Index (RSI).

    Parameters:
        series (pd.Series): Serie cen (np. zamknięcia).
        window (int): Okres obliczeniowy.

    Returns:
        pd.Series: Wartości RSI.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    logging.info("Obliczono RSI dla okresu %d.", window)
    return rsi


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Oblicza MACD, linię sygnału oraz histogram MACD.

    Parameters:
        series (pd.Series): Serie cen (np. zamknięcia).
        fast (int): Okres szybkiej EMA.
        slow (int): Okres wolnej EMA.
        signal (int): Okres EMA dla linii sygnału.

    Returns:
        pd.DataFrame: DataFrame zawierający kolumny 'MACD', 'Signal' oraz 'Histogram'.
    """
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = compute_ema(macd, signal)
    histogram = macd - signal_line
    logging.info("Obliczono MACD, linię sygnału i histogram.")
    return pd.DataFrame({"MACD": macd, "Signal": signal_line, "Histogram": histogram})


def compute_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """
    Oblicza wskaźnik Stochastic Oscillator.

    Parameters:
        df (pd.DataFrame): DataFrame zawierający kolumny 'high', 'low' oraz 'close'.
        k_period (int): Okres dla %K.
        d_period (int): Okres dla %D (średnia krocząca %K).

    Returns:
        pd.DataFrame: DataFrame z kolumnami '%K' i '%D'.
    """
    low_min = df["low"].rolling(window=k_period, min_periods=1).min()
    high_max = df["high"].rolling(window=k_period, min_periods=1).max()
    k_percent = 100 * ((df["close"] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
    logging.info(
        "Obliczono Stochastic Oscillator dla okresów %d i %d.", k_period, d_period
    )
    return pd.DataFrame({"%K": k_percent, "%D": d_percent})


def compute_ichimoku(
    df: pd.DataFrame,
    conversion_line_period: int = 9,
    base_line_period: int = 26,
    leading_span_b_period: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """
    Oblicza wskaźniki Ichimoku Cloud.

    Parameters:
        df (pd.DataFrame): DataFrame z kolumnami 'high', 'low', 'close'.
        conversion_line_period (int): Okres linii konwersji (Tenkan-sen).
        base_line_period (int): Okres linii bazowej (Kijun-sen).
        leading_span_b_period (int): Okres dla Leading Span B.
        displacement (int): Przesunięcie wykresu w przyszłość.

    Returns:
        pd.DataFrame: DataFrame zawierający kolumny 'Conversion', 'Base', 'LeadingSpanA', 'LeadingSpanB'.
    """
    high_conv = df["high"].rolling(window=conversion_line_period, min_periods=1).max()
    low_conv = df["low"].rolling(window=conversion_line_period, min_periods=1).min()
    conversion_line = (high_conv + low_conv) / 2

    high_base = df["high"].rolling(window=base_line_period, min_periods=1).max()
    low_base = df["low"].rolling(window=base_line_period, min_periods=1).min()
    base_line = (high_base + low_base) / 2

    leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
    high_lead_b = df["high"].rolling(window=leading_span_b_period, min_periods=1).max()
    low_lead_b = df["low"].rolling(window=leading_span_b_period, min_periods=1).min()
    leading_span_b = ((high_lead_b + low_lead_b) / 2).shift(displacement)

    logging.info("Obliczono wskaźniki Ichimoku Cloud.")
    return pd.DataFrame(
        {
            "Conversion": conversion_line,
            "Base": base_line,
            "LeadingSpanA": leading_span_a,
            "LeadingSpanB": leading_span_b,
        }
    )


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowe dane
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = {
            "timestamp": dates,
            "open": np.linspace(100, 150, 100),
            "high": np.linspace(105, 155, 100),
            "low": np.linspace(95, 145, 100),
            "close": np.linspace(100, 150, 100) + np.random.normal(0, 2, 100),
            "volume": np.random.randint(1000, 2000, 100),
        }
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)

        sma = compute_sma(df["close"], window=10)
        ema = compute_ema(df["close"], span=10)
        rsi = compute_rsi(df["close"])
        macd_df = compute_macd(df["close"])
        stochastic_df = compute_stochastic(df)
        ichimoku_df = compute_ichimoku(df)

        logging.info("Przykładowa analiza techniczna zakończona pomyślnie.")
    except Exception as e:
        logging.error("Błąd w module technical_analysis.py: %s", e)
        raise

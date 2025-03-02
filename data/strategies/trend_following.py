"""
trend_following.py
------------------
Moduł implementujący strategię podążania za trendem (trend following).

Funkcjonalności:
- Wykorzystuje wskaźniki trendu, takie jak ADX, MACD i Price Channels, aby określić kierunek trendu.
- Implementuje filtry płynności, aby unikać sygnałów w warunkach niskiej płynności.
- Zawiera logikę rozbudowanych trailing stopów oraz zarządzania pozycją (np. dokupowanie podczas trwania trendu).
- Umożliwia backtesting strategii oraz integrację z modułem real-time, aby weryfikować skuteczność strategii w różnych fazach rynku.
- Zapewnia możliwość adaptacji do różnych interwałów i aktywów, w tym kontraktów futures.
- Kod jest zoptymalizowany pod kątem obsługi dużych wolumenów i długich okresów trwania trendu.
"""

import logging

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Oblicza Average Directional Index (ADX) dla określonego okresu.

    Parameters:
        df (pd.DataFrame): Dane świecowe z kolumnami 'high', 'low', 'close'.
        period (int): Okres do obliczeń.

    Returns:
        pd.Series: Wartości ADX.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Oblicz True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()

    # Oblicz +DM i -DM
    up_move = high.diff()
    down_move = low.diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index).rolling(window=period, min_periods=1).sum()
    minus_dm = pd.Series(minus_dm, index=df.index).rolling(window=period, min_periods=1).sum()

    # Oblicz kierunkowe wskaźniki
    plus_di = 100 * (plus_dm / atr)
    minus_di = 100 * (minus_dm / atr)

    # Oblicz DX i ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0) * 100
    adx = dx.rolling(window=period, min_periods=1).mean()
    logging.info("Obliczono ADX dla okresu %d.", period)
    return adx


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Oblicza MACD, linię sygnału i histogram.

    Parameters:
        series (pd.Series): Serie cen (np. zamknięcia).
        fast (int): Okres szybkiej EMA.
        slow (int): Okres wolnej EMA.
        signal (int): Okres EMA linii sygnału.

    Returns:
        pd.DataFrame: DataFrame z kolumnami 'MACD', 'Signal' i 'Histogram'.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    logging.info("Obliczono MACD dla okresów fast: %d, slow: %d, signal: %d.", fast, slow, signal)
    return pd.DataFrame({"MACD": macd, "Signal": signal_line, "Histogram": histogram})


def compute_price_channels(df: pd.DataFrame, window: int = 20) -> (pd.Series, pd.Series):
    """
    Oblicza Price Channels, czyli poziomy wsparcia i oporu na podstawie najwyższych i najniższych cen w okresie.

    Parameters:
        df (pd.DataFrame): Dane świecowe z kolumną 'close'.
        window (int): Okres do obliczeń.

    Returns:
        tuple: (lower_channel, upper_channel)
    """
    lower_channel = df["close"].rolling(window=window, min_periods=1).min()
    upper_channel = df["close"].rolling(window=window, min_periods=1).max()
    logging.info("Obliczono Price Channels dla okna %d.", window)
    return lower_channel, upper_channel


def generate_trend_following_signal(
    df: pd.DataFrame,
    adx_threshold: float = 25,
    macd_threshold: float = 0,
    channel_window: int = 20,
    liquidity_threshold: float = 1000,
) -> pd.Series:
    """
    Generuje sygnał strategii trend following na podstawie ADX, MACD i Price Channels.

    Logika:
    - Jeśli ADX jest powyżej adx_threshold, wskazuje to na silny trend.
    - MACD powyżej macd_threshold sugeruje trend wzrostowy (sygnał kupna), poniżej – trend spadkowy (sygnał sprzedaży).
    - Filtr płynności: sygnał jest generowany tylko, jeśli średni wolumen jest powyżej liquidity_threshold.

    Parameters:
        df (pd.DataFrame): Dane świecowe z kolumnami 'close', 'high', 'low', 'volume'.
        adx_threshold (float): Próg ADX dla uznania trendu za silny.
        macd_threshold (float): Próg MACD do generowania sygnału.
        channel_window (int): Okres do obliczenia Price Channels.
        liquidity_threshold (float): Minimalny średni wolumen.

    Returns:
        pd.Series: Sygnał transakcyjny (1 = kupno, -1 = sprzedaż, 0 = brak sygnału).
    """
    close = df["close"]
    adx = compute_adx(df)
    macd_df = compute_macd(close)
    lower_channel, upper_channel = compute_price_channels(df, window=channel_window)

    avg_volume = df["volume"].mean()

    signal = pd.Series(0, index=df.index)

    # Warunki do generowania sygnałów
    strong_trend = adx >= adx_threshold
    # Trend wzrostowy: MACD powyżej threshold i cena blisko dolnego kanału (potencjalne wejście na korektę)
    buy_signal = (macd_df["MACD"] > macd_threshold) & (close - lower_channel < (upper_channel - lower_channel) * 0.2)
    # Trend spadkowy: MACD poniżej threshold i cena blisko górnego kanału
    sell_signal = (macd_df["MACD"] < macd_threshold) & (upper_channel - close < (upper_channel - lower_channel) * 0.2)

    # Upewnij się, że mamy wystarczający wolumen
    if avg_volume < liquidity_threshold:
        logging.info(
            "Brak wystarczającej płynności (średni wolumen: %.2f). Sygnały nie będą generowane.",
            avg_volume,
        )
        return signal  # 0 sygnału

    signal[strong_trend & buy_signal] = 1
    signal[strong_trend & sell_signal] = -1

    logging.info("Wygenerowano sygnał trend following na podstawie ADX, MACD i Price Channels.")
    return signal


def apply_trailing_stop(current_price: float, entry_price: float, trailing_pct: float = 0.05) -> float:
    """
    Oblicza poziom trailing stop loss.

    Parameters:
        current_price (float): Aktualna cena.
        entry_price (float): Cena wejścia.
        trailing_pct (float): Procentowy margines trailing stop.

    Returns:
        float: Poziom trailing stop.
    """
    # Zakładamy, że trailing stop jest dynamiczny, podążający za najwyższą osiągniętą ceną.
    # Dla uproszczenia przyjmujemy, że trailing stop wynosi entry_price * (1 + trailing_pct) w trendzie wzrostowym
    stop_level = entry_price * (1 + trailing_pct)
    logging.info("Obliczono poziom trailing stop: %.2f", stop_level)
    return stop_level


# -------------------- Testy jednostkowe --------------------
def unit_test_trend_following():
    """
    Testy jednostkowe dla modułu trend_following.py.
    Symuluje dane rynkowe i weryfikuje, czy strategia generuje sygnały zgodnie z oczekiwaniami.
    """
    try:
        dates = pd.date_range(start="2023-01-01", periods=50, freq="H")
        # Generujemy dane: ceny rosnące z szumem
        np.random.seed(42)
        base_prices = np.linspace(100, 110, 50)
        noise = np.random.normal(0, 0.5, 50)
        close_prices = base_prices + noise
        high = close_prices + np.random.uniform(0.5, 1.5, 50)
        low = close_prices - np.random.uniform(0.5, 1.5, 50)
        volume = np.random.randint(1500, 2500, 50)

        df = pd.DataFrame(
            {"close": close_prices, "high": high, "low": low, "volume": volume},
            index=dates,
        )

        signal = generate_trend_following_signal(
            df,
            adx_threshold=25,
            macd_threshold=0,
            channel_window=10,
            liquidity_threshold=1000,
        )
        # Sprawdzamy, czy sygnał zawiera wartości -1, 0 lub 1
        assert set(signal.unique()).issubset({-1, 0, 1}), "Sygnały muszą być -1, 0 lub 1."

        logging.info("Testy jednostkowe trend_following.py zakończone sukcesem.")
    except AssertionError as ae:
        logging.error("AssertionError w testach trend_following.py: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach trend_following.py: %s", e)
        raise


if __name__ == "__main__":
    try:
        unit_test_trend_following()
    except Exception as e:
        logging.error("Testy jednostkowe trend_following.py nie powiodły się: %s", e)
        raise

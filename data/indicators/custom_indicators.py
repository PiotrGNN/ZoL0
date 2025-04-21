"""
custom_indicators.py
--------------------
Moduł definiujący niestandardowe wskaźniki techniczne.

Funkcjonalności:
- Obliczanie wskaźników wolumetrycznych, np. On-Balance Volume (OBV) oraz Volume Weighted Average Price (VWAP).
- Implementacja niestandardowych oscylatorów, np. wskaźnika Z-Score dla cen.
- Funkcja hybrydowa łącząca wskaźniki z wykorzystaniem logiki fuzzy, umożliwiająca generowanie hybrydowych sygnałów.
- Integracja z modułami technical_analysis.py i sentiment_analysis.py poprzez udostępnianie funkcji obliczeniowych.
- Moduł jest łatwo rozszerzalny o kolejne wskaźniki oraz umożliwia parametryzację, wspierając automatyczne testy A/B.
- Zadbano o optymalizację wydajności przy dużych wolumenach danych.
"""

import logging

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def calculate_obv(prices: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Oblicza On-Balance Volume (OBV) na podstawie serii cen i wolumenu.

    Parameters:
        prices (pd.Series): Serie cen (np. zamknięcia).
        volumes (pd.Series): Serie wolumenu.

    Returns:
        pd.Series: Wartości OBV.
    """
    try:
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
    except Exception as e:
        logging.error("Błąd przy obliczaniu OBV: %s", e)
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
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        logging.info("VWAP obliczone pomyślnie.")
        return vwap
    except Exception as e:
        logging.error("Błąd przy obliczaniu VWAP: %s", e)
        raise


def calculate_zscore(series: pd.Series) -> pd.Series:
    """
    Oblicza Z-Score dla podanej serii danych.

    Parameters:
        series (pd.Series): Serie danych (np. ceny zamknięcia).

    Returns:
        pd.Series: Z-Score dla każdej wartości w serii.
    """
    try:
        mean_val = series.mean()
        std_val = series.std()
        zscore = (series - mean_val) / std_val
        logging.info("Z-Score obliczony pomyślnie.")
        return zscore
    except Exception as e:
        logging.error("Błąd przy obliczaniu Z-Score: %s", e)
        raise


def fuzzy_hybrid_signal(
    price_series: pd.Series,
    obv_series: pd.Series,
    zscore_series: pd.Series,
    price_threshold: float = 0.01,
    obv_threshold: float = 0.05,
    zscore_threshold: float = 1.0,
) -> pd.Series:
    """
    Generuje hybrydowy sygnał tradingowy przy użyciu logiki fuzzy, łącząc sygnały z ceny, OBV oraz Z-Score.

    Mechanizm:
    - Jeśli zmiana ceny przekracza price_threshold, generowany jest sygnał.
    - Jeśli OBV wykazuje znaczącą zmianę (przekraczając obv_threshold), sygnał jest wzmocniony.
    - Wartość Z-Score powyżej zscore_threshold sugeruje ekstremalne warunki.

    Wynik:
    - Sygnał "1" sugeruje kupno, "-1" sugeruje sprzedaż, "0" brak jednoznacznego sygnału.

    Parameters:
        price_series (pd.Series): Serie cen (np. zamknięcia).
        obv_series (pd.Series): Serie OBV.
        zscore_series (pd.Series): Serie Z-Score dla cen.
        price_threshold (float): Minimalna zmiana ceny, aby wygenerować sygnał.
        obv_threshold (float): Minimalna zmiana OBV, aby wpłynąć na sygnał.
        zscore_threshold (float): Próg Z-Score wskazujący na ekstremalne warunki.

    Returns:
        pd.Series: Hybrydowy sygnał tradingowy.
    """
    try:
        # Obliczenia składowych sygnału
        price_change = price_series.pct_change().fillna(0)
        obv_change = (
            obv_series.diff().fillna(0) / 
            obv_series.replace(0, np.nan)
        ).fillna(0)

        # Wagi wskaźników
        price_weight = 0.5
        obv_weight = 0.3
        zscore_weight = 0.2

        # Sygnały składowe
        price_signal = price_change.apply(
            lambda x: 1 if x > price_threshold else (-1 if x < -price_threshold else 0)
        )
        obv_signal = obv_change.apply(
            lambda x: 1 if x > obv_threshold else (-1 if x < -obv_threshold else 0)
        )
        zscore_signal = zscore_series.apply(
            lambda x: 1 if x > zscore_threshold else (-1 if x < -zscore_threshold else 0)
        )

        # Połączenie sygnałów
        combined_score = (
            price_weight * price_signal +
            obv_weight * obv_signal +
            zscore_weight * zscore_signal
        )

        # Finalna interpretacja
        hybrid_signal = combined_score.apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        
        logging.info("Hybrydowy sygnał tradingowy obliczony pomyślnie.")
        return hybrid_signal
        
    except Exception as e:
        logging.error("Błąd przy generowaniu hybrydowego sygnału: %s", e)
        raise


def calculate_pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Oblicza poziomy pivot points metodą standardową.
    """
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high 
    s2 = pivot - (high - low)
    
    return pd.DataFrame({
        'pivot': pivot,
        'r1': r1,
        'r2': r2,
        's1': s1,
        's2': s2
    })

def calculate_momentum_quality(close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Oblicza wskaźnik jakości momentum uwzględniający cenę i wolumen.
    """
    price_change = close.pct_change(period)
    volume_change = volume.pct_change(period)
    
    # Normalizacja zmian
    norm_price = (price_change - price_change.mean()) / price_change.std()
    norm_volume = (volume_change - volume_change.mean()) / volume_change.std()
    
    # Połączenie sygnałów ceny i wolumenu
    momentum_quality = norm_price * np.where(norm_volume > 0, 1 + norm_volume, 1 / (1 - norm_volume))
    
    return momentum_quality

def calculate_market_depth(bid_volume: pd.Series, ask_volume: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Oblicza wskaźnik głębokości rynku na podstawie wolumenów bid/ask.
    """
    total_volume = bid_volume + ask_volume
    bid_ratio = bid_volume / total_volume
    ask_ratio = ask_volume / total_volume
    
    # Średnia krocząca różnicy między bid i ask
    depth_imbalance = (bid_ratio - ask_ratio).rolling(window=window).mean()
    
    # Wskaźnik płynności
    liquidity_score = (
        total_volume.rolling(window=window).mean() / 
        total_volume.rolling(window=window).std()
    )
    
    return pd.DataFrame({
        'depth_imbalance': depth_imbalance,
        'liquidity_score': liquidity_score,
        'bid_ratio': bid_ratio,
        'ask_ratio': ask_ratio
    })

def calculate_volatility_breakout(high: pd.Series, low: pd.Series, close: pd.Series, 
                                atr_period: int = 14, multiplier: float = 2.0) -> pd.DataFrame:
    """
    Oblicza poziomy przełamania zmienności używając ATR.
    """
    # Obliczenie True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Average True Range
    atr = tr.rolling(window=atr_period).mean()
    
    # Poziomy przełamania
    upper_band = close + (multiplier * atr)
    lower_band = close - (multiplier * atr)
    
    return pd.DataFrame({
        'atr': atr,
        'upper_band': upper_band,
        'lower_band': lower_band
    })


# -------------------- Przykładowe testy jednostkowe --------------------
def unit_test_custom_indicators():
    """
    Testy jednostkowe dla modułu custom_indicators.py.
    Tworzy przykładowe dane i weryfikuje poprawność obliczeń wskaźników.
    """
    try:
        # Przygotowanie przykładowych danych
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        prices = pd.Series(np.linspace(100, 150, 100), index=dates)
        volumes = pd.Series(np.random.randint(1000, 2000, size=100), index=dates)
        df = pd.DataFrame(
            {"close": prices, "high": prices + 5, "low": prices - 5, "volume": volumes}
        )

        # Test OBV
        obv = calculate_obv(prices, volumes)
        assert len(obv) == len(prices), "Długość OBV nie zgadza się z długością cen."

        # Test VWAP
        vwap = calculate_vwap(df)
        assert len(vwap) == len(df), "Długość VWAP nie zgadza się z długością danych."

        # Test Z-Score
        zscore = calculate_zscore(prices)
        assert not zscore.isnull().any(), "Z-Score zawiera wartości null."

        # Test sygnału hybrydowego
        hybrid = fuzzy_hybrid_signal(
            prices,
            obv,
            zscore,
            price_threshold=0.005,
            obv_threshold=0.01,
            zscore_threshold=0.5,
        )
        assert set(hybrid.unique()).issubset(
            {-1, 0, 1}
        ), "Hybrydowy sygnał zawiera nieoczekiwane wartości."

        logging.info(
            "Testy jednostkowe modułu custom_indicators.py zakończone sukcesem."
        )
    except AssertionError as ae:
        logging.error("AssertionError w testach jednostkowych: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach jednostkowych modułu custom_indicators.py: %s", e)
        raise


if __name__ == "__main__":
    try:
        unit_test_custom_indicators()
    except Exception as e:
        logging.error("Testy jednostkowe nie powiodły się: %s", e)
        raise

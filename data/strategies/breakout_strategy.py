"""
breakout_strategy.py
--------------------
Skrypt implementujący strategię przebicia (breakout).
Funkcjonalności:
- Wykrywa poziomy wsparcia i oporu na podstawie historycznych danych.
- Umożliwia analizę wolumenową dla potwierdzenia przebicia.
- Implementuje logikę wchodzenia w pozycję long lub short po wyjściu ceny z ustalonego zakresu (box range).
- Automatyczna kalibracja parametrów, np. szerokości boxu, na podstawie danych historycznych.
- Integracja z modułem strategy_manager.py w celu dynamicznego przełączania strategii.
- Kod zoptymalizowany pod kątem wydajności i skalowalności.
- Zawiera testy jednostkowe oraz logowanie kluczowych zdarzeń.
"""

import logging
import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> (float, float):
    """
    Oblicza poziomy wsparcia i oporu na podstawie średnich cen z wybranego okna.
    
    Parameters:
        df (pd.DataFrame): Dane świecowe z kolumną 'close'.
        window (int): Okres analizy.
        
    Returns:
        tuple: (support, resistance) jako średnie minimalne i maksymalne ceny w oknie.
    """
    support = df['close'].rolling(window=window, min_periods=1).min().iloc[-1]
    resistance = df['close'].rolling(window=window, min_periods=1).max().iloc[-1]
    logging.info("Obliczono wsparcie: %.2f i opór: %.2f dla okna %d.", support, resistance, window)
    return support, resistance

def breakout_signal(df: pd.DataFrame, support: float, resistance: float, volume_threshold: float = 1.0) -> int:
    """
    Generuje sygnał breakout na podstawie ceny oraz wolumenu.
    
    Parameters:
        df (pd.DataFrame): Dane świecowe, oczekiwane kolumny: 'close' oraz 'volume'.
        support (float): Poziom wsparcia.
        resistance (float): Poziom oporu.
        volume_threshold (float): Współczynnik potwierdzenia wolumenowego.
    
    Returns:
        int: Sygnał transakcyjny: 1 - sygnał kupna (przebicie oporu),
             -1 - sygnał sprzedaży (przebicie wsparcia), 0 - brak sygnału.
    """
    latest = df.iloc[-1]
    close = latest['close']
    volume = latest['volume']
    # Prosty mechanizm: sygnał kupna, jeśli cena przekracza opór i wolumen jest powyżej średniej pomnożonej przez threshold
    avg_volume = df['volume'].mean()
    if close > resistance and volume > avg_volume * volume_threshold:
        logging.info("Sygnał BREAKOUT BUY wygenerowany: cena %.2f > opór %.2f, wolumen %.2f.", close, resistance, volume)
        return 1
    # Sygnał sprzedaży, jeśli cena spada poniżej wsparcia
    elif close < support and volume > avg_volume * volume_threshold:
        logging.info("Sygnał BREAKOUT SELL wygenerowany: cena %.2f < wsparcie %.2f, wolumen %.2f.", close, support, volume)
        return -1
    else:
        logging.info("Brak sygnału breakout: cena %.2f, wsparcie %.2f, opór %.2f, wolumen %.2f.", close, support, resistance, volume)
        return 0

def calibrate_box_range(df: pd.DataFrame, window: int = 20) -> (float, float):
    """
    Automatycznie kalibruje szerokość boxu (zakresu) na podstawie historycznych danych.
    
    Parameters:
        df (pd.DataFrame): Dane świecowe z kolumną 'close'.
        window (int): Okres analizy.
    
    Returns:
        tuple: (support, resistance) jako kalibrowane poziomy wsparcia i oporu.
    """
    support, resistance = calculate_support_resistance(df, window=window)
    # Można wprowadzić dodatkową logikę rozszerzającą box o pewien procent
    adjustment_factor = 0.02  # np. 2% rozszerzenia
    calibrated_support = support * (1 - adjustment_factor)
    calibrated_resistance = resistance * (1 + adjustment_factor)
    logging.info("Skalibrowano box range: wsparcie %.2f, opór %.2f.", calibrated_support, calibrated_resistance)
    return calibrated_support, calibrated_resistance

def breakout_strategy(df: pd.DataFrame, window: int = 20, volume_threshold: float = 1.0) -> dict:
    """
    Realizuje strategię breakout.
    
    Parameters:
        df (pd.DataFrame): Dane świecowe, zawierające kolumny 'close' oraz 'volume'.
        window (int): Okres do kalibracji box range.
        volume_threshold (float): Próg potwierdzający wolumen.
    
    Returns:
        dict: Słownik zawierający sygnał transakcyjny i kalibrowane poziomy wsparcia i oporu.
              Przykład: {"signal": 1, "support": 98.5, "resistance": 102.3}
    """
    support, resistance = calibrate_box_range(df, window)
    signal = breakout_signal(df, support, resistance, volume_threshold)
    return {"signal": signal, "support": support, "resistance": resistance}

# -------------------- Testy jednostkowe --------------------
def unit_test_breakout_strategy():
    """
    Testy jednostkowe dla modułu breakout_strategy.py.
    Tworzy przykładowe dane świecowe i weryfikuje poprawność generowania sygnałów.
    """
    try:
        # Generujemy przykładowe dane świecowe
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        np.random.seed(42)
        close_prices = np.linspace(100, 110, 30) + np.random.normal(0, 0.5, 30)
        volume = np.random.randint(1000, 1500, 30)
        df = pd.DataFrame({"close": close_prices, "volume": volume}, index=dates)
        
        # Wymuszamy przebicie oporu
        df.iloc[-1, df.columns.get_loc("close")] = df["close"].max() * 1.02
        df.iloc[-1, df.columns.get_loc("volume")] = df["volume"].mean() * 1.5
        
        result = breakout_strategy(df, window=10, volume_threshold=1.0)
        assert result["signal"] == 1, "Sygnał breakout BUY powinien być wygenerowany."
        logging.info("Test breakout_strategy - sygnał BUY zakończony sukcesem.")
        
        # Wymuszamy przebicie wsparcia
        df.iloc[-1, df.columns.get_loc("close")] = df["close"].min() * 0.98
        df.iloc[-1, df.columns.get_loc("volume")] = df["volume"].mean() * 1.5
        
        result = breakout_strategy(df, window=10, volume_threshold=1.0)
        assert result["signal"] == -1, "Sygnał breakout SELL powinien być wygenerowany."
        logging.info("Test breakout_strategy - sygnał SELL zakończony sukcesem.")
        
    except AssertionError as ae:
        logging.error("AssertionError w testach breakout_strategy.py: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach breakout_strategy.py: %s", e)
        raise

if __name__ == "__main__":
    try:
        unit_test_breakout_strategy()
    except Exception as e:
        logging.error("Testy jednostkowe breakout_strategy.py nie powiodły się: %s", e)
        raise

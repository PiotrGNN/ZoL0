"""
stop_loss_manager.py
--------------------
Moduł zarządzania mechanizmami stop-loss.

Funkcjonalności:
- Obsługa różnych typów stop-loss: fixed, trailing, ATR-based, time-based.
- Integracja z modułami strategii i wykonawczymi.
- Możliwość dynamicznej regulacji poziomów stop-loss w zależności od warunków rynkowych.
"""

import logging

import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def fixed_stop_loss(entry_price: float, stop_loss_percent: float) -> float:
    """
    Oblicza poziom stop-loss jako stały procent od ceny wejścia.

    Parameters:
        entry_price (float): Cena wejścia.
        stop_loss_percent (float): Procent poziomu stop-loss.

    Returns:
        float: Poziom stop-loss.
    """
    if entry_price <= 0 or stop_loss_percent <= 0:
        raise ValueError(
            "Cena wejścia i stop-loss procentowy muszą być większe od zera."
        )

    stop_loss_price = entry_price * (1 - stop_loss_percent / 100)
    logging.info(
        "Fixed Stop-Loss: Entry Price=%.2f, Stop-Loss Percent=%.2f%%, Stop-Loss Price=%.2f",
        entry_price,
        stop_loss_percent,
        stop_loss_price,
    )
    return stop_loss_price


def trailing_stop_loss(
    current_price: float, highest_price: float, trail_percent: float
) -> float:
    """
    Oblicza trailing stop-loss jako dynamiczny próg na podstawie najwyższej ceny.

    Parameters:
        current_price (float): Aktualna cena rynkowa.
        highest_price (float): Najwyższa cena osiągnięta od momentu wejścia w pozycję.
        trail_percent (float): Odstęp trailing stop-loss w procentach.

    Returns:
        float: Nowy poziom stop-loss.
    """
    if current_price <= 0 or highest_price <= 0 or trail_percent <= 0:
        raise ValueError("Ceny i trail percent muszą być większe od zera.")

    stop_price = highest_price * (1 - trail_percent / 100)
    logging.info(
        "Trailing Stop-Loss: Current Price=%.2f, Highest Price=%.2f, Trail Percent=%.2f%%, Stop-Loss Price=%.2f",
        current_price,
        highest_price,
        trail_percent,
        stop_price,
    )
    return stop_price


def atr_based_stop_loss(
    df: pd.DataFrame, atr_multiplier: float = 1.5, atr_period: int = 14
) -> pd.Series:
    """
    Oblicza poziom stop-loss na podstawie ATR (Average True Range).

    Parameters:
        df (pd.DataFrame): Dane rynkowe zawierające kolumny 'high', 'low', 'close'.
        atr_multiplier (float): Współczynnik do obliczania poziomu stop-loss.
        atr_period (int): Okres obliczeniowy ATR.

    Returns:
        pd.Series: Poziomy ATR-based stop-loss.
    """
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dane rynkowe muszą zawierać kolumny: {required_cols}")

    if len(df) < atr_period:
        raise ValueError(
            f"Dane muszą zawierać co najmniej {atr_period} wierszy do obliczenia ATR."
        )

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_period, min_periods=1).mean()

    stop_loss_levels = df["close"] - (atr * atr_multiplier)
    logging.info(
        "ATR-based Stop-Loss: ATR Period=%d, ATR Multiplier=%.2f",
        atr_period,
        atr_multiplier,
    )
    return stop_loss_levels


def time_based_stop_loss(
    entry_time: pd.Timestamp, max_hold_time: pd.Timedelta, current_time: pd.Timestamp
) -> bool:
    """
    Sprawdza, czy minął maksymalny czas trzymania pozycji i czy należy ją zamknąć.

    Parameters:
        entry_time (pd.Timestamp): Czas otwarcia pozycji.
        max_hold_time (pd.Timedelta): Maksymalny czas trzymania pozycji.
        current_time (pd.Timestamp): Aktualny czas.

    Returns:
        bool: True jeśli pozycja powinna zostać zamknięta, False w przeciwnym razie.
    """
    if current_time - entry_time >= max_hold_time:
        logging.info(
            "Time-Based Stop-Loss aktywowany: Entry Time=%s, Current Time=%s, Max Hold Time=%s",
            entry_time,
            current_time,
            max_hold_time,
        )
        return True
    return False


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowe dane rynkowe
        data = {
            "high": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "low": [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        }
        df = pd.DataFrame(data)

        # Testy funkcji
        entry_price = 100
        highest_price = 110
        current_price = 105
        stop_loss_percent = 5
        trail_percent = 2

        print("Fixed Stop-Loss:", fixed_stop_loss(entry_price, stop_loss_percent))
        print(
            "Trailing Stop-Loss:",
            trailing_stop_loss(current_price, highest_price, trail_percent),
        )
        print(
            "ATR-based Stop-Loss (ostatnia wartość):", atr_based_stop_loss(df).iloc[-1]
        )

        # Test Time-Based Stop-Loss
        entry_time = pd.Timestamp("2025-03-04 10:00:00")
        current_time = pd.Timestamp("2025-03-04 15:00:00")
        max_hold_time = pd.Timedelta(hours=4)

        print(
            "Time-Based Stop-Loss:",
            time_based_stop_loss(entry_time, max_hold_time, current_time),
        )

    except Exception as e:
        logging.error("Błąd w module stop_loss_manager.py: %s", e)
        raise

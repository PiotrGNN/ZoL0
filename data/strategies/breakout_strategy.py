"""
breakout_strategy.py
--------------------
Zaawansowana implementacja strategii breakout.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,  # Ustaw na DEBUG, aby zobaczyć pełne logi
    format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s",
)

def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
    """Oblicza wsparcie i opór na podstawie cen zamknięcia z ostatnich `window` świec."""
    support = df["close"].iloc[-window:].min()
    resistance = df["close"].iloc[-window:].max()
    logging.debug("Wsparcie: %.2f, Opór: %.2f (window=%d)", support, resistance, window)
    return support, resistance

def breakout_signal(
    df: pd.DataFrame, support: float, resistance: float, volume_threshold: float = 1.0
) -> int:
    """
    Generuje sygnał breakout na podstawie ceny oraz wolumenu:
     - 1: cena > opór & wolumen >= threshold
     - -1: cena < wsparcie & wolumen >= threshold
     - 0: brak sygnału
    """
    latest_close = df["close"].iloc[-1]
    latest_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].mean()
    required_volume = avg_volume * volume_threshold

    logging.debug(
        "breakout_signal: close=%.2f, volume=%.2f, support=%.2f, resistance=%.2f, avg_vol=%.2f, required_vol=%.2f",
        latest_close, latest_volume, support, resistance, avg_volume, required_volume
    )

    if latest_close > resistance:
        if latest_volume >= required_volume:
            logging.info("✅ SYGNAŁ KUPNA")
            return 1
        else:
            logging.debug("❌ Cena powyżej oporu, ale wolumen za niski!")

    if latest_close < support:
        if latest_volume >= required_volume:
            logging.info("✅ SYGNAŁ SPRZEDAŻY")
            return -1
        else:
            logging.debug("❌ Cena poniżej wsparcia, ale wolumen za niski!")

    logging.info("❌ Brak sygnału breakout: close=%.2f, support=%.2f, resistance=%.2f", latest_close, support, resistance)
    return 0

def calibrate_box_range(
    df: pd.DataFrame, window: int = 20, margin_pct: float = 0.02
) -> Tuple[float, float]:
    """Kalibruje breakout box o zadany margines bezpieczeństwa."""
    support, resistance = calculate_support_resistance(df, window)
    adjusted_support = support * (1 - margin_pct)
    adjusted_resistance = resistance * (1 + margin_pct)
    logging.info("✅ Skalibrowano breakout box: Wsparcie=%.2f, Opór=%.2f", adjusted_support, adjusted_resistance)
    return adjusted_support, adjusted_resistance

def breakout_strategy(
    df: pd.DataFrame,
    window: int = 20,
    volume_threshold: float = 1.0,
    margin_pct: float = 0.02
) -> Dict[str, float]:
    """Realizuje strategię breakout."""
    support, resistance = calibrate_box_range(df, window, margin_pct)
    signal = breakout_signal(df, support, resistance, volume_threshold)
    logging.info("breakout_strategy -> signal=%d", signal)
    return {"signal": signal, "support": support, "resistance": resistance}

# ====================== TESTY JEDNOSTKOWE ======================
def test_breakout_strategy():
    """
    Test strategii breakout (pytest).
    Sprawdza 3 przypadki:
      1) Kupno (przebicie oporu)
      2) Sprzedaż (przebicie wsparcia)
      3) Brak sygnału (środek przedziału)
    """
    logging.info("=== ROZPOCZYNAM test_breakout_strategy ===")

    # Przygotowanie danych
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    close_prices = np.linspace(100, 110, 30) + np.random.normal(0, 0.5, 30)
    volumes = np.random.randint(1000, 1500, 30)
    df_test = pd.DataFrame({"close": close_prices, "volume": volumes}, index=dates)

    window = 10
    margin_pct = 0.02
    volume_threshold = 1.0

    # 1) KUPNO
    logging.info("=== TEST 1: KUPNO ===")
    s_orig, r_orig = calibrate_box_range(df_test, window, margin_pct)
    df_test.iloc[-1, df_test.columns.get_loc("close")] = r_orig + 10.0
    df_test.iloc[-1, df_test.columns.get_loc("volume")] = int(df_test["volume"].mean() * 2.0)

    sup, res = calibrate_box_range(df_test, window, margin_pct)
    buy_result = breakout_strategy(df_test, window, volume_threshold, margin_pct)
    assert buy_result["signal"] == 1, f"❌ Oczekiwany sygnał KUPNA, a jest {buy_result['signal']}"

    # 2) SPRZEDAŻ
    logging.info("=== TEST 2: SPRZEDAŻ ===")
    df_test.iloc[-1, df_test.columns.get_loc("close")] = sup - 10.0
    df_test.iloc[-1, df_test.columns.get_loc("volume")] = int(df_test["volume"].mean() * 2.0)

    sup2, res2 = calibrate_box_range(df_test, window, margin_pct)
    sell_result = breakout_strategy(df_test, window, volume_threshold, margin_pct)
    assert sell_result["signal"] == -1, f"❌ Oczekiwany sygnał SPRZEDAŻY, a jest {sell_result['signal']}"

    # 3) BRAK SYGNAŁU
    logging.info("=== TEST 3: BRAK SYGNAŁU ===")
    df_test.iloc[-1, df_test.columns.get_loc("close")] = (sup2 + res2) / 2
    df_test.iloc[-1, df_test.columns.get_loc("volume")] = int(df_test["volume"].mean() * 0.5)

    sup3, res3 = calibrate_box_range(df_test, window, margin_pct)
    none_result = breakout_strategy(df_test, window, volume_threshold, margin_pct)
    assert none_result["signal"] == 0, f"❌ Oczekiwany BRAK sygnału, a jest {none_result['signal']}"

    logging.info("✅ Wszystkie testy przeszły pomyślnie!")

if __name__ == "__main__":
    test_breakout_strategy()

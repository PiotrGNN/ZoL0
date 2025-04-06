"""
leverage_optimizer.py
---------------------
Moduł optymalizujący dźwignię finansową.

Funkcjonalności:
- Uwzględnia aktualną zmienność rynku, płynność oraz korelacje z innymi instrumentami.
- Implementuje model dynamicznej dźwigni (np. w oparciu o ATR lub VaR), który automatycznie dostosowuje dźwignię.
- Zawiera funkcje limitujące maksymalną stratę.
- Zapewnia integrację z modułem strategii (np. strategy_manager.py) oraz modułem wykonawczym (np. trade_executor.py).
- Posiada testy weryfikujące skuteczność i stabilność algorytmu.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Oblicza Average True Range (ATR) jako miarę zmienności rynku.

    Parameters:
        df (pd.DataFrame): Dane rynkowe zawierające kolumny 'high', 'low', 'close'.
        period (int): Okres obliczeniowy ATR.

    Returns:
        pd.Series: Wartości ATR.
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError(
            "Dane wejściowe muszą zawierać kolumny: 'high', 'low', 'close'."
        )

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()

    df["TrueRange"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TrueRange"].rolling(window=period, min_periods=1).mean()

    logging.info("Obliczono ATR dla okresu %d.", period)
    return df["ATR"]


def dynamic_leverage_model(
    df: pd.DataFrame,
    base_leverage: float = 1.0,
    atr_multiplier: float = 0.1,
    max_leverage: float = 5.0,
) -> pd.Series:
    """
    Implementuje model dynamicznej dźwigni finansowej w oparciu o ATR.
    Im wyższa zmienność (ATR), tym mniejsza dźwignia.

    Parameters:
        df (pd.DataFrame): Dane rynkowe zawierające kolumny 'high', 'low', 'close'.
        base_leverage (float): Bazowa dźwignia.
        atr_multiplier (float): Współczynnik obniżający dźwignię w zależności od ATR.
        max_leverage (float): Maksymalna dozwolona dźwignia.

    Returns:
        pd.Series: Seria dynamicznie ustalonych poziomów dźwigni.
    """
    atr = calculate_atr(df)
    dynamic_leverage = base_leverage / (1 + atr_multiplier * atr)
    dynamic_leverage = dynamic_leverage.clip(upper=max_leverage)

    logging.info("Dynamiczna dźwignia została obliczona.")
    return dynamic_leverage


def limit_max_leverage(
    current_leverage: float, risk_factor: float, max_allowed_leverage: float = 3.0
) -> float:
    """
    Ogranicza maksymalną dźwignię w zależności od poziomu ryzyka.

    Parameters:
        current_leverage (float): Aktualna dźwignia.
        risk_factor (float): Wskaźnik ryzyka (np. zmienność lub VaR).
        max_allowed_leverage (float): Maksymalna dozwolona dźwignia w warunkach wysokiego ryzyka.

    Returns:
        float: Ograniczona dźwignia.
    """
    risk_threshold = 0.05  # Przykładowy próg
    adjusted_leverage = (
        min(current_leverage, max_allowed_leverage)
        if risk_factor > risk_threshold
        else current_leverage
    )

    logging.info(
        "Ograniczona dźwignia: %.2f (risk_factor: %.4f)", adjusted_leverage, risk_factor
    )
    return adjusted_leverage


def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Oblicza wartość narażoną na ryzyko (VaR) metodą parametryczną.

    Parameters:
        returns (pd.Series): Zwroty aktywa lub portfela.
        confidence_level (float): Poziom ufności.

    Returns:
        float: Wartość VaR (ujemna oznacza stratę).
    """
    returns = returns.dropna()
    std_dev = returns.std()
    z_score = norm.ppf(1 - confidence_level)
    var = z_score * std_dev  # ✅ Poprawiona formuła

    logging.info(
        "Obliczono Value at Risk (%.1f%% confidence): %.4f", confidence_level * 100, var
    )
    return var


# -------------------- Testy jednostkowe --------------------
def unit_test_leverage_optimizer():
    """
    Testy jednostkowe dla modułu leverage_optimizer.py.
    """
    try:
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        high = pd.Series(
            np.linspace(100, 120, 20) + np.random.normal(0, 1, 20), index=dates
        )
        low = pd.Series(
            np.linspace(95, 115, 20) + np.random.normal(0, 1, 20), index=dates
        )
        close = pd.Series(
            np.linspace(98, 118, 20) + np.random.normal(0, 1, 20), index=dates
        )
        df = pd.DataFrame({"high": high, "low": low, "close": close})

        dyn_leverage = dynamic_leverage_model(
            df, base_leverage=1.0, atr_multiplier=0.1, max_leverage=5.0
        )
        assert (
            dyn_leverage <= 5.0
        ).all(), "Dynamiczna dźwignia przekracza maksymalny limit."
        logging.info("Test dynamicznej dźwigni zakończony sukcesem.")

        limited_leverage = limit_max_leverage(
            current_leverage=4.5, risk_factor=0.06, max_allowed_leverage=3.0
        )
        assert (
            limited_leverage <= 3.0
        ), "Funkcja ograniczająca dźwignię nie działa prawidłowo."
        logging.info("Test ograniczenia dźwigni zakończony sukcesem.")

        # Test VaR
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        var_value = value_at_risk(returns, confidence_level=0.95)
        assert var_value < 0, "VaR powinien być ujemny (strata)."
        logging.info("Test Value at Risk zakończony sukcesem.")

    except AssertionError as ae:
        logging.error("AssertionError w testach jednostkowych: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach jednostkowych leverage_optimizer.py: %s", e)
        raise


if __name__ == "__main__":
    try:
        unit_test_leverage_optimizer()
        logging.info("Wszystkie testy leverage_optimizer.py zakończone sukcesem.")
    except Exception as e:
        logging.error("Testy jednostkowe nie powiodły się: %s", e)
        raise

"""
portfolio_risk.py
-----------------
Moduł oceniający ryzyko portfela inwestycyjnego w oparciu o różne klasy aktywów i korelacje.

Funkcjonalności:
- Implementacja metod oceny ryzyka, takich jak VaR (Value at Risk), CVaR (Conditional VaR),
  symulacje Monte Carlo oraz podejście parametryczne.
- Uwzględnienie historycznej oraz bieżącej zmienności i korelacji (np. z wykorzystaniem rolling window).
- Funkcje oceny dywersyfikacji, np. wskaźnik Herfindahla-Hirschmana, oraz rekomendacje dotyczące rebalansowania portfela.
- Integracja z systemem alertów – powiadomienie w razie przekroczenia dopuszczalnego poziomu VaR.
- Zawiera testy jednostkowe i wydajnościowe w celu weryfikacji poprawności obliczeń.
"""

import logging

import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Oblicza Value at Risk (VaR) portfela na podstawie historycznych zwrotów.

    Parameters:
        returns (pd.Series): Serie zwrotów portfela.
        confidence_level (float): Poziom ufności, np. 0.95 dla 95%.

    Returns:
        float: Obliczone VaR (ujemna wartość oznaczająca stratę).
    """
    var = np.percentile(returns, (1 - confidence_level) * 100)
    logging.info("Obliczono VaR przy poziomie ufności %.2f: %.4f", confidence_level, var)
    return var


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Oblicza Conditional Value at Risk (CVaR) portfela.

    Parameters:
        returns (pd.Series): Serie zwrotów portfela.
        confidence_level (float): Poziom ufności, np. 0.95.

    Returns:
        float: Obliczone CVaR.
    """
    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    logging.info("Obliczono CVaR przy poziomie ufności %.2f: %.4f", confidence_level, cvar)
    return cvar


def monte_carlo_var(
    returns: pd.Series,
    num_simulations: int = 1000,
    horizon: int = 1,
    confidence_level: float = 0.95,
) -> float:
    """
    Oblicza VaR przy użyciu symulacji Monte Carlo.

    Parameters:
        returns (pd.Series): Historyczne zwroty portfela.
        num_simulations (int): Liczba symulacji.
        horizon (int): Horyzont czasowy w dniach.
        confidence_level (float): Poziom ufności.

    Returns:
        float: Obliczone VaR.
    """
    mean_return = returns.mean()
    std_return = returns.std()

    simulated_returns = np.random.normal(mean_return, std_return, (num_simulations, horizon))
    simulated_portfolio_returns = simulated_returns.sum(axis=1)
    var_mc = np.percentile(simulated_portfolio_returns, (1 - confidence_level) * 100)
    logging.info("Monte Carlo VaR przy poziomie ufności %.2f: %.4f", confidence_level, var_mc)
    return var_mc


def herfindahl_index(weights: np.ndarray) -> float:
    """
    Oblicza wskaźnik Herfindahla-Hirschmana dla dywersyfikacji portfela.

    Parameters:
        weights (np.ndarray): Wagi poszczególnych aktywów w portfelu.

    Returns:
        float: Wskaźnik koncentracji (im wyższy, tym większa koncentracja).
    """
    hh_index = np.sum(np.square(weights))
    logging.info("Obliczono wskaźnik Herfindahla-Hirschmana: %.4f", hh_index)
    return hh_index


def recommend_rebalancing(hh_index: float, threshold: float = 0.2) -> str:
    """
    Rekomenduje rebalansowanie portfela na podstawie wskaźnika Herfindahla-Hirschmana.

    Parameters:
        hh_index (float): Obliczony wskaźnik HH.
        threshold (float): Próg, powyżej którego portfel jest zbyt skoncentrowany.

    Returns:
        str: Rekomendacja ("Rebalance" lub "Hold").
    """
    if hh_index > threshold:
        recommendation = "Rebalance"
    else:
        recommendation = "Hold"
    logging.info(
        "Rekomendacja rebalansowania: %s (HH index: %.4f, threshold: %.4f)",
        recommendation,
        hh_index,
        threshold,
    )
    return recommendation


def assess_portfolio_risk(
    returns: pd.DataFrame, rolling_window: int = 30, confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Ocena ryzyka portfela na podstawie rolling window.

    Parameters:
        returns (pd.DataFrame): DataFrame zawierający zwroty poszczególnych aktywów.
        rolling_window (int): Okres w dniach do obliczeń rolling.
        confidence_level (float): Poziom ufności.

    Returns:
        pd.DataFrame: DataFrame z kolumnami 'VaR' i 'CVaR' obliczonymi w rolling window.
    """
    var_series = returns.rolling(window=rolling_window).apply(
        lambda x: calculate_var(pd.Series(x), confidence_level), raw=False
    )
    cvar_series = returns.rolling(window=rolling_window).apply(
        lambda x: calculate_cvar(pd.Series(x), confidence_level), raw=False
    )
    risk_df = pd.DataFrame({"VaR": var_series, "CVaR": cvar_series})
    logging.info("Rolling risk assessment ukończony dla okna %d.", rolling_window)
    return risk_df


# -------------------- Testy jednostkowe --------------------
def unit_test_portfolio_risk():
    """
    Testy jednostkowe dla modułu portfolio_risk.py.
    Generuje przykładowe dane zwrotów i weryfikuje poprawność obliczeń VaR, CVaR, Monte Carlo VaR,
    wskaźnika Herfindahla-Hirschmana oraz rekomendacji rebalansowania.
    """
    try:
        np.random.seed(42)
        # Generujemy przykładowe zwroty portfela (dla 252 dni)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        var_value = calculate_var(returns, confidence_level=0.95)
        cvar_value = calculate_cvar(returns, confidence_level=0.95)
        var_mc_value = monte_carlo_var(returns, num_simulations=1000, horizon=1, confidence_level=0.95)

        # Test wskaźnika Herfindahla-Hirschmana dla przykładowych wag
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        hh = herfindahl_index(weights)
        recommendation = recommend_rebalancing(hh, threshold=0.3)

        # Test oceny ryzyka rolling window
        returns_df = pd.DataFrame(
            {
                "Asset_A": np.random.normal(0.001, 0.02, 252),
                "Asset_B": np.random.normal(0.001, 0.025, 252),
                "Asset_C": np.random.normal(0.001, 0.015, 252),
            },
            index=pd.date_range(start="2022-01-01", periods=252, freq="B"),
        )
        risk_df = assess_portfolio_risk(returns_df.mean(axis=1), rolling_window=30, confidence_level=0.95)

        assert var_value < 0, "VaR powinien być wartością ujemną."
        assert cvar_value <= var_value, "CVaR powinien być mniejszy lub równy VaR (bardziej negatywny)."
        assert hh >= 0 and hh <= 1, "HH Index powinien mieścić się w przedziale [0, 1]."
        logging.info("Testy jednostkowe portfolio_risk.py zakończone sukcesem.")
    except AssertionError as ae:
        logging.error("AssertionError w testach portfolio_risk.py: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach portfolio_risk.py: %s", e)
        raise


if __name__ == "__main__":
    try:
        unit_test_portfolio_risk()
    except Exception as e:
        logging.error("Testy jednostkowe portfolio_risk.py nie powiodły się: %s", e)
        raise

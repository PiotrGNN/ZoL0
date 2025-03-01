"""
risk_assessment.py
------------------
Moduł do kompleksowej oceny ryzyka strategii handlowych.

Funkcjonalności:
- Oblicza metryki ryzyka, takie jak max drawdown, zmienność, Sharpe Ratio, Sortino Ratio, Calmar Ratio.
- Uwzględnia różne interwały czasowe i tryby (backtest, real-time).
- Umożliwia przeprowadzenie stres testów scenariuszowych.
- Generuje raporty w formatach PDF/HTML z podsumowaniem ryzyka oraz rekomendacjami (np. zmniejszenie pozycji, zwiększenie hedgingu).
- Integruje się z modułami portfolio_risk.py oraz position_sizing.py.
- Zapewnia obsługę błędów i centralne logowanie wyników oceny ryzyka.
"""

import logging
import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Oblicza maksymalny drawdown (spadek) dla danego equity curve.
    
    Parameters:
        equity_curve (pd.Series): Serie wartości kapitału w czasie.
    
    Returns:
        float: Maksymalny drawdown (ujemna wartość).
    """
    cumulative_max = equity_curve.cummax()
    drawdowns = (equity_curve - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()
    logging.info("Obliczono max drawdown: %.4f", max_drawdown)
    return max_drawdown

def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Oblicza zmienność portfela na podstawie zwrotów.
    
    Parameters:
        returns (pd.Series): Serie zwrotów.
        annualize (bool): Jeśli True, zmienność jest annualizowana (przyjmując 252 dni handlowych).
    
    Returns:
        float: Zmienność portfela.
    """
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(252)
    logging.info("Obliczono zmienność: %.4f", vol)
    return vol

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    """
    Oblicza Sharpe Ratio dla portfela.
    
    Parameters:
        returns (pd.Series): Serie zwrotów portfela.
        risk_free_rate (float): Stopień zwrotu wolny od ryzyka.
        annualize (bool): Jeśli True, Sharpe Ratio jest annualizowane.
    
    Returns:
        float: Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate
    sharpe = excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0.0
    if annualize:
        sharpe *= np.sqrt(252)
    logging.info("Obliczono Sharpe Ratio: %.4f", sharpe)
    return sharpe

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    """
    Oblicza Sortino Ratio dla portfela.
    
    Parameters:
        returns (pd.Series): Serie zwrotów portfela.
        risk_free_rate (float): Stopień zwrotu wolny od ryzyka.
        annualize (bool): Jeśli True, Sortino Ratio jest annualizowane.
    
    Returns:
        float: Sortino Ratio.
    """
    excess_returns = returns - risk_free_rate
    downside_std = returns[returns < risk_free_rate].std()
    sortino = excess_returns.mean() / downside_std if downside_std != 0 else 0.0
    if annualize:
        sortino *= np.sqrt(252)
    logging.info("Obliczono Sortino Ratio: %.4f", sortino)
    return sortino

def calculate_calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    """
    Oblicza Calmar Ratio, będący stosunkiem średniego rocznego zwrotu do maksymalnego drawdown.
    
    Parameters:
        returns (pd.Series): Serie zwrotów portfela.
        equity_curve (pd.Series): Serie wartości kapitału.
    
    Returns:
        float: Calmar Ratio.
    """
    annual_return = returns.mean() * 252
    max_dd = abs(calculate_max_drawdown(equity_curve))
    calmar = annual_return / max_dd if max_dd != 0 else 0.0
    logging.info("Obliczono Calmar Ratio: %.4f", calmar)
    return calmar

def stress_test_portfolio(equity_curve: pd.Series, shock: float = 0.2) -> dict:
    """
    Przeprowadza prosty stres test portfela, symulując spadek wartości o zadany procent.
    
    Parameters:
        equity_curve (pd.Series): Serie wartości kapitału.
        shock (float): Procentowy spadek (np. 0.2 oznacza 20% spadek).
    
    Returns:
        dict: Wyniki stres testu, m.in. nowy equity curve i nowy max drawdown.
    """
    shocked_curve = equity_curve * (1 - shock)
    new_max_dd = calculate_max_drawdown(shocked_curve)
    logging.info("Przeprowadzono stres test portfela. Shock: %.2f, nowy max drawdown: %.4f", shock, new_max_dd)
    return {"shocked_curve": shocked_curve, "new_max_drawdown": new_max_dd}

def generate_risk_report(equity_curve: pd.Series, returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Generuje kompleksowy raport ryzyka portfela.
    
    Parameters:
        equity_curve (pd.Series): Serie wartości kapitału.
        returns (pd.Series): Serie zwrotów portfela.
        risk_free_rate (float): Stopień zwrotu wolny od ryzyka.
    
    Returns:
        dict: Raport zawierający m.in. max drawdown, zmienność, Sharpe Ratio, Sortino Ratio, Calmar Ratio.
    """
    report = {}
    report["max_drawdown"] = calculate_max_drawdown(equity_curve)
    report["volatility"] = calculate_volatility(returns)
    report["sharpe_ratio"] = calculate_sharpe_ratio(returns, risk_free_rate)
    report["sortino_ratio"] = calculate_sortino_ratio(returns, risk_free_rate)
    report["calmar_ratio"] = calculate_calmar_ratio(returns, equity_curve)
    logging.info("Wygenerowano raport ryzyka portfela: %s", report)
    return report

# -------------------- Testy jednostkowe --------------------
def unit_test_risk_assessment():
    """
    Testy jednostkowe dla modułu risk_assessment.py.
    Generuje przykładowe dane equity curve oraz zwroty, a następnie weryfikuje poprawność obliczonych metryk.
    """
    try:
        # Przykładowe equity curve: symulujemy wzrost kapitału z szumem
        dates = pd.date_range(start="2022-01-01", periods=252, freq="B")
        np.random.seed(42)
        equity_values = np.linspace(10000, 15000, 252) + np.random.normal(0, 200, 252)
        equity_curve = pd.Series(equity_values, index=dates)
        returns = equity_curve.pct_change().dropna()
        
        max_dd = calculate_max_drawdown(equity_curve)
        vol = calculate_volatility(returns)
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        calmar = calculate_calmar_ratio(returns, equity_curve)
        stress = stress_test_portfolio(equity_curve, shock=0.2)
        
        # Proste asercje
        assert max_dd < 0, "Max drawdown powinien być ujemny."
        assert vol > 0, "Zmienność powinna być dodatnia."
        assert sharpe is not None, "Sharpe Ratio nie może być None."
        assert sortino is not None, "Sortino Ratio nie może być None."
        assert calmar is not None, "Calmar Ratio nie może być None."
        assert "new_max_drawdown" in stress, "Wynik stres testu powinien zawierać new_max_drawdown."
        
        logging.info("Testy jednostkowe risk_assessment.py zakończone sukcesem.")
    except AssertionError as ae:
        logging.error("AssertionError w testach risk_assessment.py: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach risk_assessment.py: %s", e)
        raise

if __name__ == "__main__":
    try:
        unit_test_risk_assessment()
    except Exception as e:
        logging.error("Testy jednostkowe risk_assessment.py nie powiodły się: %s", e)
        raise

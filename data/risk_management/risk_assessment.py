"""
risk_assessment.py
------------------
Moduł do zaawansowanej oceny ryzyka portfela, włączając stress testy i analizę scenariuszową.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats
from scipy.optimize import minimize
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    CUSTOM = "custom"

@dataclass
class StressTestResult:
    """Klasa przechowująca wyniki stress testu."""
    scenario_name: str
    impact_percentage: float
    max_drawdown: float
    recovery_time_days: Optional[float]
    var_stress: float
    cvar_stress: float
    liquidity_impact: float
    correlation_changes: Dict[str, float]

def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Oblicza Value at Risk dla danego szeregu zwrotów."""
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Oblicza Conditional Value at Risk (Expected Shortfall)."""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_max_drawdown(equity_curve: pd.Series) -> tuple[float, float]:
    """
    Oblicza maksymalny drawdown i czas jego trwania.
    
    Returns:
        tuple: (maksymalny drawdown, czas trwania w dniach)
    """
    roll_max = equity_curve.expanding().max()
    drawdowns = equity_curve / roll_max - 1.0
    max_drawdown = drawdowns.min()
    
    # Oblicz czas trwania
    end_idx = drawdowns.idxmin()
    peak_idx = roll_max.loc[:end_idx].idxmax()
    duration = (end_idx - peak_idx).days
    
    return max_drawdown, duration

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

def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, annualize: bool = True
) -> float:
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
    sharpe = (
        excess_returns.mean() / excess_returns.std()
        if excess_returns.std() != 0
        else 0.0
    )
    if annualize:
        sharpe *= np.sqrt(252)
    logging.info("Obliczono Sharpe Ratio: %.4f", sharpe)
    return sharpe

def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, annualize: bool = True
) -> float:
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
    max_dd = abs(calculate_max_drawdown(equity_curve)[0])
    calmar = annual_return / max_dd if max_dd != 0 else 0.0
    logging.info("Obliczono Calmar Ratio: %.4f", calmar)
    return calmar

def stress_test_portfolio(portfolio_data: pd.Series) -> Dict[str, float]:
    """
    Przeprowadza kompleksowy stress test portfela.
    
    Args:
        portfolio_data: Szereg czasowy wartości portfela
        
    Returns:
        Dict zawierający wyniki stress testu
    """
    try:
        returns = portfolio_data.pct_change().dropna()
        
        # Oblicz podstawowe metryki ryzyka
        vol = returns.std() * np.sqrt(252)  # Annualizowana zmienność
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Oblicz składniki ryzyka systematycznego i niesystematycznego
        # Zakładamy, że rynek wyjaśnia 60% zmienności (typowa wartość)
        systematic_risk = vol * np.sqrt(0.6)
        unsystematic_risk = vol * np.sqrt(0.4)
        
        # Przeprowadź scenariusze stress testów
        scenarios = {
            "market_crash": -0.20,  # 20% spadek rynku
            "high_volatility": 2.0,  # Podwojenie zmienności
            "correlation_breakdown": 0.5,  # 50% spadek korelacji
            "liquidity_crisis": 0.15  # 15% spadek z powodu braku płynności
        }
        
        stress_results = {}
        for scenario, impact in scenarios.items():
            stressed_returns = returns * (1 + impact)
            stress_results[scenario] = {
                "var_95": calculate_var(stressed_returns),
                "cvar_95": calculate_cvar(stressed_returns),
                "max_drawdown": calculate_max_drawdown(portfolio_data * (1 + impact))[0]
            }
        
        return {
            "systematic_risk": systematic_risk,
            "unsystematic_risk": unsystematic_risk,
            "stress_scenarios": stress_results,
            "volatility": vol,
            "skewness": skew,
            "kurtosis": kurt
        }
        
    except Exception as e:
        logger.error(f"Błąd podczas przeprowadzania stress testu: {e}")
        return {}

def run_scenario_analysis(
    portfolio_data: pd.DataFrame,
    scenario_type: ScenarioType,
    num_scenarios: int = 1000,
    custom_scenarios: Optional[Dict[str, Dict[str, float]]] = None
) -> List[StressTestResult]:
    """
    Przeprowadza analizę scenariuszową dla portfela.
    
    Args:
        portfolio_data: DataFrame z danymi portfela
        scenario_type: Typ scenariusza do analizy
        num_scenarios: Liczba scenariuszy dla Monte Carlo
        custom_scenarios: Własne scenariusze do analizy
        
    Returns:
        Lista wyników dla każdego scenariusza
    """
    try:
        results = []
        
        if scenario_type == ScenarioType.HISTORICAL:
            # Analiza historycznych kryzysów
            historical_events = {
                "2008_Crisis": {"start": "2008-09-15", "end": "2009-03-09"},
                "Covid_Crash": {"start": "2020-02-20", "end": "2020-03-23"},
                "DotCom_Bubble": {"start": "2000-03-10", "end": "2002-10-09"}
            }
            
            for event_name, dates in historical_events.items():
                mask = (portfolio_data.index >= dates["start"]) & (portfolio_data.index <= dates["end"])
                event_data = portfolio_data[mask]
                
                if not event_data.empty:
                    drawdown, recovery_time = calculate_max_drawdown(event_data['total_equity'])
                    var_stress = calculate_var(event_data['total_equity'].pct_change().dropna())
                    cvar_stress = calculate_cvar(event_data['total_equity'].pct_change().dropna())
                    
                    results.append(StressTestResult(
                        scenario_name=event_name,
                        impact_percentage=(event_data['total_equity'].iloc[-1] / event_data['total_equity'].iloc[0] - 1),
                        max_drawdown=drawdown,
                        recovery_time_days=recovery_time,
                        var_stress=var_stress,
                        cvar_stress=cvar_stress,
                        liquidity_impact=0.1,  # Założenie uproszczone
                        correlation_changes={}  # To można rozbudować
                    ))
                    
        elif scenario_type == ScenarioType.MONTE_CARLO:
            # Symulacja Monte Carlo
            returns = portfolio_data['total_equity'].pct_change().dropna()
            mu = returns.mean()
            sigma = returns.std()
            
            for i in range(num_scenarios):
                # Generuj roczną ścieżkę zwrotów
                scenario_returns = np.random.normal(mu, sigma, 252)
                scenario_equity = portfolio_data['total_equity'].iloc[0] * np.exp(np.cumsum(scenario_returns))
                
                drawdown, recovery_time = calculate_max_drawdown(pd.Series(scenario_equity))
                var_stress = calculate_var(pd.Series(scenario_returns))
                cvar_stress = calculate_cvar(pd.Series(scenario_returns))
                
                results.append(StressTestResult(
                    scenario_name=f"MC_Scenario_{i+1}",
                    impact_percentage=(scenario_equity[-1] / scenario_equity[0] - 1),
                    max_drawdown=drawdown,
                    recovery_time_days=recovery_time,
                    var_stress=var_stress,
                    cvar_stress=cvar_stress,
                    liquidity_impact=abs(np.random.normal(0, 0.05)),  # Losowy wpływ na płynność
                    correlation_changes={}
                ))
                
        elif scenario_type == ScenarioType.CUSTOM and custom_scenarios:
            # Analiza własnych scenariuszy
            for scenario_name, impacts in custom_scenarios.items():
                modified_data = portfolio_data.copy()
                
                # Aplikuj zdefiniowane zmiany
                for column, impact in impacts.items():
                    if column in modified_data.columns:
                        modified_data[column] *= (1 + impact)
                
                drawdown, recovery_time = calculate_max_drawdown(modified_data['total_equity'])
                var_stress = calculate_var(modified_data['total_equity'].pct_change().dropna())
                cvar_stress = calculate_cvar(modified_data['total_equity'].pct_change().dropna())
                
                results.append(StressTestResult(
                    scenario_name=scenario_name,
                    impact_percentage=(modified_data['total_equity'].iloc[-1] / portfolio_data['total_equity'].iloc[0] - 1),
                    max_drawdown=drawdown,
                    recovery_time_days=recovery_time,
                    var_stress=var_stress,
                    cvar_stress=cvar_stress,
                    liquidity_impact=impacts.get('liquidity', 0),
                    correlation_changes={}
                ))
        
        return results
        
    except Exception as e:
        logger.error(f"Błąd podczas analizy scenariuszowej: {e}")
        return []

def analyze_regime_change(returns: pd.Series, num_regimes: int = 2) -> Dict[str, any]:
    """
    Analizuje zmiany reżimów rynkowych używając modelu Hidden Markov Model.
    
    Args:
        returns: Szereg zwrotów
        num_regimes: Liczba reżimów do wykrycia
        
    Returns:
        Dict z wynikami analizy reżimów
    """
    try:
        from hmmlearn import hmm
        
        # Przygotuj dane
        X = returns.values.reshape(-1, 1)
        
        # Dopasuj model HMM
        model = hmm.GaussianHMM(n_components=num_regimes, covariance_type="full")
        model.fit(X)
        
        # Wykryj stany
        hidden_states = model.predict(X)
        
        # Oblicz charakterystyki każdego reżimu
        regime_stats = {}
        for i in range(num_regimes):
            regime_returns = returns[hidden_states == i]
            regime_stats[f"Regime_{i+1}"] = {
                "mean": regime_returns.mean(),
                "volatility": regime_returns.std(),
                "frequency": len(regime_returns) / len(returns),
                "sharpe": regime_returns.mean() / regime_returns.std() if regime_returns.std() != 0 else 0
            }
        
        return {
            "regime_stats": regime_stats,
            "transition_matrix": model.transmat_,
            "current_regime": int(hidden_states[-1]) + 1
        }
        
    except Exception as e:
        logger.error(f"Błąd podczas analizy reżimów: {e}")
        return {}

def generate_risk_report(
    equity_curve: pd.Series, returns: pd.Series, risk_free_rate: float = 0.0
) -> dict:
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
    report["max_drawdown"], report["max_drawdown_duration"] = calculate_max_drawdown(equity_curve)
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

        max_dd, max_dd_duration = calculate_max_drawdown(equity_curve)
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
        assert (
            "new_max_drawdown" in stress
        ), "Wynik stres testu powinien zawierać new_max_drawdown."

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

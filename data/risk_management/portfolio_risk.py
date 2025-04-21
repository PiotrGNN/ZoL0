"""
portfolio_risk.py
---------------
Moduł do zarządzania ryzykiem portfela.
"""

import logging
import os
import random
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# Konfiguracja logowania
logger = logging.getLogger("portfolio_risk")
if not logger.handlers:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "portfolio_risk.log"))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


class PortfolioRiskManager:
    """
    Klasa do zarządzania ryzykiem portfela.
    """

    def __init__(
        self,
        max_portfolio_risk: float = 0.05,
        max_position_size_pct: float = 0.2,
        max_drawdown_pct: float = 0.1,
        correlation_threshold: float = 0.7,
        var_confidence_level: float = 0.95,
        kelly_fraction: float = 0.5
    ):
        """
        Inicjalizuje zarządcę ryzyka portfela.

        Parameters:
            max_portfolio_risk (float): Maksymalne ryzyko portfela (0.05 = 5%).
            max_position_size_pct (float): Maksymalny rozmiar pozycji jako procent portfela (0.2 = 20%).
            max_drawdown_pct (float): Maksymalny dozwolony drawdown (0.1 = 10%).
            correlation_threshold (float): Próg korelacji dla dywersyfikacji.
            var_confidence_level (float): Poziom ufności dla Value at Risk.
            kelly_fraction (float): Frakcja Kelly'ego dla obliczeń rozmiaru pozycji.
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size_pct = max_position_size_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.correlation_threshold = correlation_threshold
        self.var_confidence_level = var_confidence_level
        self.kelly_fraction = kelly_fraction

        self.portfolio = {}
        self.current_drawdown = 0.0
        self.peak_value = 0.0

        logger.info(f"Zainicjalizowano zarządcę ryzyka portfela. Max ryzyko: {max_portfolio_risk}, "
                    f"Max rozmiar pozycji: {max_position_size_pct}, Max drawdown: {max_drawdown_pct}")

    def calculate_portfolio_risk(self) -> float:
        """
        Oblicza całkowite ryzyko portfela.

        Returns:
            float: Całkowite ryzyko portfela.
        """
        # Implementacja obliczania ryzyka portfela
        # W szablonie zwracamy dummy wartość
        return random.uniform(0, self.max_portfolio_risk)

    def calculate_position_size(self, symbol: str, risk_per_trade: float, stop_loss_pct: float) -> float:
        """
        Oblicza optymalny rozmiar pozycji uwzględniając wszystkie metryki ryzyka.

        Parameters:
            symbol (str): Symbol pary handlowej.
            risk_per_trade (float): Ryzyko na transakcję jako procent kapitału.
            stop_loss_pct (float): Procent stop loss.

        Returns:
            float: Optymalny rozmiar pozycji.
        """
        total_capital = self.get_total_capital()
        max_position = total_capital * self.max_position_size_pct

        if stop_loss_pct == 0:
            return 0

        risk_based_size = (total_capital * risk_per_trade) / stop_loss_pct

        # Uwzględnienie aktualnego drawdown
        if self.current_drawdown > self.max_drawdown_pct * 0.5:
            risk_based_size *= (1 - (self.current_drawdown / self.max_drawdown_pct))

        return min(risk_based_size, max_position)

    def update_portfolio(self, portfolio_data: Dict[str, Any]) -> None:
        """
        Aktualizuje dane portfela.

        Parameters:
            portfolio_data (Dict[str, Any]): Dane portfela.
        """
        self.portfolio = portfolio_data

        # Aktualizacja drawdown
        total_value = self.get_total_capital()
        if total_value > self.peak_value:
            self.peak_value = total_value

        if self.peak_value > 0:
            self.current_drawdown = 1 - (total_value / self.peak_value)
            if self.current_drawdown > self.max_drawdown_pct:
                logger.warning(f"Przekroczono maksymalny drawdown: {self.current_drawdown:.2%} > {self.max_drawdown_pct:.2%}")

    def get_total_capital(self) -> float:
        """
        Zwraca całkowity kapitał portfela.

        Returns:
            float: Całkowity kapitał.
        """
        # W szablonie zwracamy dummy wartość
        if not self.portfolio:
            return 10000.0

        try:
            # Próba wyciągnięcia danych z faktycznego portfela
            total = 0.0
            if 'balances' in self.portfolio:
                for coin, data in self.portfolio['balances'].items():
                    if isinstance(data, dict) and 'equity' in data:
                        # Dla uproszczenia zakładamy, że wszystkie wartości są w USD
                        total += float(data['equity'])
            return total if total > 0 else 10000.0
        except Exception as e:
            logger.error(f"Błąd podczas obliczania całkowitego kapitału: {e}")
            return 10000.0

    def check_risk_limits(self) -> Dict[str, Any]:
        """
        Sprawdza limity ryzyka portfela.

        Returns:
            Dict[str, Any]: Status limitów ryzyka.
        """
        portfolio_risk = self.calculate_portfolio_risk()
        risk_status = {
            "portfolio_risk": portfolio_risk,
            "portfolio_risk_pct": f"{portfolio_risk:.2%}",
            "max_portfolio_risk": self.max_portfolio_risk,
            "max_portfolio_risk_pct": f"{self.max_portfolio_risk:.2%}",
            "current_drawdown": self.current_drawdown,
            "current_drawdown_pct": f"{self.current_drawdown:.2%}",
            "max_drawdown": self.max_drawdown_pct,
            "max_drawdown_pct": f"{self.max_drawdown_pct:.2%}",
            "risk_exceeded": portfolio_risk > self.max_portfolio_risk or self.current_drawdown > self.max_drawdown_pct
        }

        if risk_status["risk_exceeded"]:
            logger.warning(f"Przekroczono limity ryzyka: {risk_status}")

        return risk_status

    def calculate_correlation(self, symbols: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Oblicza korelację między parami handlowymi.

        Parameters:
            symbols (List[str]): Lista symboli par handlowych.

        Returns:
            Dict[Tuple[str, str], float]: Słownik korelacji.
        """
        # Implementacja obliczania korelacji
        # W szablonie zwracamy dummy wartości
        correlations = {}
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlations[(symbol1, symbol2)] = random.uniform(-1, 1)

        return correlations

    def get_diversification_recommendations(self) -> List[Dict[str, Any]]:
        """
        Zwraca rekomendacje dotyczące dywersyfikacji portfela.

        Returns:
            List[Dict[str, Any]]: Lista rekomendacji.
        """
        recommendations = []

        # Analiza koncentracji portfela
        if len(self.portfolio) < 3:
            recommendations.append({
                "message": "Zwiększ liczbę instrumentów w portfelu",
                "severity": "high",
                "action": "diversify"
            })

        # Analiza korelacji
        correlations = self.calculate_correlation(list(self.portfolio.keys()))
        high_corr_pairs = [(s1, s2) for (s1, s2), corr in correlations.items() 
                          if abs(corr) > self.correlation_threshold]

        if high_corr_pairs:
            recommendations.append({
                "message": f"Wysoka korelacja między {len(high_corr_pairs)} parami instrumentów",
                "severity": "medium",
                "pairs": high_corr_pairs,
                "action": "reduce_correlation"
            })

        # Analiza koncentracji ryzyka
        position_sizes = {symbol: data.get('position_size', 0) 
                        for symbol, data in self.portfolio.items()}
        max_position = max(position_sizes.values()) if position_sizes else 0

        if max_position > self.max_position_size_pct:
            recommendations.append({
                "message": f"Zbyt duża pojedyncza pozycja: {max_position:.1%}",
                "severity": "high",
                "action": "reduce_position"
            })

        return recommendations

    def calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Oblicza metryki ryzyka ogona rozkładu.
        """
        var = self._calculate_var(returns)
        cvar = self._calculate_cvar(returns)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'var': var,
            'cvar': cvar,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def _calculate_var(self, returns: pd.Series) -> float:
        """
        Oblicza Value at Risk.
        """
        return abs(np.percentile(returns, (1 - self.var_confidence_level) * 100))

    def _calculate_cvar(self, returns: pd.Series) -> float:
        """
        Oblicza Conditional Value at Risk (Expected Shortfall).
        """
        var = self._calculate_var(returns)
        return abs(returns[returns <= -var].mean())

    def calculate_risk_contribution(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Oblicza kontrybucję ryzyka dla każdego składnika portfela.
        """
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contribution = weights * marginal_risk
        return risk_contribution

    def optimize_risk_parity(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Optymalizuje alokację zgodnie z metodologią risk parity.
        """
        cov_matrix = returns.cov()
        n_assets = len(returns.columns)
        equal_weights = np.array([1/n_assets] * n_assets)
        
        risk_contrib = self.calculate_risk_contribution(equal_weights, cov_matrix)
        target_risk = np.mean(risk_contrib)
        
        # Iteracyjna optymalizacja wag
        weights = equal_weights.copy()
        for _ in range(100):
            risk_contrib = self.calculate_risk_contribution(weights, cov_matrix)
            weights = weights * (target_risk / risk_contrib)
            weights = weights / np.sum(weights)
            
        return {
            'weights': pd.Series(weights, index=returns.columns),
            'risk_contribution': pd.Series(self.calculate_risk_contribution(weights, cov_matrix), 
                                        index=returns.columns)
        }

    def calculate_kelly_position_size(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Oblicza optymalny rozmiar pozycji według kryterium Kelly'ego.
        """
        kelly_size = win_rate - ((1 - win_rate) / win_loss_ratio)
        return max(0, kelly_size * self.kelly_fraction)  # Stosujemy frakcję Kelly dla bezpieczeństwa

    def calculate_portfolio_beta(self, portfolio_returns: pd.Series, market_returns: pd.Series) -> Dict[str, float]:
        """
        Oblicza współczynnik beta portfela względem rynku.
        """
        covariance = portfolio_returns.cov(market_returns)
        market_variance = market_returns.var()
        beta = covariance / market_variance
        
        # R-kwadrat regresji
        slope, intercept = np.polyfit(market_returns, portfolio_returns, 1)
        r_squared = np.corrcoef(market_returns, portfolio_returns)[0,1]**2
        
        return {
            'beta': beta,
            'r_squared': r_squared,
            'alpha': intercept
        }

    def calculate_stress_metrics(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """
        Oblicza metryki stress-testowe portfela.
        """
        returns = portfolio_values.pct_change().dropna()
        worst_drawdown = self.calculate_max_drawdown(portfolio_values)
        worst_month = returns.groupby(pd.Grouper(freq='M')).sum().min()
        volatility = returns.std() * np.sqrt(252)  # Annualizowana zmienność
        
        # Obliczenia dla różnych scenariuszy stress-testowych
        stress_scenarios = {
            'market_crash': returns.quantile(0.01),  # 1% najgorszy scenariusz
            'high_volatility': volatility * 2,  # Podwójna zmienność
            'correlation_breakdown': self.simulate_correlation_breakdown(returns),
            'liquidity_crisis': self.simulate_liquidity_crisis(returns)
        }
        
        return {
            'worst_drawdown': worst_drawdown,
            'worst_month': worst_month,
            'volatility': volatility,
            'stress_scenarios': stress_scenarios
        }

    def simulate_correlation_breakdown(self, returns: pd.Series) -> float:
        """
        Symuluje wpływ załamania korelacji na portfel.
        """
        # Uproszczona symulacja - zakładamy 50% większą zmienność
        stressed_returns = returns * np.random.uniform(0.5, 1.5, len(returns))
        return stressed_returns.std() * np.sqrt(252)

    def simulate_liquidity_crisis(self, returns: pd.Series) -> float:
        """
        Symuluje wpływ kryzysu płynności na portfel.
        """
        # Zakładamy większy spread i poślizg cenowy
        liquidity_impact = returns * 1.5  # 50% większy wpływ na ceny
        return liquidity_impact.std() * np.sqrt(252)

    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """
        Oblicza maksymalny drawdown dla serii wartości portfela.
        """
        cummax = portfolio_values.cummax()
        drawdown = (portfolio_values - cummax) / cummax
        return abs(drawdown.min())


# Singleton instancja dla łatwego dostępu z różnych modułów
portfolio_risk_manager = PortfolioRiskManager()


def calculate_position_size(symbol: str, risk_per_trade: float, stop_loss_pct: float) -> float:
    """
    Funkcja pomocnicza do obliczania rozmiaru pozycji.

    Parameters:
        symbol (str): Symbol pary handlowej.
        risk_per_trade (float): Ryzyko na transakcję jako procent kapitału.
        stop_loss_pct (float): Procent stop loss.

    Returns:
        float: Optymalny rozmiar pozycji.
    """
    return portfolio_risk_manager.calculate_position_size(symbol, risk_per_trade, stop_loss_pct)

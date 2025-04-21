"""
Moduł do analizy dywersyfikacji i ryzyka portfela.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Klasa przechowująca metryki ryzyka."""
    market_risk: float
    credit_risk: float
    liquidity_risk: float
    volatility_risk: float
    concentration_risk: float
    systematic_risk: float
    unsystematic_risk: float
    var_95: float
    cvar_95: float
    stress_test_results: Dict
    scenario_analysis: Dict

class PortfolioAnalyzer:
    def __init__(self):
        """Inicjalizuje analizator portfela."""
        self.historical_data = None
        self.current_portfolio = None
    
    def load_data(self, historical_data: pd.DataFrame, current_portfolio: Dict):
        """Ładuje dane historyczne i obecny stan portfela."""
        self.historical_data = historical_data
        self.current_portfolio = current_portfolio
        self.returns = historical_data.pct_change().dropna()
        
    def calculate_diversification_metrics(self) -> Dict:
        """
        Oblicza metryki dywersyfikacji portfela.
        
        Returns:
            Dict: Metryki dywersyfikacji
        """
        weights = np.array(list(self.current_portfolio['weights'].values()))
        
        # Indeks Herfindahla-Hirschmana (HHI)
        hhi = np.sum(weights ** 2)
        
        # Efektywna liczba aktywów
        effective_n = 1 / hhi
        
        # Korelacja między aktywami
        correlation_matrix = self.returns.corr()
        avg_correlation = (correlation_matrix.sum().sum() - len(correlation_matrix)) / \
                         (len(correlation_matrix) ** 2 - len(correlation_matrix))
        
        # Kontrybucja do ryzyka
        cov_matrix = self.returns.cov()
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_variance)
        risk_concentration = np.sum(risk_contrib ** 2)
        
        # Dywersyfikacja klas aktywów
        asset_classes = self.current_portfolio.get('asset_classes', {})
        class_weights = {}
        for asset, weight in self.current_portfolio['weights'].items():
            asset_class = asset_classes.get(asset, 'other')
            class_weights[asset_class] = class_weights.get(asset_class, 0) + weight
            
        class_hhi = np.sum([w**2 for w in class_weights.values()])
        
        return {
            'herfindahl_index': hhi,
            'effective_n': effective_n,
            'avg_correlation': avg_correlation,
            'risk_concentration': risk_concentration,
            'asset_class_diversity': 1 - class_hhi,
            'class_weights': class_weights
        }
    
    def analyze_risk_exposure(self) -> RiskMetrics:
        """
        Analizuje ekspozycję portfela na różne rodzaje ryzyka.
        
        Returns:
            RiskMetrics: Metryki ryzyka portfela
        """
        weights = np.array(list(self.current_portfolio['weights'].values()))
        returns = self.returns.values
        
        # Ryzyko rynkowe (zmienność)
        market_risk = np.std(np.dot(returns, weights)) * np.sqrt(252)
        
        # VaR i CVaR
        portfolio_returns = np.dot(returns, weights)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        # Ryzyko systematyczne vs niesystematyczne
        market_returns = np.mean(returns, axis=1)  # jako proxy rynku
        beta = np.cov(portfolio_returns, market_returns)[0,1] / np.var(market_returns)
        systematic_risk = (beta ** 2) * np.var(market_returns)
        total_risk = np.var(portfolio_returns)
        unsystematic_risk = total_risk - systematic_risk
        
        # Ryzyko płynności (uproszczone)
        liquidity_risk = self._calculate_liquidity_risk()
        
        # Ryzyko kredytowe (uproszczone)
        credit_risk = self._calculate_credit_risk()
        
        # Ryzyko koncentracji
        concentration_risk = self._calculate_concentration_risk(weights)
        
        # Testy warunków skrajnych
        stress_test_results = self._run_stress_tests()
        
        # Analiza scenariuszy
        scenario_analysis = self._run_scenario_analysis()
        
        return RiskMetrics(
            market_risk=market_risk,
            credit_risk=credit_risk,
            liquidity_risk=liquidity_risk,
            volatility_risk=market_risk,  # jako proxy
            concentration_risk=concentration_risk,
            systematic_risk=systematic_risk,
            unsystematic_risk=unsystematic_risk,
            var_95=var_95,
            cvar_95=cvar_95,
            stress_test_results=stress_test_results,
            scenario_analysis=scenario_analysis
        )
    
    def _calculate_liquidity_risk(self) -> float:
        """Oblicza ryzyko płynności portfela."""
        # Implementacja uproszczona - można rozszerzyć o rzeczywiste dane rynkowe
        liquidity_scores = {
            'BTCUSDT': 0.1,  # Niskie ryzyko płynności
            'ETHUSDT': 0.15,
            'BNBUSDT': 0.2,
            'SOLUSDT': 0.3,
            'ADAUSDT': 0.25
        }
        
        liquidity_risk = 0
        for asset, weight in self.current_portfolio['weights'].items():
            liquidity_risk += weight * liquidity_scores.get(asset, 0.5)
            
        return liquidity_risk
    
    def _calculate_credit_risk(self) -> float:
        """Oblicza ryzyko kredytowe portfela."""
        # Implementacja uproszczona - można rozszerzyć o rzeczywiste ratingi
        credit_scores = {
            'BTCUSDT': 0.15,
            'ETHUSDT': 0.2,
            'BNBUSDT': 0.25,
            'SOLUSDT': 0.35,
            'ADAUSDT': 0.3
        }
        
        credit_risk = 0
        for asset, weight in self.current_portfolio['weights'].items():
            credit_risk += weight * credit_scores.get(asset, 0.4)
            
        return credit_risk
    
    def _calculate_concentration_risk(self, weights: np.ndarray) -> float:
        """Oblicza ryzyko koncentracji portfela."""
        return np.sum(weights[weights > 0.1] ** 2)
    
    def _run_stress_tests(self) -> Dict:
        """Przeprowadza testy warunków skrajnych."""
        results = {}
        
        # Test 1: Spadek rynku o 20%
        market_crash = -0.20
        crash_impact = self._simulate_market_scenario(market_crash)
        results['market_crash_20'] = crash_impact
        
        # Test 2: Wzrost zmienności o 100%
        volatility_shock = 2.0
        vol_impact = self._simulate_volatility_scenario(volatility_shock)
        results['volatility_double'] = vol_impact
        
        # Test 3: Kryzys płynności
        liquidity_crisis = self._simulate_liquidity_crisis()
        results['liquidity_crisis'] = liquidity_crisis
        
        return results
    
    def _run_scenario_analysis(self) -> Dict:
        """Przeprowadza analizę scenariuszy."""
        scenarios = {
            'bull_market': {
                'market_return': 0.30,
                'volatility': 0.8,
                'correlation': 0.9
            },
            'bear_market': {
                'market_return': -0.25,
                'volatility': 1.5,
                'correlation': 0.95
            },
            'recovery': {
                'market_return': 0.15,
                'volatility': 1.2,
                'correlation': 0.7
            },
            'stagflation': {
                'market_return': -0.10,
                'volatility': 1.3,
                'correlation': 0.8
            }
        }
        
        results = {}
        for name, params in scenarios.items():
            impact = self._simulate_scenario(params)
            results[name] = impact
            
        return results
    
    def _simulate_market_scenario(self, market_return: float) -> Dict:
        """Symuluje wpływ scenariusza rynkowego."""
        weights = np.array(list(self.current_portfolio['weights'].values()))
        beta = self._calculate_portfolio_beta()
        
        portfolio_impact = market_return * beta
        var_impact = self._calculate_scenario_var(market_return)
        
        return {
            'portfolio_return': portfolio_impact,
            'var_impact': var_impact,
            'beta': beta
        }
    
    def _simulate_volatility_scenario(self, volatility_multiplier: float) -> Dict:
        """Symuluje wpływ szoku zmienności."""
        weights = np.array(list(self.current_portfolio['weights'].values()))
        current_vol = np.std(np.dot(self.returns.values, weights)) * np.sqrt(252)
        new_vol = current_vol * volatility_multiplier
        
        var_current = self._calculate_current_var()
        var_stressed = var_current * np.sqrt(volatility_multiplier)
        
        return {
            'current_volatility': current_vol,
            'stressed_volatility': new_vol,
            'var_impact': var_stressed - var_current
        }
    
    def _simulate_liquidity_crisis(self) -> Dict:
        """Symuluje wpływ kryzysu płynności."""
        # Uproszczona implementacja - można rozszerzyć o bardziej zaawansowane modelowanie
        liquidity_haircuts = {
            'BTCUSDT': 0.05,
            'ETHUSDT': 0.08,
            'BNBUSDT': 0.12,
            'SOLUSDT': 0.15,
            'ADAUSDT': 0.15
        }
        
        portfolio_value = 0
        liquidation_value = 0
        
        for asset, weight in self.current_portfolio['weights'].items():
            value = weight * self.current_portfolio.get('total_value', 100000)
            haircut = liquidity_haircuts.get(asset, 0.20)
            
            portfolio_value += value
            liquidation_value += value * (1 - haircut)
        
        return {
            'portfolio_value': portfolio_value,
            'liquidation_value': liquidation_value,
            'haircut_impact': (portfolio_value - liquidation_value) / portfolio_value
        }
    
    def _simulate_scenario(self, params: Dict) -> Dict:
        """Symuluje wpływ określonego scenariusza."""
        weights = np.array(list(self.current_portfolio['weights'].values()))
        
        # Dostosuj parametry rozkładu zwrotów
        adjusted_returns = self.returns.copy()
        adjusted_returns = adjusted_returns * params['volatility']
        adjusted_returns = adjusted_returns + params['market_return'] / 252
        
        # Oblicz nowe metryki
        portfolio_return = np.mean(np.dot(adjusted_returns, weights)) * 252
        portfolio_vol = np.std(np.dot(adjusted_returns, weights)) * np.sqrt(252)
        
        # Oblicz nowy VaR
        scenario_var = self._calculate_scenario_var(params['market_return'])
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'var': scenario_var
        }
    
    def _calculate_portfolio_beta(self) -> float:
        """Oblicza betę portfela względem rynku."""
        weights = np.array(list(self.current_portfolio['weights'].values()))
        market_returns = np.mean(self.returns.values, axis=1)
        portfolio_returns = np.dot(self.returns.values, weights)
        
        return np.cov(portfolio_returns, market_returns)[0,1] / np.var(market_returns)
    
    def _calculate_current_var(self, confidence: float = 0.95) -> float:
        """Oblicza obecny VaR portfela."""
        weights = np.array(list(self.current_portfolio['weights'].values()))
        portfolio_returns = np.dot(self.returns.values, weights)
        
        return np.percentile(portfolio_returns, (1 - confidence) * 100)
    
    def _calculate_scenario_var(self, market_return: float, confidence: float = 0.95) -> float:
        """Oblicza VaR w zadanym scenariuszu."""
        weights = np.array(list(self.current_portfolio['weights'].values()))
        portfolio_returns = np.dot(self.returns.values, weights)
        
        # Dostosuj rozkład zwrotów o scenariusz
        adjusted_returns = portfolio_returns + market_return / np.sqrt(252)
        
        return np.percentile(adjusted_returns, (1 - confidence) * 100)
    
    def calculate_performance_attribution(self) -> Dict:
        """
        Przeprowadza analizę atrybucji wyników portfela.
        
        Returns:
            Dict: Wyniki analizy atrybucji
        """
        weights = np.array(list(self.current_portfolio['weights'].values()))
        returns = self.returns.values
        
        # Oblicz zwroty portfela
        portfolio_returns = np.dot(returns, weights)
        
        # Oblicz zwroty rynku (proxy)
        market_returns = np.mean(returns, axis=1)
        
        # Oblicz alfę i betę
        beta = np.cov(portfolio_returns, market_returns)[0,1] / np.var(market_returns)
        alpha = np.mean(portfolio_returns) - beta * np.mean(market_returns)
        
        # Dekomponuj zwroty
        market_contribution = beta * market_returns
        security_selection = portfolio_returns - market_contribution
        
        # Oblicz metryki dla każdego składnika
        attribution = {
            'total_return': np.mean(portfolio_returns) * 252,
            'market_return': np.mean(market_contribution) * 252,
            'selection_return': np.mean(security_selection) * 252,
            'alpha': alpha * 252,
            'beta': beta,
            'r_squared': np.corrcoef(portfolio_returns, market_returns)[0,1]**2
        }
        
        # Dodaj atrybucję na poziomie pojedynczych aktywów
        asset_attribution = {}
        for i, asset in enumerate(self.current_portfolio['weights'].keys()):
            asset_returns = returns[:, i]
            asset_weight = weights[i]
            
            asset_attribution[asset] = {
                'weight': asset_weight,
                'return': np.mean(asset_returns) * 252,
                'contribution': asset_weight * np.mean(asset_returns) * 252,
                'beta': np.cov(asset_returns, market_returns)[0,1] / np.var(market_returns)
            }
        
        attribution['asset_attribution'] = asset_attribution
        
        return attribution
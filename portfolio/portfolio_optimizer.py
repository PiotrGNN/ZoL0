"""
Moduł do optymalizacji i rebalancingu portfela.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self):
        """Inicjalizuje optymalizator portfela."""
        self.historical_data = None
        self.risk_free_rate = 0.02  # 2% jako domyślna stopa wolna od ryzyka
        
    def load_historical_data(self, data: pd.DataFrame):
        """Ładuje historyczne dane cenowe."""
        self.historical_data = data
        self.returns = data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Oblicza podstawowe metryki portfela."""
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return portfolio_return, portfolio_std, sharpe_ratio
        
    def optimize_portfolio(self, risk_tolerance: float, constraints: Dict = None) -> Dict:
        """
        Optymalizuje skład portfela przy danym poziomie tolerancji ryzyka.
        
        Args:
            risk_tolerance (float): Docelowy poziom ryzyka (odchylenie standardowe)
            constraints (Dict): Dodatkowe ograniczenia dla optymalizacji
            
        Returns:
            Dict: Optymalna alokacja aktywów
        """
        n_assets = len(self.returns.columns)
        
        # Domyślne ograniczenia
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # suma wag = 1
        ]
        
        # Dodaj ograniczenie ryzyka
        if risk_tolerance:
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: risk_tolerance - np.sqrt(np.dot(x.T, np.dot(self.cov_matrix * 252, x)))
            })
        
        # Dodaj dodatkowe ograniczenia
        if constraints:
            if 'min_weight' in constraints:
                bounds = tuple((constraints['min_weight'], 1) for _ in range(n_assets))
            if 'max_weight' in constraints:
                bounds = tuple((0, constraints['max_weight']) for _ in range(n_assets))
            if 'asset_classes' in constraints:
                # Implementacja ograniczeń dla klas aktywów
                pass
        
        # Początkowe wagi (równe wagi)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Funkcja celu: maksymalizacja Sharpe Ratio
        def objective(weights):
            return -self._calculate_portfolio_metrics(weights)[2]  # minus bo minimize
        
        # Optymalizacja
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_std, sharpe = self._calculate_portfolio_metrics(optimal_weights)
            
            return {
                'weights': dict(zip(self.returns.columns, optimal_weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe
            }
        else:
            logger.error(f"Optymalizacja nie powiodła się: {result.message}")
            return None
            
    def calculate_rebalancing_trades(self, current_weights: Dict[str, float], 
                                   target_weights: Dict[str, float], 
                                   portfolio_value: float) -> Dict[str, float]:
        """
        Oblicza wymagane transakcje do rebalancingu portfela.
        
        Args:
            current_weights: Obecne wagi aktywów
            target_weights: Docelowe wagi aktywów
            portfolio_value: Całkowita wartość portfela
            
        Returns:
            Dict: Wymagane zmiany dla każdego aktywa (wartości dodatnie = kupno, ujemne = sprzedaż)
        """
        trades = {}
        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            trade_value = (target - current) * portfolio_value
            if abs(trade_value) > 0.01:  # Ignoruj bardzo małe transakcje
                trades[asset] = trade_value
                
        return trades
        
    def calculate_turnover(self, old_weights: Dict[str, float], 
                         new_weights: Dict[str, float]) -> float:
        """Oblicza wskaźnik obrotu portfela."""
        turnover = 0
        for asset in set(old_weights.keys()) | set(new_weights.keys()):
            old = old_weights.get(asset, 0)
            new = new_weights.get(asset, 0)
            turnover += abs(new - old)
            
        return turnover / 2  # Dzielimy przez 2, bo każda zmiana jest liczona podwójnie
        
    def optimize_with_constraints(self, constraints: List[Dict]) -> Dict:
        """
        Optymalizuje portfel z zestawem ograniczeń.
        
        Args:
            constraints: Lista słowników z ograniczeniami
            
        Returns:
            Dict: Optymalna alokacja i metryki
        """
        n_assets = len(self.returns.columns)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        def objective(weights):
            return -self._calculate_portfolio_metrics(weights)[2]
            
        all_constraints = []
        
        # Suma wag = 1
        all_constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        })
        
        # Dodaj ograniczenia użytkownika
        for constraint in constraints:
            if constraint['type'] == 'min_weight':
                all_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=constraint['asset_index']: 
                        x[i] - constraint['value']
                })
            elif constraint['type'] == 'max_weight':
                all_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=constraint['asset_index']: 
                        constraint['value'] - x[i]
                })
            elif constraint['type'] == 'group':
                assets = constraint['assets']
                min_weight = constraint.get('min_weight', 0)
                max_weight = constraint.get('max_weight', 1)
                
                all_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: np.sum([x[i] for i in assets]) - min_weight
                })
                all_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: max_weight - np.sum([x[i] for i in assets])
                })
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=tuple((0, 1) for _ in range(n_assets)),
            constraints=all_constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_std, sharpe = self._calculate_portfolio_metrics(optimal_weights)
            
            return {
                'weights': dict(zip(self.returns.columns, optimal_weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            logger.error(f"Optymalizacja z ograniczeniami nie powiodła się: {result.message}")
            return {'success': False, 'error': result.message}
            
    def get_allocation_metrics(self, weights: Dict[str, float]) -> Dict:
        """Oblicza metryki dla danej alokacji portfela."""
        weight_array = np.array([weights[asset] for asset in self.returns.columns])
        portfolio_return, portfolio_std, sharpe = self._calculate_portfolio_metrics(weight_array)
        
        # Oblicz dodatkowe metryki
        asset_contribution = {}
        for i, asset in enumerate(self.returns.columns):
            # Kontrybucja do ryzyka
            marginal_risk = np.dot(self.cov_matrix[i], weight_array) / portfolio_std
            risk_contribution = weight_array[i] * marginal_risk
            
            # Kontrybucja do zwrotu
            return_contribution = weight_array[i] * self.mean_returns[i]
            
            asset_contribution[asset] = {
                'weight': weight_array[i],
                'risk_contribution': risk_contribution,
                'return_contribution': return_contribution,
                'marginal_risk': marginal_risk
            }
        
        return {
            'portfolio_metrics': {
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe
            },
            'asset_metrics': asset_contribution
        }
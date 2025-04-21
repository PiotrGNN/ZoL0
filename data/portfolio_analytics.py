"""
portfolio_analytics.py
---------------------
Moduł dostarczający zaawansowane funkcje analizy portfela.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
from scipy import stats
from scipy.optimize import minimize

from data.database import DB_PATH
from data.risk_management.risk_assessment import (
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
    stress_test_portfolio
)

logger = logging.getLogger(__name__)

class PortfolioAnalytics:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.conn = sqlite3.connect(DB_PATH)
        
    def __del__(self):
        self.conn.close()

    def get_allocation_history(self, days: int = 30) -> pd.DataFrame:
        """Pobiera historię alokacji aktywów."""
        query = """
        SELECT timestamp, asset_symbol, allocation_percentage, market_value, asset_class
        FROM asset_allocation_history
        WHERE user_id = ? AND timestamp >= datetime('now', ?)
        ORDER BY timestamp
        """
        try:
            df = pd.read_sql_query(
                query, 
                self.conn, 
                params=(self.user_id, f'-{days} days'),
                parse_dates=['timestamp']
            )
            return df.pivot(index='timestamp', columns='asset_symbol', values='allocation_percentage')
        except Exception as e:
            logger.error(f"Błąd podczas pobierania historii alokacji: {e}")
            return pd.DataFrame()

    def calculate_diversification_metrics(self) -> Dict[str, float]:
        """Oblicza metryki dywersyfikacji portfela."""
        try:
            # Pobierz aktualną alokację
            query = """
            SELECT asset_symbol, allocation_percentage
            FROM asset_allocation_history
            WHERE user_id = ? 
            AND timestamp = (
                SELECT MAX(timestamp) 
                FROM asset_allocation_history 
                WHERE user_id = ?
            )
            """
            df = pd.read_sql_query(query, self.conn, params=(self.user_id, self.user_id))
            
            if df.empty:
                return {}
            
            weights = df['allocation_percentage'].values / 100
            
            # Indeks Herfindahla (koncentracja portfela)
            herfindahl = np.sum(weights ** 2)
            
            # Efektywna liczba aktywów
            effective_n = 1 / herfindahl if herfindahl > 0 else 0
            
            # Pobierz dane o klasach aktywów
            query_classes = """
            SELECT asset_class, SUM(allocation_percentage) as total_allocation
            FROM asset_allocation_history
            WHERE user_id = ? 
            AND timestamp = (
                SELECT MAX(timestamp) 
                FROM asset_allocation_history 
                WHERE user_id = ?
            )
            GROUP BY asset_class
            """
            df_classes = pd.read_sql_query(query_classes, self.conn, params=(self.user_id, self.user_id))
            
            # Oblicz różnorodność klas aktywów (entropy)
            class_weights = df_classes['total_allocation'].values / 100
            class_diversity = -np.sum(class_weights * np.log(class_weights)) if len(class_weights) > 0 else 0
            
            metrics = {
                'herfindahl_index': herfindahl,
                'effective_n': effective_n,
                'asset_class_diversity': class_diversity,
            }
            
            # Zapisz metryki do bazy
            self.conn.execute("""
            INSERT INTO diversification_metrics (
                user_id, herfindahl_index, effective_n, asset_class_diversity
            ) VALUES (?, ?, ?, ?)
            """, (self.user_id, herfindahl, effective_n, class_diversity))
            self.conn.commit()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Błąd podczas obliczania metryk dywersyfikacji: {e}")
            return {}

    def analyze_risk_exposure(self) -> Dict[str, float]:
        """Analizuje ekspozycję na różne rodzaje ryzyka."""
        try:
            # Pobierz dane historyczne portfela
            query = """
            SELECT timestamp, total_equity, unrealized_pnl
            FROM portfolio_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 252  -- Rok handlowy
            """
            df = pd.read_sql_query(query, self.conn, params=(self.user_id,), parse_dates=['timestamp'])
            
            if len(df) < 30:  # Minimum 30 dni danych
                return {}
            
            returns = df['total_equity'].pct_change().dropna()
            
            # Oblicz różne metryki ryzyka
            var_95 = calculate_var(returns, 0.95)
            cvar_95 = calculate_cvar(returns, 0.95)
            
            # Przeprowadź stress testy
            stress_results = stress_test_portfolio(df['total_equity'])
            
            # Pobierz aktualną alokację do obliczenia ryzyka koncentracji
            current_allocation = self.get_allocation_history(days=1).iloc[-1]
            concentration_risk = np.std(current_allocation)
            
            risk_metrics = {
                'market_risk': returns.std() * np.sqrt(252),  # Annualized volatility
                'var_95': var_95,
                'cvar_95': cvar_95,
                'concentration_risk': concentration_risk,
                'systematic_risk': stress_results['systematic_risk'],
                'unsystematic_risk': stress_results['unsystematic_risk']
            }
            
            # Zapisz do bazy danych
            self.conn.execute("""
            INSERT INTO risk_exposure (
                user_id, market_risk, var_95, cvar_95, concentration_risk,
                systematic_risk, unsystematic_risk, stress_test_results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.user_id, risk_metrics['market_risk'], var_95, cvar_95,
                concentration_risk, stress_results['systematic_risk'],
                stress_results['unsystematic_risk'], str(stress_results)
            ))
            self.conn.commit()
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Błąd podczas analizy ekspozycji na ryzyko: {e}")
            return {}

    def calculate_turnover_metrics(self, period_days: int = 30) -> Dict[str, float]:
        """Oblicza metryki rotacji kapitału."""
        try:
            # Pobierz transakcje z danego okresu
            query = """
            SELECT executed_at, side, quantity * price as value
            FROM trades
            WHERE user_id = ? 
            AND executed_at >= datetime('now', ?)
            """
            df = pd.read_sql_query(
                query, 
                self.conn, 
                params=(self.user_id, f'-{period_days} days'),
                parse_dates=['executed_at']
            )
            
            if df.empty:
                return {}
            
            # Oblicz sumy transakcji kupna i sprzedaży
            buys = df[df['side'] == 'BUY']['value'].sum()
            sells = df[df['side'] == 'SELL']['value'].sum()
            
            # Pobierz średnią wartość portfela w tym okresie
            query_portfolio = """
            SELECT AVG(total_equity) as avg_equity
            FROM portfolio_history
            WHERE user_id = ? 
            AND timestamp >= datetime('now', ?)
            """
            avg_portfolio_value = pd.read_sql_query(
                query_portfolio,
                self.conn,
                params=(self.user_id, f'-{period_days} days')
            )['avg_equity'].iloc[0]
            
            # Oblicz wskaźnik obrotu
            turnover_ratio = (buys + sells) / (2 * avg_portfolio_value) if avg_portfolio_value > 0 else 0
            
            # Szacuj koszty transakcyjne (załóżmy 0.1% per transakcja)
            trading_costs = (buys + sells) * 0.001
            
            metrics = {
                'total_buys': buys,
                'total_sells': sells,
                'turnover_ratio': turnover_ratio,
                'trading_costs': trading_costs,
                'portfolio_value': avg_portfolio_value
            }
            
            # Zapisz do bazy danych
            self.conn.execute("""
            INSERT INTO capital_turnover (
                user_id, period_start, period_end, total_buys, total_sells,
                turnover_ratio, portfolio_value, trading_costs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.user_id,
                (datetime.now() - timedelta(days=period_days)).date(),
                datetime.now().date(),
                buys, sells, turnover_ratio,
                avg_portfolio_value, trading_costs
            ))
            self.conn.commit()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Błąd podczas obliczania metryk rotacji kapitału: {e}")
            return {}

    def calculate_asset_correlation(self) -> pd.DataFrame:
        """Oblicza macierz korelacji między aktywami w portfelu."""
        try:
            # Pobierz listę aktywów w portfelu
            query = """
            SELECT DISTINCT asset_symbol 
            FROM asset_allocation_history
            WHERE user_id = ? 
            AND timestamp >= datetime('now', '-30 days')
            """
            assets = pd.read_sql_query(query, self.conn, params=(self.user_id,))['asset_symbol'].tolist()
            
            if not assets:
                logger.warning("Brak danych o aktywach w portfelu")
                return pd.DataFrame()
            
            # Pobierz dane historyczne dla tych aktywów
            query_prices = """
            SELECT timestamp, symbol, close
            FROM historical_prices
            WHERE symbol IN ({}) 
            AND timestamp >= datetime('now', '-90 days')
            ORDER BY timestamp
            """.format(','.join(['?'] * len(assets)))
            
            df = pd.read_sql_query(
                query_prices, 
                self.conn, 
                params=assets,
                parse_dates=['timestamp']
            )
            
            if df.empty:
                logger.warning("Brak danych historycznych dla aktywów")
                return pd.DataFrame()
            
            # Przekształć dane do formatu szeregów czasowych
            prices = df.pivot(index='timestamp', columns='symbol', values='close')
            
            # Oblicz zwroty dzienne
            returns = prices.pct_change().dropna()
            
            # Oblicz macierz korelacji
            correlation_matrix = returns.corr()
            
            # Zapisz wyniki do bazy danych
            timestamp = datetime.now().isoformat()
            for i, asset1 in enumerate(correlation_matrix.index):
                for j, asset2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Zapisuj tylko unikalną połowę macierzy (jest symetryczna)
                        self.conn.execute("""
                        INSERT INTO asset_correlations (
                            user_id, asset1, asset2, correlation, timestamp
                        ) VALUES (?, ?, ?, ?, ?)
                        """, (
                            self.user_id, asset1, asset2, 
                            correlation_matrix.loc[asset1, asset2],
                            timestamp
                        ))
            
            self.conn.commit()
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Błąd podczas obliczania korelacji aktywów: {e}")
            return pd.DataFrame()
    
    def optimize_portfolio(self, risk_tolerance: float = 0.02) -> Dict[str, float]:
        """Optymalizuje alokację portfela metodą Markowitza."""
        try:
            # Pobierz historyczne dane cenowe
            query = """
            SELECT h.timestamp, h.symbol, h.close
            FROM historical_prices h
            JOIN (
                SELECT DISTINCT asset_symbol as symbol
                FROM asset_allocation_history
                WHERE user_id = ?
                AND timestamp >= datetime('now', '-90 days')
            ) a ON h.symbol = a.symbol
            WHERE h.timestamp >= datetime('now', '-90 days')
            ORDER BY h.timestamp
            """
            df = pd.read_sql_query(query, self.conn, params=(self.user_id,), parse_dates=['timestamp'])
            
            if df.empty:
                return {}
            
            # Przekształć dane do formatu szeregów czasowych
            prices = df.pivot(index='timestamp', columns='symbol', values='close')
            returns = prices.pct_change().dropna()
            
            # Oblicz średnie zwroty i macierz kowariancji
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Funkcja celu do minimalizacji (minimalizacja ryzyka)
            def objective(weights):
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return portfolio_std
            
            # Ograniczenia
            n_assets = len(mean_returns)
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # suma wag = 1
                {'type': 'ineq', 'fun': lambda x: x}  # wagi >= 0
            )
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Początkowe równe wagi
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optymalizacja
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if not result.success:
                logger.warning("Optymalizacja portfela nie powiodła się")
                return {}
            
            # Przygotuj rekomendacje
            optimal_weights = {
                symbol: weight 
                for symbol, weight in zip(returns.columns, result.x)
            }
            
            # Zapisz rekomendacje do bazy
            self.conn.execute("""
            INSERT INTO allocation_recommendations (
                user_id, risk_profile, recommendations, status
            ) VALUES (?, ?, ?, ?)
            """, (
                self.user_id,
                'optimal',
                str(optimal_weights),
                'pending'
            ))
            self.conn.commit()
            
            return optimal_weights
            
        except Exception as e:
            logger.error(f"Błąd podczas optymalizacji portfela: {e}")
            return {}
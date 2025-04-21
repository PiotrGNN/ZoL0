#!/usr/bin/env python3
"""
risk_manager.py - Zarządzanie ryzykiem dla portfela inwestycyjnego

Ten moduł zawiera klasę RiskManager, która odpowiada za:
- Obliczanie metryk ryzyka dla portfela
- Zarządzanie limitami ryzyka
- Kalkulację wielkości pozycji
"""

import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/risk_manager.log")
    ]
)
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Klasa do zarządzania ryzykiem portfela inwestycyjnego.
    
    Odpowiada za obliczanie różnych metryk ryzyka, zarządzanie limitami ryzyka
    oraz kalkulację optymalnych wielkości pozycji na podstawie parametrów ryzyka.
    """
    
    def __init__(self, db_path='users.db', config=None):
        """
        Inicjalizuje zarządcę ryzyka.
        
        Args:
            db_path: Ścieżka do bazy danych SQLite
            config: Słownik z konfiguracją zarządcy ryzyka
        """
        self.db_path = db_path
        self.config = config or {
            'max_position_size': 0.1,  # Maksymalna wielkość pozycji jako % portfela
            'max_risk_per_trade': 0.02,  # Maksymalne ryzyko na jedną transakcję (2%)
            'max_portfolio_risk': 0.05,  # Maksymalne ryzyko całego portfela (5%)
            'stop_loss_percentage': 0.05,  # Domyślny poziom stop-loss (5%)
            'risk_free_rate': 0.02,  # Stopa zwrotu wolna od ryzyka (2%)
            'max_leverage': 3.0,  # Maksymalna dźwignia
            'min_assets': 3  # Minimalna liczba aktywów w portfelu
        }
        
        logger.info("Inicjalizacja RiskManager z konfiguracją: %s", self.config)
    
    def get_risk_metrics_history(self, user_id, days=30):
        """
        Pobiera historię metryk ryzyka dla danego użytkownika.
        
        Args:
            user_id: ID użytkownika
            days: Liczba dni historii do pobrania
            
        Returns:
            Słownik zawierający metryki ryzyka
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Ustal datę graniczną
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Sprawdź, czy istnieje tabela z metrykami ryzyka
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_metrics'")
            if not c.fetchone():
                logger.warning("Tabela risk_metrics nie istnieje. Zwracam symulowane dane.")
                conn.close()
                return self._generate_simulated_metrics(days)
            
            # Pobierz metryki ryzyka
            c.execute('''
                SELECT timestamp, var_95, var_99, expected_shortfall, sharpe_ratio,
                       sortino_ratio, max_drawdown, beta, correlation_sp500
                FROM risk_metrics
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp
            ''', (user_id, cutoff_date))
            
            rows = c.fetchall()
            conn.close()
            
            if not rows:
                logger.info(f"Brak danych metryk ryzyka dla użytkownika {user_id}. Zwracam symulowane dane.")
                return self._generate_simulated_metrics(days)
            
            columns = ['timestamp', 'var_95', 'var_99', 'expected_shortfall', 'sharpe_ratio',
                      'sortino_ratio', 'max_drawdown', 'beta', 'correlation_sp500']
            
            metrics = {
                'historical': [dict(zip(columns, row)) for row in rows],
                'current': {
                    'var_95': rows[-1][1],
                    'var_99': rows[-1][2],
                    'expected_shortfall': rows[-1][3],
                    'sharpe_ratio': rows[-1][4],
                    'sortino_ratio': rows[-1][5],
                    'max_drawdown': rows[-1][6],
                    'beta': rows[-1][7],
                    'correlation_sp500': rows[-1][8]
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Błąd podczas pobierania metryk ryzyka: {e}")
            return self._generate_simulated_metrics(days)
    
    def _generate_simulated_metrics(self, days):
        """
        Generuje symulowane metryki ryzyka dla demonstracji.
        
        Args:
            days: Liczba dni do wygenerowania
            
        Returns:
            Słownik zawierający symulowane metryki ryzyka
        """
        import random
        from datetime import datetime, timedelta
        
        historical = []
        current_date = datetime.now() - timedelta(days=days)
        
        # Bazowe wartości
        var_95 = 0.02
        var_99 = 0.03
        es = 0.035
        sharpe = 1.2
        sortino = 1.4
        drawdown = 0.05
        beta = 0.8
        corr = 0.6
        
        for _ in range(days):
            # Dodaj losowe wahania do bazowych wartości
            var_95_daily = max(0.01, var_95 + random.uniform(-0.005, 0.005))
            var_99_daily = max(var_95_daily + 0.005, var_99 + random.uniform(-0.007, 0.007))
            es_daily = max(var_99_daily + 0.002, es + random.uniform(-0.008, 0.008))
            sharpe_daily = sharpe + random.uniform(-0.1, 0.1)
            sortino_daily = sortino + random.uniform(-0.15, 0.15)
            drawdown_daily = max(0.01, drawdown + random.uniform(-0.01, 0.01))
            beta_daily = beta + random.uniform(-0.05, 0.05)
            corr_daily = max(-1, min(1, corr + random.uniform(-0.04, 0.04)))
            
            historical.append({
                'timestamp': current_date.isoformat(),
                'var_95': var_95_daily,
                'var_99': var_99_daily,
                'expected_shortfall': es_daily,
                'sharpe_ratio': sharpe_daily,
                'sortino_ratio': sortino_daily,
                'max_drawdown': drawdown_daily,
                'beta': beta_daily,
                'correlation_sp500': corr_daily
            })
            
            current_date += timedelta(days=1)
            
            # Aktualizuj bazowe wartości
            var_95 = var_95_daily
            var_99 = var_99_daily
            es = es_daily
            sharpe = sharpe_daily
            sortino = sortino_daily
            drawdown = drawdown_daily
            beta = beta_daily
            corr = corr_daily
        
        return {
            'historical': historical,
            'current': {
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall': es,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': drawdown,
                'beta': beta,
                'correlation_sp500': corr
            }
        }
    
    def update_risk_limits(self, user_id, limits):
        """
        Aktualizuje limity ryzyka dla danego użytkownika.
        
        Args:
            user_id: ID użytkownika
            limits: Słownik zawierający limity ryzyka do zaktualizowania
            
        Returns:
            bool: True jeśli aktualizacja się powiodła, False w przeciwnym razie
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Sprawdź, czy istnieje tabela z limitami ryzyka
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_limits'")
            if not c.fetchone():
                c.execute('''
                    CREATE TABLE risk_limits (
                        user_id INTEGER PRIMARY KEY,
                        max_position_size REAL,
                        max_risk_per_trade REAL,
                        max_portfolio_risk REAL,
                        stop_loss_percentage REAL,
                        max_leverage REAL,
                        min_assets INTEGER,
                        last_updated TIMESTAMP
                    )
                ''')
                conn.commit()
            
            # Sprawdź, czy istnieje rekord dla danego użytkownika
            c.execute("SELECT * FROM risk_limits WHERE user_id = ?", (user_id,))
            if c.fetchone():
                # Aktualizuj istniejący rekord
                update_query = "UPDATE risk_limits SET "
                update_params = []
                
                for key, value in limits.items():
                    if key in ['max_position_size', 'max_risk_per_trade', 'max_portfolio_risk', 
                              'stop_loss_percentage', 'max_leverage', 'min_assets']:
                        update_query += f"{key} = ?, "
                        update_params.append(value)
                
                update_query += "last_updated = ? WHERE user_id = ?"
                update_params.extend([datetime.now().isoformat(), user_id])
                
                c.execute(update_query, update_params)
            else:
                # Wstaw nowy rekord
                insert_query = '''
                    INSERT INTO risk_limits (
                        user_id, max_position_size, max_risk_per_trade,
                        max_portfolio_risk, stop_loss_percentage, max_leverage,
                        min_assets, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                '''
                
                c.execute(insert_query, (
                    user_id,
                    limits.get('max_position_size', self.config['max_position_size']),
                    limits.get('max_risk_per_trade', self.config['max_risk_per_trade']),
                    limits.get('max_portfolio_risk', self.config['max_portfolio_risk']),
                    limits.get('stop_loss_percentage', self.config['stop_loss_percentage']),
                    limits.get('max_leverage', self.config['max_leverage']),
                    limits.get('min_assets', self.config['min_assets']),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Zaktualizowano limity ryzyka dla użytkownika {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji limitów ryzyka: {e}")
            return False
    
    def calculate_position_size(self, symbol, entry_price, stop_loss, risk_per_trade=None, current_capital=None):
        """
        Oblicza optymalną wielkość pozycji na podstawie parametrów ryzyka.
        
        Args:
            symbol: Symbol instrumentu
            entry_price: Cena wejścia
            stop_loss: Poziom stop-loss
            risk_per_trade: Procent kapitału do zaryzykowania (opcjonalne)
            current_capital: Aktualny kapitał (opcjonalne)
            
        Returns:
            Słownik zawierający szczegóły kalkulacji wielkości pozycji
        """
        try:
            # Jeśli nie podano risk_per_trade, użyj wartości z konfiguracji
            if risk_per_trade is None:
                risk_per_trade = self.config['max_risk_per_trade']
            else:
                risk_per_trade = float(risk_per_trade)
            
            # Jeśli nie podano current_capital, użyj wartości symulowanej
            if current_capital is None:
                current_capital = 10000.0
            else:
                current_capital = float(current_capital)
            
            # Konwersja cen na float
            entry_price = float(entry_price)
            stop_loss = float(stop_loss)
            
            # Oblicz różnicę między ceną wejścia a stop-loss
            if entry_price > stop_loss:  # Pozycja długa
                risk_per_unit = entry_price - stop_loss
                side = "LONG"
            else:  # Pozycja krótka
                risk_per_unit = stop_loss - entry_price
                side = "SHORT"
            
            # Oblicz wielkość pozycji na podstawie ryzyka
            max_position_value = current_capital * risk_per_trade
            
            # Kwota do zaryzykowania
            risk_amount = current_capital * risk_per_trade
            
            # Maksymalna wielkość pozycji przy danym ryzyku
            units = risk_amount / risk_per_unit
            position_value = units * entry_price
            
            # Sprawdź, czy nie przekroczono limitu wielkości pozycji
            max_position_by_size = current_capital * self.config['max_position_size']
            if position_value > max_position_by_size:
                units = max_position_by_size / entry_price
                position_value = units * entry_price
            
            # Oblicz potencjalną stratę i zysk
            potential_loss = units * risk_per_unit
            r_ratio = 2.0  # Domyślny stosunek zysku do ryzyka (R:R)
            potential_profit = potential_loss * r_ratio
            
            # Oblicz poziom take-profit dla R:R = 2
            if side == "LONG":
                take_profit = entry_price + (entry_price - stop_loss) * r_ratio
            else:
                take_profit = entry_price - (stop_loss - entry_price) * r_ratio
            
            return {
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "units": units,
                "position_value": position_value,
                "potential_loss": potential_loss,
                "potential_profit": potential_profit,
                "capital_at_risk_percentage": risk_per_trade * 100,
                "r_ratio": r_ratio
            }
            
        except Exception as e:
            logger.error(f"Błąd podczas obliczania wielkości pozycji: {e}")
            return {
                "error": str(e),
                "symbol": symbol,
                "units": 0,
                "position_value": 0
            }
    
    def calculate_var(self, returns, confidence=0.95, lookback_days=252):
        """
        Oblicza Value at Risk (VaR) dla portfela.
        
        Args:
            returns: Szereg pandas z dziennymi zwrotami
            confidence: Poziom ufności (domyślnie 0.95)
            lookback_days: Liczba dni do analizy (domyślnie 252 - rok handlowy)
            
        Returns:
            float: Wartość VaR
        """
        if len(returns) < lookback_days:
            lookback_days = len(returns)
            
        returns = returns[-lookback_days:]
        return np.percentile(returns, 100 * (1 - confidence)) * -1
    
    def calculate_expected_shortfall(self, returns, var_confidence=0.95):
        """
        Oblicza Expected Shortfall (ES) dla portfela.
        
        Args:
            returns: Szereg pandas z dziennymi zwrotami
            var_confidence: Poziom ufności dla VaR (domyślnie 0.95)
            
        Returns:
            float: Wartość ES
        """
        var = self.calculate_var(returns, var_confidence)
        return returns[returns <= -var].mean() * -1
    
    def calculate_sharpe_ratio(self, returns, periods_per_year=252):
        """
        Oblicza współczynnik Sharpe'a dla portfela.
        
        Args:
            returns: Szereg pandas z dziennymi zwrotami
            periods_per_year: Liczba okresów w roku (domyślnie 252 dla dni handlowych)
            
        Returns:
            float: Wartość współczynnika Sharpe'a
        """
        expected_return = returns.mean() * periods_per_year
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        if volatility == 0:
            return 0
            
        return (expected_return - self.config['risk_free_rate']) / volatility
    
    def calculate_sortino_ratio(self, returns, periods_per_year=252):
        """
        Oblicza współczynnik Sortino dla portfela.
        
        Args:
            returns: Szereg pandas z dziennymi zwrotami
            periods_per_year: Liczba okresów w roku (domyślnie 252 dla dni handlowych)
            
        Returns:
            float: Wartość współczynnika Sortino
        """
        expected_return = returns.mean() * periods_per_year
        
        # Oblicz odchylenie standardowe tylko dla ujemnych zwrotów
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # Nie ma ujemnych zwrotów
            
        downside_deviation = negative_returns.std() * np.sqrt(periods_per_year)
        
        if downside_deviation == 0:
            return float('inf')  # Unikaj dzielenia przez zero
            
        return (expected_return - self.config['risk_free_rate']) / downside_deviation
    
    def calculate_max_drawdown(self, equity_curve):
        """
        Oblicza maksymalny drawdown portfela.
        
        Args:
            equity_curve: Szereg pandas z wartością portfela
            
        Returns:
            float: Wartość maksymalnego drawdownu (jako procent)
        """
        # Oblicz narastające maksimum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Oblicz drawdown z narastającego maksimum
        drawdown = (equity_curve - running_max) / running_max
        
        # Znajdź maksymalny drawdown
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown) if not np.isnan(max_drawdown) else 0
    
    def calculate_beta(self, returns, market_returns):
        """
        Oblicza współczynnik beta portfela.
        
        Args:
            returns: Szereg pandas z dziennymi zwrotami portfela
            market_returns: Szereg pandas z dziennymi zwrotami rynku
            
        Returns:
            float: Wartość współczynnika beta
        """
        # Upewnij się, że szeregi mają tę samą długość
        if len(returns) != len(market_returns):
            min_len = min(len(returns), len(market_returns))
            returns = returns[-min_len:]
            market_returns = market_returns[-min_len:]
        
        # Oblicz kowariancję między zwrotami portfela a rynku
        cov = np.cov(returns, market_returns)[0, 1]
        
        # Oblicz wariancję zwrotów rynku
        var = np.var(market_returns)
        
        if var == 0:
            return 0
            
        return cov / var
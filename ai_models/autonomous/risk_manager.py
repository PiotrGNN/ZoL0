"""
Autonomiczny system zarządzania ryzykiem - moduł dostosowujący parametry ryzyka 
na podstawie warunków rynkowych i wyników.

Ten moduł odpowiada za:
1. Dynamiczną adaptację wielkości pozycji
2. Dostosowywanie poziomów stop-loss i take-profit
3. Zarządzanie ryzykiem portfela
4. Predykcję zmienności i dostosowywanie parametrów
"""

import numpy as np
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

class AutonomousRiskManager:
    """
    System autonomicznego zarządzania ryzykiem, który dynamicznie 
    dostosowuje parametry na podstawie warunków rynkowych i wyników.
    """
    
    def __init__(self, initial_config: Dict[str, Any] = None):
        """
        Inicjalizacja autonomicznego zarządzania ryzykiem.
        
        Args:
            initial_config: Początkowa konfiguracja ryzyka
        """
        # Ustaw domyślną konfigurację, jeśli nie podano
        self.config = initial_config or {
            "base_position_size": 0.05,  # Podstawowa wielkość pozycji (% kapitału)
            "max_position_size": 0.2,    # Maksymalna wielkość pozycji (% kapitału)
            "min_position_size": 0.01,   # Minimalna wielkość pozycji (% kapitału)
            "base_stop_loss_pct": 0.02,  # Podstawowy poziom stop-loss (%)
            "max_stop_loss_pct": 0.05,   # Maksymalny poziom stop-loss (%)
            "min_stop_loss_pct": 0.005,  # Minimalny poziom stop-loss (%)
            "base_take_profit_pct": 0.04, # Podstawowy poziom take-profit (%)
            "position_size_volatility_factor": 0.5,  # Wpływ zmienności na wielkość pozycji
            "max_capital_at_risk": 0.1,   # Maksymalny kapitał narażony na ryzyko (%)
            "adaptation_speed": 0.2      # Szybkość adaptacji parametrów
        }
        
        # Inicjalizuj zmienne stanu
        self.market_volatility = None     # Obecna zmienność rynku
        self.performance_history = []     # Historia wyników
        self.drawdown_history = []        # Historia drawdownów
        self.position_sizes_history = []  # Historia wielkości pozycji
        self.trade_results = []           # Historia wyników transakcji
        
        # Stan aktualnego ryzyka
        self.current_risk = {
            "position_size": self.config["base_position_size"],
            "stop_loss_pct": self.config["base_stop_loss_pct"],
            "take_profit_pct": self.config["base_take_profit_pct"],
            "max_open_positions": 3,
            "capital_at_risk": 0.0
        }
        
        # Ścieżka do zapisywania danych
        self.storage_path = os.path.join(os.path.dirname(__file__), "risk_data")
        os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info("Zainicjalizowano autonomiczny system zarządzania ryzykiem")
        
    def update_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Aktualizuje warunki rynkowe używane do kalkulacji ryzyka.
        
        Args:
            market_data: Dane rynkowe (ceny, wolumeny, wskaźniki)
            
        Returns:
            Słownik z aktualizowanymi parametrami ryzyka
        """
        # Oblicz zmienność rynku
        if 'close' in market_data and isinstance(market_data['close'], (list, np.ndarray)) and len(market_data['close']) > 20:
            # Oblicz zmienność na podstawie ostatnich 20 świec
            closes = np.array(market_data['close'][-20:])
            returns = np.diff(closes) / closes[:-1]
            self.market_volatility = np.std(returns) * np.sqrt(252)  # Annualizowana zmienność
            
            # Dostosuj parametry ryzyka na podstawie zmienności
            self._adapt_risk_parameters()
            
            logger.info(f"Zaktualizowano warunki rynkowe. Zmienność: {self.market_volatility:.4f}")
            
        return self.get_risk_parameters()
    
    def _adapt_risk_parameters(self) -> None:
        """Dostosowuje parametry ryzyka na podstawie warunków rynkowych i wyników."""
        if self.market_volatility is None:
            return
            
        # Dostosuj wielkość pozycji odwrotnie proporcjonalnie do zmienności
        # Wyższa zmienność = mniejsza pozycja
        volatility_factor = np.clip(1.0 / (self.market_volatility * self.config["position_size_volatility_factor"] * 10), 0.5, 2.0)
        new_position_size = self.config["base_position_size"] * volatility_factor
        
        # Dostosuj na podstawie historii drawdownów
        if self.drawdown_history and np.mean(self.drawdown_history[-5:]) > 0.1:
            # Jeśli ostatnie drawdowny są duże, zmniejsz wielkość pozycji
            new_position_size *= 0.8
            
        # Dostosuj na podstawie ostatnich wyników
        if self.trade_results:
            # Oblicz średni wynik ostatnich 10 transakcji
            recent_results = [r.get('profit_pct', 0) for r in self.trade_results[-10:]]
            if recent_results:
                avg_profit = np.mean(recent_results)
                if avg_profit > 0:
                    # Jeśli wyniki są dobre, możemy delikatnie zwiększyć pozycję
                    profit_factor = min(1.2, 1 + avg_profit / 2)
                    new_position_size *= profit_factor
                else:
                    # Jeśli wyniki są złe, zmniejszamy pozycję
                    loss_factor = max(0.7, 1 + avg_profit)
                    new_position_size *= loss_factor
        
        # Zastosuj granice
        new_position_size = np.clip(
            new_position_size, 
            self.config["min_position_size"], 
            self.config["max_position_size"]
        )
        
        # Dostosuj stoplossy - przy wyższej zmienności, szerszy stop loss
        new_stop_loss = self.config["base_stop_loss_pct"] * (1 + self.market_volatility)
        new_stop_loss = np.clip(
            new_stop_loss,
            self.config["min_stop_loss_pct"],
            self.config["max_stop_loss_pct"]
        )
        
        # Dostosuj take profit - proporcjonalnie do stop loss
        new_take_profit = new_stop_loss * 2.0  # Przykładowy stosunek risk/reward 1:2
        
        # Adaptacja parametrów z uwzględnieniem tempa adaptacji
        adaptation_rate = self.config["adaptation_speed"]
        self.current_risk["position_size"] = (1 - adaptation_rate) * self.current_risk["position_size"] + adaptation_rate * new_position_size
        self.current_risk["stop_loss_pct"] = (1 - adaptation_rate) * self.current_risk["stop_loss_pct"] + adaptation_rate * new_stop_loss
        self.current_risk["take_profit_pct"] = (1 - adaptation_rate) * self.current_risk["take_profit_pct"] + adaptation_rate * new_take_profit
        
        logger.info(f"Zaadaptowano parametry ryzyka: pozycja={self.current_risk['position_size']:.4f}, "
                    f"stop_loss={self.current_risk['stop_loss_pct']:.4f}, "
                    f"take_profit={self.current_risk['take_profit_pct']:.4f}")
        
        # Zapisz historię wielkości pozycji
        self.position_sizes_history.append((datetime.now().isoformat(), self.current_risk["position_size"]))
        
        # Okresowe zapisywanie historii
        if len(self.position_sizes_history) % 100 == 0:
            self.save_state()
            
    def calculate_position_size(self, portfolio_value: float, signal_strength: float = 1.0) -> Dict[str, Any]:
        """
        Oblicz adaptacyjną wielkość pozycji na podstawie sygnału i warunków rynkowych.
        
        Args:
            portfolio_value: Wartość portfela
            signal_strength: Siła sygnału (0.0-1.0, wyższa = silniejszy sygnał)
            
        Returns:
            Słownik z parametrami pozycji
        """
        # Podstawowa wielkość pozycji jako procent kapitału
        base_size = self.current_risk["position_size"] * portfolio_value
        
        # Dostosuj na podstawie siły sygnału
        signal_adjustment = np.clip(signal_strength, 0.5, 1.5)
        adjusted_size = base_size * signal_adjustment
        
        # Sprawdź maksymalny kapitał narażony na ryzyko
        max_capital_at_risk = portfolio_value * self.config["max_capital_at_risk"]
        capital_at_risk = self.current_risk["capital_at_risk"] + (adjusted_size * self.current_risk["stop_loss_pct"])
        
        if capital_at_risk > max_capital_at_risk:
            # Dostosuj wielkość pozycji, aby nie przekraczać maksymalnego ryzyka
            risk_ratio = max_capital_at_risk / capital_at_risk
            adjusted_size *= risk_ratio
        
        # Oblicz poziomy stop loss i take profit
        stop_loss_amount = adjusted_size * self.current_risk["stop_loss_pct"]
        take_profit_amount = adjusted_size * self.current_risk["take_profit_pct"]
        
        # Zwróć pełną konfigurację pozycji
        position_config = {
            "size": adjusted_size,
            "size_pct": adjusted_size / portfolio_value,
            "stop_loss_pct": self.current_risk["stop_loss_pct"],
            "take_profit_pct": self.current_risk["take_profit_pct"],
            "max_loss": stop_loss_amount,
            "expected_profit": take_profit_amount,
            "risk_reward_ratio": self.current_risk["take_profit_pct"] / self.current_risk["stop_loss_pct"]
        }
        
        logger.info(f"Obliczono wielkość pozycji: {adjusted_size:.2f} ({position_config['size_pct']*100:.2f}% kapitału), "
                   f"R:R = 1:{position_config['risk_reward_ratio']:.2f}")
        
        return position_config
        
    def update_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """
        Aktualizuje historię wyników i drawdownów po zakończeniu transakcji.
        
        Args:
            trade_result: Wynik transakcji (zawiera profit_pct, max_drawdown)
        """
        self.trade_results.append(trade_result)
        
        if 'profit_pct' in trade_result:
            self.performance_history.append(trade_result['profit_pct'])
            
        if 'max_drawdown' in trade_result:
            self.drawdown_history.append(trade_result['max_drawdown'])
        
        # Po każdej transakcji aktualizuj dostępny kapitał narażony na ryzyko
        # Zakładamy, że po zakończeniu transakcji jej ryzyko znika z kalkulacji
        if 'position_risk' in trade_result:
            self.current_risk["capital_at_risk"] = max(0, self.current_risk["capital_at_risk"] - trade_result['position_risk'])
            
        # Re-adaptacja parametrów po transakcji
        self._adapt_risk_parameters()
        
        logger.info(f"Zaktualizowano wyniki handlu: profit={trade_result.get('profit_pct', 'N/A')}%, "
                   f"drawdown={trade_result.get('max_drawdown', 'N/A')}%")
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Zwraca aktualne parametry ryzyka."""
        return {
            "position_size_pct": self.current_risk["position_size"],
            "stop_loss_pct": self.current_risk["stop_loss_pct"],
            "take_profit_pct": self.current_risk["take_profit_pct"],
            "market_volatility": self.market_volatility,
            "capital_at_risk_pct": self.current_risk["capital_at_risk"],
            "max_capital_at_risk_pct": self.config["max_capital_at_risk"],
            "risk_reward_ratio": self.current_risk["take_profit_pct"] / self.current_risk["stop_loss_pct"] if self.current_risk["stop_loss_pct"] > 0 else 0
        }
    
    def visualize_risk_parameters(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generuje wizualizację parametrów ryzyka w czasie.
        
        Args:
            save_path: Opcjonalna ścieżka do zapisania wykresu
            
        Returns:
            Obiekt Figure z wykresem
        """
        if not self.position_sizes_history:
            return None
            
        # Przygotuj dane
        dates = []
        position_sizes = []
        
        for date_str, size in self.position_sizes_history:
            try:
                date = pd.to_datetime(date_str)
                dates.append(date)
                position_sizes.append(size)
            except:
                pass
                
        if not dates:
            return None
            
        # Utwórz wykres
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dates, position_sizes, 'b-', label='Wielkość pozycji (%)')
        
        # Dodaj etykiety i tytuł
        ax.set_xlabel('Data')
        ax.set_ylabel('Wielkość pozycji (% kapitału)')
        ax.set_title('Adaptacja wielkości pozycji w czasie')
        ax.legend()
        
        plt.tight_layout()
        
        # Zapisz wykres jeśli podano ścieżkę
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def visualize_trade_performance(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generuje wizualizację wyników transakcji.
        
        Args:
            save_path: Opcjonalna ścieżka do zapisania wykresu
            
        Returns:
            Obiekt Figure z wykresem
        """
        if not self.trade_results:
            return None
            
        # Przygotuj dane
        dates = []
        profits = []
        cumulative_profit = []
        
        running_sum = 0
        for result in self.trade_results:
            if 'timestamp' in result and 'profit_pct' in result:
                try:
                    date = pd.to_datetime(result['timestamp'])
                    profit = result['profit_pct']
                    
                    dates.append(date)
                    profits.append(profit)
                    
                    running_sum += profit
                    cumulative_profit.append(running_sum)
                except:
                    pass
                    
        if not dates:
            return None
            
        # Utwórz wykres
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Wykres pojedynczych zysków/strat
        bars = ax1.bar(dates, profits, color=['g' if p > 0 else 'r' for p in profits])
        ax1.set_ylabel('Zysk/strata (%)')
        ax1.set_title('Wyniki poszczególnych transakcji')
        
        # Wykres skumulowanego zysku
        ax2.plot(dates, cumulative_profit, 'b-')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Skumulowany zysk/strata (%)')
        ax2.set_title('Skumulowany wynik')
        
        plt.tight_layout()
        
        # Zapisz wykres jeśli podano ścieżkę
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def save_state(self, filepath: str = None) -> None:
        """
        Zapisuje stan menedżera ryzyka.
        
        Args:
            filepath: Ścieżka do pliku (domyślnie: risk_manager_state_[timestamp].json)
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                self.storage_path,
                f"risk_manager_state_{timestamp}.json"
            )
        
        state = {
            'config': self.config,
            'current_risk': self.current_risk,
            'market_volatility': self.market_volatility,
            'performance_history': self.performance_history[-100:] if self.performance_history else [],  # Ostatnie 100 wyników
            'drawdown_history': self.drawdown_history[-100:] if self.drawdown_history else [],  # Ostatnie 100 drawdownów
            'position_sizes_history': self.position_sizes_history[-100:] if self.position_sizes_history else [],  # Ostatnie 100 wielkości pozycji
            'trade_results': self.trade_results[-50:] if self.trade_results else []  # Ostatnie 50 wyników transakcji
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, default=str)
            
        logger.info(f"Zapisano stan menedżera ryzyka do pliku: {filepath}")
    
    def load_state(self, filepath: str) -> bool:
        """
        Wczytuje stan menedżera ryzyka z pliku.
        
        Args:
            filepath: Ścieżka do pliku stanu
            
        Returns:
            True jeśli wczytano pomyślnie, False w przeciwnym przypadku
        """
        if not os.path.exists(filepath):
            logger.warning(f"Nie można wczytać stanu menedżera ryzyka: plik {filepath} nie istnieje")
            return False
            
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.config = state.get('config', self.config)
            self.current_risk = state.get('current_risk', self.current_risk)
            self.market_volatility = state.get('market_volatility')
            
            # Wczytaj historię
            self.performance_history = state.get('performance_history', [])
            self.drawdown_history = state.get('drawdown_history', [])
            self.position_sizes_history = state.get('position_sizes_history', [])
            self.trade_results = state.get('trade_results', [])
            
            logger.info(f"Wczytano stan menedżera ryzyka z pliku: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania stanu menedżera ryzyka: {str(e)}")
            return False
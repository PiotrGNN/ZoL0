"""
market_dummy_env.py
-------------------
Moduł oferujący uproszczone środowisko rynkowe do szybkiego prototypowania i testowania algorytmów.
Symulowane są losowe wahania cen z uwzględnieniem spreadu, prowizji oraz potencjalnych poślizgów cenowych.
Środowisko umożliwia regulację zmienności oraz płynności rynku, co pozwala testować strategie na różnych poziomach kapitału.
Implementuje metody: reset(), step(action) i reward() z prostą obsługą logowania oraz wyjątków.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import numpy as np

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class MarketDummyEnv:
    """Market environment for testing trading strategies"""
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 volatility: float = 0.01,
                 commission: float = 0.001,
                 spread: float = 0.002,
                 base_price: float = 100.0):
        """Initialize market environment
        
        Args:
            initial_capital: Starting capital
            volatility: Market price volatility
            commission: Trading commission as decimal
            spread: Bid-ask spread as decimal
            base_price: Initial asset price
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.volatility = volatility
        self.commission = commission
        self.spread = spread
        self.base_price = base_price
        self.current_price = base_price
        
        self.position = None
        self.entry_price = 0
        self.step_count = 0
        self.logger = logging.getLogger(__name__)
        
        self.reset()

    def reset(self):
        """Reset the environment"""
        self.current_capital = self.initial_capital
        self.current_position = None
        self.entry_price = None
        self.current_step = 0
        self.current_price = 100.0
        self.logger.info(f"Reset środowiska: kapitał = {self.current_capital:f}, krok = {self.current_step}")
        return self._get_state()

    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step in the environment"""
        # Generate random price movement
        price_change = np.random.normal(0, self.volatility)
        self.current_price = self.current_price * (1 + price_change)

        # Calculate effective price with spread
        if action == "buy":
            effective_price = self.current_price * (1 + self.spread/2)  # Add half spread for buys
            self.logger.info(f"Wykonano akcję BUY przy cenie: {effective_price:.6f}")
        elif action == "sell":
            effective_price = self.current_price * (1 - self.spread/2)  # Subtract half spread for sells
            self.logger.info(f"Wykonano akcję SELL przy cenie: {effective_price:.6f}")
        else:  # hold
            effective_price = self.current_price
            self.logger.info("Wykonano akcję HOLD - brak zmiany pozycji.")

        self.current_step += 1
        old_capital = self.current_capital
        
        try:
            if action == "buy":
                if self.current_position is None:
                    self.entry_price = effective_price
                    self.current_position = "long"
                    # Apply commission
                    commission_cost = effective_price * self.commission
                    self.current_capital -= commission_cost
            
            elif action == "sell":
                if self.current_position == "long":
                    # Calculate P&L
                    pnl = effective_price - self.entry_price
                    # Apply commission
                    commission_cost = effective_price * self.commission
                    self.current_capital += pnl - commission_cost
                    self.current_position = None
                    self.entry_price = None
                
            elif action == "hold":
                pass
            
            else:
                self.logger.error(f"Błąd w metodzie step: Nieznana akcja: {action}")
                raise ValueError(f"Nieznana akcja: {action}")
            
            # Calculate reward as relative capital change
            reward = (self.current_capital - old_capital) / old_capital
            
            self.logger.info(f"Krok {self.current_step}: akcja = {action}, nagroda = {reward:f}, kapitał = {self.current_capital:f}")
            
            return self._get_state(), reward, False, {}
            
        except Exception as e:
            self.logger.error(f"Błąd w metodzie step: {str(e)}")
            raise

    def _get_state(self):
        """Return current state"""
        return {
            'capital': self.current_capital,
            'position': self.current_position,
            'price': self.current_price,
            'step': self.current_step
        }


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        env = MarketDummyEnv(
            initial_capital=5000,
            commission=0.002,
            spread=0.02,
        )
        state = env.reset()
        done = False

        while not done:
            # Prosty przykład strategii: losowy wybór akcji
            action = np.random.choice(["buy", "sell", "hold"])
            state, reward, done, info = env.step(action)
            print(
                f"Krok: {info['step']}, Akcja: {info['action']}, Reward: {reward:.2f}, Kapitał: {info['capital']:.2f}"
            )
    except Exception as e:
        logging.error("Błąd w symulacji środowiska: %s", e)
        raise

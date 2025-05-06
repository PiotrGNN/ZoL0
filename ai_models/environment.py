"""
environment.py
---------------
Moduł definiujący środowisko symulacyjne dla uczenia maszynowego (RL lub innych algorytmów).
"""

import logging
import gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from gym import spaces

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class MarketEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        initial_capital=10000,
        leverage=1.0,
        risk_limit=0.05,
        data=None,
        mode="simulated",
        commission=0.001  # 0.1% commission
    ):
        super(MarketEnvironment, self).__init__()
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        self.risk_limit = risk_limit
        self.mode = mode
        self.data = data
        self.commission = commission
        self.current_step = 0
        self.position = 0  # 1 = long, -1 = short, 0 = no position
        self.entry_price = None
        self.trades_history = []

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [price, position, capital, unrealized_pnl]
        self.observation_space = spaces.Box(
            low=np.array([0, -1, 0, -np.inf]),
            high=np.array([np.inf, 1, np.inf, np.inf]),
            dtype=np.float32
        )

        if self.data is None:
            self.simulated_prices = self._generate_simulated_data()
        else:
            self.simulated_prices = None

        logging.info(
            "MarketEnvironment initialized in mode '%s' with capital: %f",
            self.mode,
            self.initial_capital,
        )

    def _generate_simulated_data(self, steps=1000):
        """Generates simulated market data with realistic patterns"""
        prices = [100.0]
        trend = 0
        volatility = 0.02
        
        for _ in range(1, steps):
            # Update trend with mean reversion
            trend = 0.99 * trend + 0.01 * np.random.randn()
            
            # Generate price movement
            price_change = (
                trend +  # Trend component
                volatility * np.random.randn() +  # Random walk
                0.001 * np.sin(len(prices) / 100)  # Cyclical component
            )
            
            new_price = max(0.01, prices[-1] * (1 + price_change))
            prices.append(new_price)
            
        return np.array(prices)

    def reset(self) -> np.ndarray:
        """Resets the environment to initial state"""
        self.current_capital = self.initial_capital
        self.current_step = 0
        self.position = 0
        self.entry_price = None
        self.trades_history = []
        
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Returns the current state observation"""
        price = self._get_current_price()
        unrealized_pnl = self._calculate_unrealized_reward(price)
        
        return np.array([
            price,
            self.position,
            self.current_capital,
            unrealized_pnl
        ], dtype=np.float32)

    def _get_current_price(self) -> float:
        """Returns current price from data or simulation"""
        if self.data is not None:
            return self.data.iloc[self.current_step]["price"]
        return self.simulated_prices[self.current_step]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment
        
        Args:
            action (int): 0 = hold, 1 = buy, 2 = sell
            
        Returns:
            observation (np.ndarray): Environment observation
            reward (float): Reward for this step
            done (bool): Whether the episode has ended
            info (dict): Additional information
        """
        try:
            price = self._get_current_price()
            reward = 0.0
            
            # Convert action to trade decision
            if action == 1:  # Buy
                reward = self._execute_trade(price, "buy")
            elif action == 2:  # Sell
                reward = self._execute_trade(price, "sell")
            else:  # Hold
                reward = self._calculate_unrealized_reward(price)

            # Move to next step
            self.current_step += 1
            done = self._is_episode_done()
            
            # Get new observation
            observation = self._get_observation()
            
            info = {
                "step": self.current_step,
                "action": ["hold", "buy", "sell"][action],
                "price": price,
                "position": self.position,
                "capital": self.current_capital,
                "reward": reward
            }
            
            return observation, reward, done, info

        except Exception as e:
            logging.error(f"Error in step: {str(e)}")
            raise

    def _execute_trade(self, price: float, action: str) -> float:
        """Executes a trade and returns the reward"""
        reward = 0.0
        
        if action == "buy":
            if self.position <= 0:
                # Close short position if exists
                if self.position == -1:
                    reward = self._close_position(price)
                # Open long position
                self.position = 1
                self.entry_price = price
                reward -= price * self.commission  # Apply commission
        
        elif action == "sell":
            if self.position >= 0:
                # Close long position if exists
                if self.position == 1:
                    reward = self._close_position(price)
                # Open short position
                self.position = -1
                self.entry_price = price
                reward -= price * self.commission  # Apply commission
        
        return reward

    def _close_position(self, price: float) -> float:
        """Closes current position and returns the reward"""
        if self.entry_price is None:
            return 0.0
            
        # Calculate profit/loss
        if self.position == 1:
            pnl = (price - self.entry_price) * self.leverage
        else:  # short position
            pnl = (self.entry_price - price) * self.leverage
            
        # Apply commission
        pnl -= price * self.commission
        
        # Update capital
        self.current_capital += pnl
        
        # Record trade
        self.trades_history.append({
            "entry_price": self.entry_price,
            "exit_price": price,
            "position": self.position,
            "pnl": pnl
        })
        
        return pnl

    def _calculate_unrealized_reward(self, current_price: float) -> float:
        """Calculates unrealized PnL for current position"""
        if self.position == 0 or self.entry_price is None:
            return 0.0
            
        if self.position == 1:
            return (current_price - self.entry_price) * self.leverage
        else:  # short position
            return (self.entry_price - current_price) * self.leverage

    def _is_episode_done(self) -> bool:
        """Checks if episode is finished"""
        # End if we've run out of data
        if self.data is not None:
            if self.current_step >= len(self.data) - 1:
                return True
        elif self.current_step >= len(self.simulated_prices) - 1:
            return True
            
        # End if we've lost too much capital
        if self.current_capital <= self.initial_capital * 0.5:  # 50% max drawdown
            return True
            
        return False

    def render(self, mode='human'):
        """Renders the environment"""
        if mode == 'human':
            print(f"\nStep: {self.current_step}")
            print(f"Price: {self._get_current_price():.2f}")
            print(f"Position: {self.position}")
            print(f"Capital: {self.current_capital:.2f}")
            if self.position != 0:
                print(f"Unrealized PnL: {self._calculate_unrealized_reward(self._get_current_price()):.2f}")

# -------------------- Przykładowe użycie --------------------

if __name__ == "__main__":
    env = MarketEnvironment()
    state = env.reset()
    done = False

    while not done:
        # Przykładowo: strategia losowa
        action = np.random.choice([0, 1, 2])  # 0 = hold, 1 = buy, 2 = sell
        next_state, reward, done, info = env.step(action)
        print(
            f"Krok: {info['step']}, Akcja: {info['action']}, Nagroda: {reward:.2f}, Kapitał: {info['capital']:.2f}"
        )
    logging.info("Symulacja zakończona.")

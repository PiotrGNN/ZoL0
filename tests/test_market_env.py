"""
Tests for market environment functionality
"""
import pytest
import numpy as np
from ai_models.environment import MarketEnvironment
from ai_models.market_dummy_env import MarketDummyEnv
from utils.error_handling import TradingSystemError

@pytest.mark.unit
class TestMarketEnvironment:
    def test_initialization(self):
        """Test market environment initialization"""
        env = MarketEnvironment(initial_capital=10000)
        assert env.initial_capital == 10000
        assert env.current_capital == 10000
        assert env.position == 0
        assert env.entry_price is None

    def test_reset(self):
        """Test reset functionality"""
        env = MarketEnvironment(initial_capital=10000)
        env.current_capital = 5000  # Simulate some losses
        env.position = 1
        env.entry_price = 100
        
        state = env.reset()
        assert env.current_capital == 10000
        assert env.position == 0
        assert env.entry_price is None
        assert isinstance(state, dict)

    @pytest.mark.parametrize("action", ["buy", "sell", "hold"])
    def test_valid_actions(self, action):
        """Test valid trading actions"""
        env = MarketEnvironment(initial_capital=10000)
        state, reward, done, info = env.step(action)
        
        assert isinstance(state, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "action" in info

    def test_invalid_action(self):
        """Test handling of invalid actions"""
        env = MarketEnvironment(initial_capital=10000)
        with pytest.raises(ValueError):
            env.step("invalid_action")

@pytest.mark.unit
class TestMarketDummyEnv:
    """Test MarketDummyEnv simulation"""

    def test_price_generation(self):
        """Test price generation in dummy environment"""
        env = MarketDummyEnv(
            initial_capital=10000,
            volatility=0.01,
            commission=0.001,
            spread=0.002
        )
        
        # Reset and get initial state
        state = env.reset()
        assert state is not None
        assert state['capital'] == 10000
        assert state['price'] == 100.0  # Default base price
        
        # Take some steps and verify price changes
        prices = []
        for _ in range(10):
            state, _, _, _ = env.step('hold')
            prices.append(state['price'])
            
        # Verify price volatility
        price_changes = [abs(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
        assert all(change <= 0.02 for change in price_changes)  # Max 2% change per step

    def test_commission_and_spread(self):
        """Test commission and spread calculations"""
        env = MarketDummyEnv(
            initial_capital=10000,
            commission=0.001,  # 0.1%
            spread=0.002      # 0.2%
        )

        # Open and close a position to test costs
        initial_capital = env.current_capital
        base_price = env.current_price

        # Buy position with current price and spread
        state, _, _, _ = env.step("buy")

        # Verify spread impact on entry
        expected_entry_spread = base_price * (env.spread)
        actual_entry_spread = state['price'] - base_price
        assert actual_entry_spread >= 0  # Spread should increase buy price

        # Close position and verify commission
        final_state, _, _, _ = env.step("sell")
        
        # Calculate total costs (commissions + spread)
        total_cost = initial_capital - final_state['capital']
        assert total_cost > 0  # Should have some trading costs

        # Verify within expected range
        expected_commission = base_price * env.commission * 2  # Buy + Sell
        expected_spread_cost = base_price * env.spread  # Only applied on entry for this test
        expected_total = expected_commission + expected_spread_cost
        
        assert abs(total_cost - expected_total) < 0.01  # Allow small floating point differences
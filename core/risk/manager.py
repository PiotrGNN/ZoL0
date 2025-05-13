"""Risk management system."""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from config import get_logger
from ..exchange.types import Position

logger = get_logger()


class RiskManager:
    """Manages trading risk controls."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize risk manager.

        Args:
            config: Risk configuration
        """
        self.config = config

        # Risk limits
        self.position_limits = config.get(
            "position_limits",
            {
                "max_position_size": 0.1,  # Max 10% of equity per position
                "max_leverage": 5.0,  # Max 5x leverage
                "max_positions": 5,  # Max 5 concurrent positions
                "min_margin": 0.05,  # Min 5% free margin
            },
        )

        self.portfolio_limits = config.get(
            "portfolio_limits",
            {
                "max_drawdown": 0.2,  # Max 20% drawdown
                "max_daily_loss": 0.05,  # Max 5% daily loss
                "max_margin_usage": 0.8,  # Max 80% margin usage
                "max_portfolio_leverage": 3.0,  # Max 3x portfolio leverage
            },
        )

        # Current state
        self._positions: Dict[str, Position] = {}
        self._equity_history = []
        self._daily_pnl = 0.0
        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._last_update = datetime.now()

    def can_open_position(
        self,
        symbol: str,
        size: Optional[float] = None,
        leverage: Optional[float] = None,
    ) -> bool:
        """Check if new position allowed.

        Args:
            symbol: Trading pair symbol
            size: Position size
            leverage: Position leverage

        Returns:
            True if position allowed
        """
        try:
            # Check position count limit
            if len(self._positions) >= self.position_limits["max_positions"]:
                logger.warning("Max positions limit reached")
                return False

            # Check existing position
            if symbol in self._positions:
                return self._check_position_increase(symbol, size, leverage)

            # Check new position
            return self._check_new_position(size, leverage)

        except Exception as e:
            logger.error(f"Position check error: {e}")
            return False

    def update_position(
        self, symbol: str, position: Position, current_price: float
    ) -> None:
        """Update position state.

        Args:
            symbol: Trading pair symbol
            position: Position details
            current_price: Current market price
        """
        try:
            # Update position tracking
            self._positions[symbol] = position

            # Update equity and PnL
            self._update_metrics(position, current_price)

            # Check risk breaches
            self._check_risk_limits(symbol)

        except Exception as e:
            logger.error(f"Position update error: {e}")

    def close_position(self, symbol: str) -> None:
        """Remove closed position.

        Args:
            symbol: Trading pair symbol
        """
        try:
            if symbol in self._positions:
                self._positions.pop(symbol)
                logger.info(f"Position closed for {symbol}")

        except Exception as e:
            logger.error(f"Position close error: {e}")

    def get_position_size(
        self, symbol: str, entry_price: float, risk_amount: float
    ) -> float:
        """Calculate position size.

        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            risk_amount: Amount to risk

        Returns:
            Position size
        """
        try:
            # Get equity and limits
            equity = self._current_equity
            max_size = equity * self.position_limits["max_position_size"]

            # Calculate size based on risk
            size = risk_amount / entry_price

            # Apply limits
            return min(size, max_size)

        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get risk status.

        Returns:
            Risk metrics
        """
        try:
            return {
                "positions": len(self._positions),
                "current_equity": self._current_equity,
                "peak_equity": self._peak_equity,
                "daily_pnl": self._daily_pnl,
                "drawdown": self._calculate_drawdown(),
                "margin_usage": self._calculate_margin_usage(),
                "portfolio_leverage": self._calculate_portfolio_leverage(),
                "risk_limits": {
                    "position": self.position_limits,
                    "portfolio": self.portfolio_limits,
                },
            }

        except Exception as e:
            logger.error(f"Status error: {e}")
            return {}

    def _check_new_position(
        self, size: Optional[float], leverage: Optional[float]
    ) -> bool:
        """Check if new position allowed.

        Args:
            size: Position size
            leverage: Position leverage

        Returns:
            True if allowed
        """
        try:
            # Check leverage limit
            if leverage and leverage > self.position_limits["max_leverage"]:
                logger.warning(f"Leverage {leverage}x exceeds limit")
                return False

            # Check size limit
            if size:
                max_size = (
                    self._current_equity * self.position_limits["max_position_size"]
                )
                if size > max_size:
                    logger.warning(f"Size {size} exceeds limit")
                    return False

            # Check margin requirement
            margin_usage = self._calculate_margin_usage()
            if margin_usage > (1 - self.position_limits["min_margin"]):
                logger.warning("Insufficient margin available")
                return False

            return True

        except Exception as e:
            logger.error(f"New position check error: {e}")
            return False

    def _check_position_increase(
        self, symbol: str, size: Optional[float], leverage: Optional[float]
    ) -> bool:
        """Check if position increase allowed.

        Args:
            symbol: Trading pair symbol
            size: Additional size
            leverage: New leverage

        Returns:
            True if allowed
        """
        try:
            position = self._positions[symbol]

            # Check leverage increase
            if leverage and leverage > position.leverage:
                if leverage > self.position_limits["max_leverage"]:
                    logger.warning(f"Leverage increase to {leverage}x denied")
                    return False

            # Check size increase
            if size:
                new_size = position.size + size
                max_size = (
                    self._current_equity * self.position_limits["max_position_size"]
                )
                if new_size > max_size:
                    logger.warning(f"Size increase to {new_size} denied")
                    return False

            return True

        except Exception as e:
            logger.error(f"Position increase check error: {e}")
            return False

    def _update_metrics(self, position: Position, current_price: float) -> None:
        """Update risk metrics.

        Args:
            position: Position details
            current_price: Current market price
        """
        try:
            # Update equity
            old_equity = self._current_equity
            unrealized_pnl = position.unrealized_pnl
            self._current_equity += unrealized_pnl

            # Update peak equity
            if self._current_equity > self._peak_equity:
                self._peak_equity = self._current_equity

            # Update daily PnL
            now = datetime.now()
            if now.date() > self._last_update.date():
                self._daily_pnl = 0.0
            self._daily_pnl += self._current_equity - old_equity

            # Update equity history
            self._equity_history.append(
                {"timestamp": now, "equity": self._current_equity}
            )

            # Trim history to last 30 days
            cutoff = now - timedelta(days=30)
            self._equity_history = [
                h for h in self._equity_history if h["timestamp"] >= cutoff
            ]

            self._last_update = now

        except Exception as e:
            logger.error(f"Metrics update error: {e}")

    def _check_risk_limits(self, symbol: str) -> None:
        """Check for risk limit breaches.

        Args:
            symbol: Trading pair symbol
        """
        try:
            # Check drawdown
            drawdown = self._calculate_drawdown()
            if drawdown > self.portfolio_limits["max_drawdown"]:
                logger.warning(f"Max drawdown breached: {drawdown:.1%}")
                self._handle_risk_breach("drawdown", symbol)

            # Check daily loss
            if abs(self._daily_pnl) > (
                self._current_equity * self.portfolio_limits["max_daily_loss"]
            ):
                logger.warning(f"Max daily loss breached: {self._daily_pnl:.1f}")
                self._handle_risk_breach("daily_loss", symbol)

            # Check margin usage
            margin_usage = self._calculate_margin_usage()
            if margin_usage > self.portfolio_limits["max_margin_usage"]:
                logger.warning(f"Max margin usage breached: {margin_usage:.1%}")
                self._handle_risk_breach("margin", symbol)

            # Check portfolio leverage
            leverage = self._calculate_portfolio_leverage()
            if leverage > self.portfolio_limits["max_portfolio_leverage"]:
                logger.warning(f"Max portfolio leverage breached: {leverage:.1f}x")
                self._handle_risk_breach("leverage", symbol)

        except Exception as e:
            logger.error(f"Risk check error: {e}")

    def _handle_risk_breach(self, breach_type: str, symbol: str) -> None:
        """Handle risk limit breach.

        Args:
            breach_type: Type of breach
            symbol: Symbol that triggered breach
        """
        try:
            # Log breach
            logger.warning(f"Risk breach: {breach_type} on {symbol}")
            # Example risk response: reduce position size and adjust leverage
            self.reduce_position(symbol)
            self.adjust_leverage(symbol)
            logger.info(f"Risk response actions executed for {symbol}")
        except Exception as e:
            logger.error(f"Error handling risk breach: {e}")
            raise

    def reduce_position(self, symbol: str):
        # Implement logic to reduce position size for the symbol
        logger.info(f"Reducing position size for {symbol}")
        # ...

    def adjust_leverage(self, symbol: str):
        # Implement logic to adjust leverage for the symbol
        logger.info(f"Adjusting leverage for {symbol}")
        # ...

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown.

        Returns:
            Current drawdown percentage
        """
        try:
            if not self._peak_equity:
                return 0.0

            return (self._peak_equity - self._current_equity) / self._peak_equity

        except Exception as e:
            logger.error(f"Drawdown calculation error: {e}")
            return 0.0

    def _calculate_margin_usage(self) -> float:
        """Calculate margin usage.

        Returns:
            Margin usage ratio
        """
        try:
            if not self._current_equity:
                return 0.0

            used_margin = sum(pos.margin for pos in self._positions.values())
            return used_margin / self._current_equity

        except Exception as e:
            logger.error(f"Margin usage calculation error: {e}")
            return 0.0

    def _calculate_portfolio_leverage(self) -> float:
        """Calculate portfolio leverage.

        Returns:
            Portfolio leverage ratio
        """
        try:
            if not self._current_equity:
                return 0.0

            total_exposure = sum(
                abs(pos.size * pos.current_price)
                for pos in self._positions.values()
                if pos.current_price
            )
            return total_exposure / self._current_equity

        except Exception as e:
            logger.error(f"Portfolio leverage calculation error: {e}")
            return 0.0

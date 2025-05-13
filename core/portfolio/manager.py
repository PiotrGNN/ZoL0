"""Portfolio management system."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from config import get_logger
from ..exchange.types import Position, Trade, Order
from ..risk.manager import RiskManager
from ..database import DatabaseManager

logger = get_logger()


class PortfolioManager:
    """Manages trading portfolio and tracks performance."""

    def __init__(self, initial_balance: float, base_currency: str, db_manager: DatabaseManager):
        """Initialize portfolio manager.

        Args:
            initial_balance: Initial balance of the portfolio
            base_currency: Base currency of the portfolio
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self.base_currency = base_currency
        self._initial_equity = initial_balance
        self._current_equity = initial_balance

        # Portfolio state
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._open_orders: Dict[str, Order] = {}

        # Performance tracking
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._fees_paid = 0.0

        # Historical metrics
        self._equity_history = []
        self._trades_history = []
        self._daily_pnl = {}

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
            old_position = self._positions.get(symbol)

            # Update position
            self._positions[symbol] = position

            # Calculate PnL changes
            if old_position:
                pnl_change = position.unrealized_pnl - old_position.unrealized_pnl
                self._unrealized_pnl += pnl_change
            else:
                self._unrealized_pnl += position.unrealized_pnl

            # Update equity
            self._update_equity()

            # Update risk manager
            self.risk_manager.update_position(symbol, position, current_price)

        except Exception as e:
            logger.error(f"Position update error: {e}")

    def add_trade(self, trade: Trade) -> None:
        """Add executed trade.

        Args:
            trade: Trade details
        """
        try:
            # Add to history
            self._trades.append(trade)
            self._trades_history.append(
                {
                    "timestamp": trade.timestamp,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "size": trade.size,
                    "price": trade.price,
                    "fee": trade.fee,
                }
            )

            # Update realized PnL
            self._realized_pnl += trade.realized_pnl
            self._fees_paid += trade.fee

            # Update daily PnL
            trade_date = trade.timestamp.date()
            if trade_date not in self._daily_pnl:
                self._daily_pnl[trade_date] = 0.0
            self._daily_pnl[trade_date] += trade.realized_pnl - trade.fee

            # Update equity
            self._update_equity()

        except Exception as e:
            logger.error(f"Trade update error: {e}")

    def update_order(self, order: Order) -> None:
        """Update order state.

        Args:
            order: Order details
        """
        try:
            if order.status == "FILLED":
                self._open_orders.pop(order.id, None)
            else:
                self._open_orders[order.id] = order

        except Exception as e:
            logger.error(f"Order update error: {e}")

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position details.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position if exists
        """
        return self._positions.get(symbol)

    def get_open_orders(self, symbol: str) -> List[Order]:
        """Get open orders.

        Args:
            symbol: Trading pair symbol

        Returns:
            List of open orders
        """
        return [order for order in self._open_orders.values() if order.symbol == symbol]

    def get_performance_metrics(self, timeframe: str = "1d") -> Dict[str, Any]:
        """Get performance metrics.

        Args:
            timeframe: Timeframe for metrics

        Returns:
            Performance metrics
        """
        try:
            now = datetime.now()

            if timeframe == "1d":
                start_time = now - timedelta(days=1)
            elif timeframe == "1w":
                start_time = now - timedelta(weeks=1)
            elif timeframe == "1m":
                start_time = now - timedelta(days=30)
            else:
                start_time = now - timedelta(days=365)

            # Filter trades in timeframe
            period_trades = [
                t for t in self._trades_history if t["timestamp"] >= start_time
            ]

            # Calculate metrics
            total_trades = len(period_trades)
            winning_trades = len([t for t in period_trades if t["realized_pnl"] > 0])

            period_pnl = sum(t["realized_pnl"] - t["fee"] for t in period_trades)

            period_fees = sum(t["fee"] for t in period_trades)

            return {
                "timeframe": timeframe,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": (
                    winning_trades / total_trades if total_trades > 0 else 0.0
                ),
                "realized_pnl": period_pnl,
                "unrealized_pnl": self._unrealized_pnl,
                "fees_paid": period_fees,
                "return_pct": (
                    (self._current_equity - self._initial_equity) / self._initial_equity
                    if self._initial_equity > 0
                    else 0.0
                ),
                "sharpe_ratio": self._calculate_sharpe_ratio(start_time),
                "max_drawdown": self._calculate_max_drawdown(start_time),
                "current_equity": self._current_equity,
                "risk_metrics": self.risk_manager.get_status(),
            }

        except Exception as e:
            logger.error(f"Performance calculation error: {e}")
            return {}

    def _update_equity(self) -> None:
        """Update portfolio equity."""
        try:
            # Calculate total equity
            self._current_equity = (
                self._initial_equity
                + self._realized_pnl
                + self._unrealized_pnl
                - self._fees_paid
            )

            # Update equity history
            self._equity_history.append(
                {
                    "timestamp": datetime.now(),
                    "equity": self._current_equity,
                    "realized_pnl": self._realized_pnl,
                    "unrealized_pnl": self._unrealized_pnl,
                    "fees": self._fees_paid,
                }
            )

            # Trim history to last 365 days
            cutoff = datetime.now() - timedelta(days=365)
            self._equity_history = [
                h for h in self._equity_history if h["timestamp"] >= cutoff
            ]

        except Exception as e:
            logger.error(f"Equity update error: {e}")

    def _calculate_sharpe_ratio(self, start_time: datetime) -> float:
        """Calculate Sharpe ratio.

        Args:
            start_time: Start time for calculation

        Returns:
            Sharpe ratio
        """
        try:
            # Get daily returns
            equity_points = [
                h["equity"]
                for h in self._equity_history
                if h["timestamp"] >= start_time
            ]

            if len(equity_points) < 2:
                return 0.0

            # Calculate daily returns
            daily_returns = np.diff(equity_points) / equity_points[:-1]

            # Calculate annualized Sharpe ratio
            if len(daily_returns) == 0:
                return 0.0

            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)

            if std_return == 0:
                return 0.0

            # Assume 252 trading days per year
            sharpe = np.sqrt(252) * mean_return / std_return

            return float(sharpe)

        except Exception as e:
            logger.error(f"Sharpe calculation error: {e}")
            return 0.0

    def _calculate_max_drawdown(self, start_time: datetime) -> float:
        """Calculate maximum drawdown.

        Args:
            start_time: Start time for calculation

        Returns:
            Maximum drawdown percentage
        """
        try:
            # Get equity points
            equity_points = [
                h["equity"]
                for h in self._equity_history
                if h["timestamp"] >= start_time
            ]

            if len(equity_points) < 2:
                return 0.0

            # Calculate running maximum
            running_max = np.maximum.accumulate(equity_points)

            # Calculate drawdowns
            drawdowns = (running_max - equity_points) / running_max

            # Get maximum drawdown
            max_drawdown = np.max(drawdowns)

            return float(max_drawdown)

        except Exception as e:
            logger.error(f"Drawdown calculation error: {e}")
            return 0.0

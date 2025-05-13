"""Core trading engine."""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from threading import Thread, Event, Lock
import queue
from config import get_logger
from ..exchange import ExchangeConnector
from ..portfolio import PortfolioManager
from ..risk import RiskManager
from ..exchange.types import OrderSide, OrderType, Position, Order, Trade
from ..strategies import StrategyManager

logger = get_logger()


class TradingEngine:
    """Manages core trading operations."""

    def __init__(
        self,
        exchange: ExchangeConnector,
        portfolio_manager: PortfolioManager,
        risk_manager: RiskManager,
        strategy_manager: StrategyManager,
        config: Dict[str, Any],
    ):
        """Initialize trading engine.

        Args:
            exchange: Exchange connector
            portfolio_manager: Portfolio manager
            risk_manager: Risk manager
            strategy_manager: Strategy manager
            config: Engine configuration
        """
        self.exchange = exchange
        self.portfolio = portfolio_manager
        self.risk = risk_manager
        self.strategy = strategy_manager
        self.config = config

        # Trading state
        self._running = False
        self._paused = False
        self._last_health_check = 0
        self._health_check_interval = 60  # seconds

        # Thread synchronization
        self._stop_event = Event()
        self._lock = Lock()
        self._order_queue = queue.Queue()

        # Active orders and positions
        self._active_orders: Dict[str, Order] = {}
        self._pending_orders: Dict[str, Order] = {}

        # Performance tracking
        self._daily_stats = {"trades": 0, "volume": 0.0, "fees": 0.0}

    def start(self) -> None:
        """Start trading engine."""
        if self._running:
            logger.warning("Trading engine already running")
            return

        try:
            # Initialize components
            if not self._initialize():
                raise RuntimeError("Failed to initialize trading engine")

            # Start processing threads
            self._running = True
            Thread(target=self._process_orders, daemon=True).start()
            Thread(target=self._monitor_positions, daemon=True).start()
            Thread(target=self._update_markets, daemon=True).start()

            logger.info("Trading engine started")

        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
            self.stop()

    def stop(self) -> None:
        """Stop trading engine."""
        if not self._running:
            return

        try:
            # Signal threads to stop
            self._running = False
            self._stop_event.set()

            # Cancel all active orders
            self._cancel_all_orders()

            # Close connections
            self.exchange.close()

            logger.info("Trading engine stopped")

        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")

    def pause(self) -> None:
        """Pause trading operations."""
        if not self._running:
            return

        self._paused = True
        logger.info("Trading engine paused")

    def resume(self) -> None:
        """Resume trading operations."""
        if not self._running:
            return

        self._paused = False
        logger.info("Trading engine resumed")

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        strategy_id: Optional[str] = None,
    ) -> Optional[str]:
        """Place new trading order.

        Args:
            symbol: Trading pair symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price
            strategy_id: Strategy identifier

        Returns:
            Order ID if successful
        """
        try:
            # Validate trading state
            if not self._running or self._paused:
                logger.warning("Trading engine not active")
                return None

            # Check risk limits
            if not self.risk.can_open_position(symbol):
                logger.warning(f"Risk check failed for {symbol}")
                return None

            # Create and queue order
            order = Order(
                id="pending",  # Will be set by exchange
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                status="NEW",
                time_in_force=TimeInForce.GOOD_TILL_CANCEL,
                reduce_only=False,
                created_time=datetime.now(),
            )

            self._order_queue.put((order, strategy_id))
            return "pending"

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order.

        Args:
            order_id: Order ID

        Returns:
            True if cancelled
        """
        try:
            # Check if order exists
            order = self._active_orders.get(order_id) or self._pending_orders.get(
                order_id
            )
            if not order:
                logger.warning(f"Order {order_id} not found")
                return False

            # Cancel on exchange
            if not self.exchange.cancel_order(order.symbol, order_id):
                return False

            # Remove from tracking
            with self._lock:
                self._active_orders.pop(order_id, None)
                self._pending_orders.pop(order_id, None)

            return True

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get engine status.

        Returns:
            Status information
        """
        return {
            "running": self._running,
            "paused": self._paused,
            "active_orders": len(self._active_orders),
            "pending_orders": len(self._pending_orders),
            "daily_stats": self._daily_stats,
            "last_health_check": datetime.fromtimestamp(
                self._last_health_check
            ).isoformat(),
            "components": {
                "exchange": self.exchange.get_status(),
                "portfolio": self.portfolio.get_summary(),
                "risk": self.risk.get_status(),
            },
        }

    def _initialize(self) -> bool:
        """Initialize trading engine.

        Returns:
            True if successful
        """
        try:
            # Check exchange connection
            if not self.exchange.is_connected():
                logger.error("Exchange not connected")
                return False

            # Load existing positions
            positions = self.exchange.get_positions()
            for pos in positions:
                self.portfolio.update_position(
                    pos.symbol, pos, pos.current_price or pos.entry_price
                )

            # Initialize strategies
            if not self.strategy.initialize():
                logger.error("Failed to initialize strategies")
                return False

            self._last_health_check = time.time()
            return True

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False

    def _process_orders(self) -> None:
        """Process order queue."""
        while self._running:
            try:
                # Wait for new orders
                try:
                    order, strategy_id = self._order_queue.get(timeout=1)
                except queue.Empty:
                    continue

                if self._paused:
                    self._order_queue.put((order, strategy_id))
                    time.sleep(1)
                    continue

                # Submit to exchange
                placed_order = self.exchange.place_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.type,
                    quantity=order.quantity,
                    price=order.price,
                    time_in_force=order.time_in_force,
                    reduce_only=order.reduce_only,
                )

                if not placed_order:
                    logger.error(f"Failed to place order: {order}")
                    continue

                # Update tracking
                with self._lock:
                    self._active_orders[placed_order.id] = placed_order
                    if strategy_id:
                        self.strategy.on_order_placed(strategy_id, placed_order)

            except Exception as e:
                logger.error(f"Order processing error: {e}")

    def _monitor_positions(self) -> None:
        """Monitor open positions."""
        while self._running:
            try:
                if self._paused:
                    time.sleep(1)
                    continue

                # Get current positions
                positions = self.exchange.get_positions()

                for pos in positions:
                    # Update portfolio
                    self.portfolio.update_position(
                        pos.symbol, pos, pos.current_price or pos.entry_price
                    )

                    # Update risk metrics
                    self.risk.update_position(
                        pos.symbol, pos, pos.current_price or pos.entry_price
                    )

                    # Check strategy exits
                    if pos.strategy:
                        self.strategy.check_exit_signals(pos.strategy, pos.symbol)

                time.sleep(5)  # Position check interval

            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                time.sleep(5)

    def _update_markets(self) -> None:
        """Update market data."""
        while self._running:
            try:
                if self._paused:
                    time.sleep(1)
                    continue

                # Update strategy data
                self.strategy.update_market_data()

                # Health check
                self._check_health()

                time.sleep(1)  # Market update interval

            except Exception as e:
                logger.error(f"Market update error: {e}")
                time.sleep(5)

    def _check_health(self) -> None:
        """Perform system health check."""
        now = time.time()

        # Check at regular intervals
        if now - self._last_health_check < self._health_check_interval:
            return

        try:
            # Check component health
            checks = {
                "exchange": self.exchange.is_connected(),
                "database": self.portfolio.is_healthy(),
                "strategies": self.strategy.is_healthy(),
            }

            # Handle unhealthy state
            if not all(checks.values()):
                logger.error(f"Health check failed: {checks}")
                self.pause()

            self._last_health_check = now

        except Exception as e:
            logger.error(f"Health check error: {e}")

    def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        try:
            # Cancel active orders
            for order_id in list(self._active_orders.keys()):
                self.cancel_order(order_id)

            # Clear pending orders
            self._pending_orders.clear()

            # Clear order queue
            while not self._order_queue.empty():
                try:
                    self._order_queue.get_nowait()
                except queue.Empty:
                    break

        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

    def _update_daily_stats(self, trade: Trade) -> None:
        """Update daily trading statistics.

        Args:
            trade: Completed trade
        """
        try:
            self._daily_stats["trades"] += 1
            self._daily_stats["volume"] += trade.size * trade.exit_price
            self._daily_stats["fees"] += trade.fees

        except Exception as e:
            logger.error(f"Error updating daily stats: {e}")


from ..exchange.types import TimeInForce  # Add missing import

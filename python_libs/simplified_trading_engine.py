#!/usr/bin/env python3
"""
simplified_trading_engine.py - A simplified trading engine implementation.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta


class SimplifiedTradingEngine:
    """
    A simplified trading engine for testing trading strategies.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
        slippage: float = 0.0005,
        log_level: str = "INFO",
    ):
        """
        Initialize the simplified trading engine.

        Args:
            initial_capital: Initial capital amount
            commission_rate: Commission rate as a decimal (e.g., 0.001 = 0.1%)
            slippage: Slippage rate as a decimal
            log_level: Logging level
        """
        # Configure logging
        self.logger = logging.getLogger(__name__)
        numeric_level = getattr(logging, log_level.upper(), None)
        if isinstance(numeric_level, int):
            self.logger.setLevel(numeric_level)

        # Trading parameters
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage

        # Trading state
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []

        # Performance metrics
        self.metrics = {
            "total_trades": 0,
            "profitable_trades": 0,
            "loss_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "annual_return": 0.0,
        }

        self.logger.info(
            f"SimplifiedTradingEngine initialized with {initial_capital} capital"
        )

    def reset(self) -> None:
        """Reset the engine to initial state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.logger.info("Trading engine reset to initial state")

    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a new order.

        Args:
            symbol: Trading symbol
            order_type: Type of order ("market", "limit", "stop", "stop_limit")
            side: Order side ("buy" or "sell")
            quantity: Order quantity
            price: Limit price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop and stop_limit orders)
            take_profit: Take profit price
            stop_loss: Stop loss price

        Returns:
            Dict containing order details
        """
        # Validate order parameters
        if order_type not in ["market", "limit", "stop", "stop_limit"]:
            self.logger.error(f"Invalid order type: {order_type}")
            return {"success": False, "error": f"Invalid order type: {order_type}"}

        if side not in ["buy", "sell"]:
            self.logger.error(f"Invalid order side: {side}")
            return {"success": False, "error": f"Invalid order side: {side}"}

        if order_type in ["limit", "stop_limit"] and price is None:
            self.logger.error(f"Price required for {order_type} orders")
            return {
                "success": False,
                "error": f"Price required for {order_type} orders",
            }

        if order_type in ["stop", "stop_limit"] and stop_price is None:
            self.logger.error(f"Stop price required for {order_type} orders")
            return {
                "success": False,
                "error": f"Stop price required for {order_type} orders",
            }

        # Create order
        order = {
            "id": f"order_{len(self.orders) + 1}",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "quantity": quantity,
            "price": price,
            "stop_price": stop_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "status": "open",
            "filled": 0.0,
            "average_price": 0.0,
            "created_time": datetime.now(),
            "updated_time": datetime.now(),
        }

        self.orders.append(order)
        self.logger.info(f"Order placed: {order['id']} ({side} {quantity} {symbol})")
        return {"success": True, "order": order}

    def update(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Update the engine with new market data and process orders.

        Args:
            market_data: Dictionary containing market data

        Returns:
            List of filled orders
        """
        symbol = market_data.get("symbol")
        timestamp = market_data.get("timestamp", datetime.now())
        current_price = market_data.get("price", 0.0)

        if not symbol or current_price <= 0:
            self.logger.error("Invalid market data provided")
            return []

        # Process open orders
        filled_orders = []
        for order in self.orders:
            if order["status"] != "open" or order["symbol"] != symbol:
                continue

            # Check if order should be filled
            should_fill = False
            fill_price = current_price

            if order["type"] == "market":
                should_fill = True
            elif order["type"] == "limit":
                if (order["side"] == "buy" and current_price <= order["price"]) or (
                    order["side"] == "sell" and current_price >= order["price"]
                ):
                    should_fill = True
                    fill_price = order["price"]
            elif order["type"] == "stop":
                if (
                    order["side"] == "buy" and current_price >= order["stop_price"]
                ) or (order["side"] == "sell" and current_price <= order["stop_price"]):
                    should_fill = True
            elif order["type"] == "stop_limit":
                if (
                    order["side"] == "buy" and current_price >= order["stop_price"]
                ) or (order["side"] == "sell" and current_price <= order["stop_price"]):
                    if (order["side"] == "buy" and current_price <= order["price"]) or (
                        order["side"] == "sell" and current_price >= order["price"]
                    ):
                        should_fill = True
                        fill_price = order["price"]

            # Fill order if conditions met
            if should_fill:
                # Apply slippage
                if order["side"] == "buy":
                    fill_price = fill_price * (1 + self.slippage)
                else:
                    fill_price = fill_price * (1 - self.slippage)

                # Calculate commission
                commission = order["quantity"] * fill_price * self.commission_rate

                # Update order
                order["status"] = "filled"
                order["filled"] = order["quantity"]
                order["average_price"] = fill_price
                order["updated_time"] = timestamp

                # Update position
                position_key = f"{symbol}"
                if position_key in self.positions:
                    position = self.positions[position_key]
                    if order["side"] == "buy":
                        new_quantity = position["quantity"] + order["quantity"]
                        new_value = position["value"] + (order["quantity"] * fill_price)
                        position["average_price"] = new_value / new_quantity
                        position["quantity"] = new_quantity
                        position["value"] = new_value
                    else:  # sell
                        position["quantity"] -= order["quantity"]
                        position["value"] -= (
                            order["quantity"] * position["average_price"]
                        )
                        if position["quantity"] <= 0:
                            # Position closed
                            pnl = (fill_price - position["average_price"]) * order[
                                "quantity"
                            ]
                            self.capital += pnl - commission
                            if pnl > 0:
                                self.metrics["profitable_trades"] += 1
                            else:
                                self.metrics["loss_trades"] += 1
                            self.metrics["total_trades"] += 1
                            del self.positions[position_key]
                        else:
                            # Position partially closed
                            pnl = (fill_price - position["average_price"]) * order[
                                "quantity"
                            ]
                            self.capital += pnl - commission
                else:
                    if order["side"] == "buy":
                        self.positions[position_key] = {
                            "symbol": symbol,
                            "quantity": order["quantity"],
                            "average_price": fill_price,
                            "value": order["quantity"] * fill_price,
                            "side": "long",
                            "open_time": timestamp,
                        }
                        self.capital -= order["quantity"] * fill_price + commission

                # Add to filled orders
                filled_orders.append(order)
                self.logger.info(f"Order filled: {order['id']} at {fill_price}")

        # Update equity curve
        portfolio_value = self.capital
        for pos in self.positions.values():
            portfolio_value += pos["quantity"] * current_price

        self.equity_curve.append({"timestamp": timestamp, "equity": portfolio_value})

        return filled_orders

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions."""
        return self.positions

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get open orders."""
        return [order for order in self.orders if order["status"] == "open"]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID."""
        for order in self.orders:
            if order["id"] == order_id and order["status"] == "open":
                order["status"] = "canceled"
                order["updated_time"] = datetime.now()
                self.logger.info(f"Order canceled: {order_id}")
                return True

        self.logger.warning(f"Order not found or already filled: {order_id}")
        return False

    def get_portfolio_value(self, market_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.

        Args:
            market_prices: Dictionary mapping symbols to current prices

        Returns:
            Total portfolio value
        """
        portfolio_value = self.capital

        for symbol, position in self.positions.items():
            if symbol in market_prices:
                price = market_prices[symbol]
                portfolio_value += position["quantity"] * price

        return portfolio_value

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics."""
        if len(self.equity_curve) < 2:
            return self.metrics

        equity_values = [point["equity"] for point in self.equity_curve]
        returns = pd.Series(equity_values).pct_change().dropna()

        # Calculate basic metrics
        total_return = (equity_values[-1] / self.initial_capital) - 1

        # Calculate drawdown
        peak = pd.Series(equity_values).cummax()
        drawdown = (pd.Series(equity_values) - peak) / peak
        max_drawdown = drawdown.min()

        # Annualized metrics (assuming daily data)
        days = (
            self.equity_curve[-1]["timestamp"] - self.equity_curve[0]["timestamp"]
        ).days
        if days > 0:
            annual_return = ((1 + total_return) ** (365 / days)) - 1
            sharpe_ratio = (
                (returns.mean() / returns.std()) * (252**0.5)
                if returns.std() > 0
                else 0
            )
        else:
            annual_return = 0
            sharpe_ratio = 0

        # Win rate
        total_trades = self.metrics["profitable_trades"] + self.metrics["loss_trades"]
        win_rate = (
            self.metrics["profitable_trades"] / total_trades if total_trades > 0 else 0
        )

        # Update metrics
        self.metrics.update(
            {
                "win_rate": win_rate,
                "total_return": total_return,
                "annual_return": annual_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
            }
        )

        return self.metrics

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run a backtest using historical data and a strategy function.

        Args:
            data: Historical price data as pandas DataFrame
            strategy_fn: Strategy function that takes current data and engine state
                         and returns trading signals

        Returns:
            Dict containing backtest results and performance metrics
        """
        self.reset()

        # Ensure data is sorted by time
        if "timestamp" in data.columns:
            data = data.sort_values("timestamp")

        for i, row in data.iterrows():
            # Prepare market data
            market_data = {
                "symbol": row.get("symbol", "UNKNOWN"),
                "timestamp": row.get("timestamp", datetime.now()),
                "price": row.get("close", 0.0),
                "open": row.get("open", 0.0),
                "high": row.get("high", 0.0),
                "low": row.get("low", 0.0),
                "volume": row.get("volume", 0.0),
            }

            # Process current market data
            self.update(market_data)

            # Get current engine state
            engine_state = {
                "capital": self.capital,
                "positions": self.positions,
                "open_orders": self.get_open_orders(),
            }

            # Call strategy function to get signals
            current_index = int(np.asarray(i))
            signals = strategy_fn(data.iloc[: current_index + 1], engine_state)

            # Process signals
            if "orders" in signals:
                for order_params in signals["orders"]:
                    self.place_order(**order_params)

            if "cancel_orders" in signals:
                for order_id in signals["cancel_orders"]:
                    self.cancel_order(order_id)

        # Calculate final metrics
        metrics = self.calculate_metrics()

        # Prepare backtest results
        results = {
            "initial_capital": self.initial_capital,
            "final_capital": self.capital,
            "positions": self.positions,
            "metrics": metrics,
            "equity_curve": self.equity_curve,
        }

        self.logger.info(f"Backtest completed: {metrics['total_return']:.2%} return")
        return results

"""Bybit exchange connector."""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hmac
import hashlib
import requests
import pandas as pd
from config import get_logger
from .types import OrderSide, OrderType, Position, TimeInForce, Order

logger = get_logger()


class ExchangeConnector:
    """Manages communication with Bybit exchange."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """Initialize exchange connector.

        Args:
            api_key: API key
            api_secret: API secret
            testnet: Use testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # API URLs
        self.base_url = (
            "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        )

        # Rate limiting
        self._request_weight = 0
        self._last_request_reset = time.time()
        self.rate_limits = {
            "weight_per_minute": 120,
            "orders_per_minute": 10,
            "order_rate_remaining": 10,
            "weight_remaining": 120,
        }

        # Connection state
        self._connected = False
        self._last_heartbeat = 0

        # Initialize connection
        self._connect()

    def _connect(self) -> bool:
        """Establish API connection.

        Returns:
            True if connected
        """
        try:
            # Test connection
            response = self._send_request(method="GET", path="/v5/market/time")

            if response and response.get("retCode") == 0:
                self._connected = True
                self._last_heartbeat = time.time()
                logger.info(
                    f"Connected to Bybit {'testnet' if self.testnet else 'mainnet'}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if exchange is connected.

        Returns:
            True if connected
        """
        # Check heartbeat age
        if time.time() - self._last_heartbeat > 60:
            self._connected = False

        return self._connected

    def get_positions(self) -> List[Position]:
        """Get current positions.

        Returns:
            List of positions
        """
        try:
            response = self._send_request(method="GET", path="/v5/position/list")

            if not response or response.get("retCode") != 0:
                return []

            positions = []
            for pos in response["result"]["list"]:
                if float(pos["size"]) == 0:
                    continue

                positions.append(
                    Position(
                        symbol=pos["symbol"],
                        size=float(pos["size"]),
                        entry_price=float(pos["entryPrice"]),
                        leverage=float(pos["leverage"]),
                        liquidation_price=float(pos["liqPrice"]),
                        unrealized_pnl=float(pos["unrealisedPnl"]),
                        margin=float(pos["positionIM"]),
                        entry_time=datetime.fromtimestamp(
                            int(pos["createdTime"]) / 1000
                        ),
                    )
                )

            return positions

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get candlestick data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            limit: Number of candles

        Returns:
            DataFrame with OHLCV data
        """
        try:
            response = self._send_request(
                method="GET",
                path="/v5/market/kline",
                params={
                    "symbol": symbol,
                    "interval": timeframe,
                    "limit": min(limit, 1000),
                },
            )

            if not response or response.get("retCode") != 0:
                return pd.DataFrame()

            # Convert to DataFrame
            data = response["result"]["list"]
            df = pd.DataFrame(
                data, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            return df.set_index("timestamp").sort_index()

        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return pd.DataFrame()

    def get_historical_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get historical candlestick data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            start_time: Start time
            end_time: End time (optional)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Set end time to now if not provided
            if end_time is None:
                end_time = datetime.now()

            # Convert times to timestamps
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)

            frames = []
            current_ts = start_ts

            # Fetch data in chunks due to API limits
            while current_ts < end_ts:
                response = self._send_request(
                    method="GET",
                    path="/v5/market/kline",
                    params={
                        "symbol": symbol,
                        "interval": timeframe,
                        "start": current_ts,
                        "limit": 1000,
                    },
                )

                if not response or response.get("retCode") != 0:
                    break

                # Convert to DataFrame
                data = response["result"]["list"]
                if not data:
                    break

                df = pd.DataFrame(
                    data,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                frames.append(df)

                # Update timestamp for next request
                current_ts = int(df["timestamp"].astype(int).max())

                # Rate limit compliance
                time.sleep(0.5)

            if not frames:
                return pd.DataFrame()

            # Combine chunks
            result = pd.concat(frames)

            # Convert types
            result["timestamp"] = pd.to_datetime(
                result["timestamp"].astype(int), unit="ms"
            )
            for col in ["open", "high", "low", "close", "volume"]:
                result[col] = result[col].astype(float)

            return result.set_index("timestamp").sort_index()

        except Exception as e:
            logger.error(f"Error getting historical klines for {symbol}: {e}")
            return pd.DataFrame()

    def get_orderbook(
        self, symbol: str, limit: int = 20
    ) -> Dict[str, List[List[float]]]:
        """Get current orderbook.

        Args:
            symbol: Trading pair symbol
            limit: Orderbook depth

        Returns:
            Dictionary with bids and asks
        """
        try:
            response = self._send_request(
                method="GET",
                path="/v5/market/orderbook",
                params={"symbol": symbol, "limit": min(limit, 200)},
            )

            if not response or response.get("retCode") != 0:
                return {"bids": [], "asks": []}

            data = response["result"]
            return {
                "bids": [[float(p), float(s)] for p, s in data["b"]],
                "asks": [[float(p), float(s)] for p, s in data["a"]],
            }

        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {e}")
            return {"bids": [], "asks": []}

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GOOD_TILL_CANCEL,
        reduce_only: bool = False,
        close_on_trigger: bool = False,
    ) -> Optional[Order]:
        """Place new order.

        Args:
            symbol: Trading pair symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price (required for limit orders)
            time_in_force: Time in force
            reduce_only: Whether order reduces position only
            close_on_trigger: Whether order closes position

        Returns:
            Order details if successful
        """
        try:
            # Check rate limits
            if self.rate_limits["order_rate_remaining"] <= 0:
                logger.warning("Order rate limit reached")
                return None

            # Prepare order parameters
            params = {
                "symbol": symbol,
                "side": side.value,
                "orderType": order_type.value,
                "qty": str(quantity),
                "timeInForce": time_in_force.value,
                "reduceOnly": reduce_only,
                "closeOnTrigger": close_on_trigger,
            }

            # Add price for limit orders
            if order_type == OrderType.LIMIT and price is not None:
                params["price"] = str(price)

            response = self._send_request(
                method="POST", path="/v5/order/create", params=params
            )

            if not response or response.get("retCode") != 0:
                return None

            # Update rate limits
            self.rate_limits["order_rate_remaining"] -= 1

            # Create order object
            order_data = response["result"]
            return Order(
                id=order_data["orderId"],
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                status="NEW",
                time_in_force=time_in_force,
                reduce_only=reduce_only,
                created_time=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel existing order.

        Args:
            symbol: Trading pair symbol
            order_id: Order ID

        Returns:
            True if cancelled
        """
        try:
            response = self._send_request(
                method="POST",
                path="/v5/order/cancel",
                params={"symbol": symbol, "orderId": order_id},
            )

            return response and response.get("retCode") == 0

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get exchange status.

        Returns:
            Status information
        """
        return {
            "connected": self.is_connected(),
            "testnet": self.testnet,
            "rate_limits": self.rate_limits,
            "last_heartbeat": datetime.fromtimestamp(self._last_heartbeat).isoformat(),
        }

    def close(self) -> None:
        """Close exchange connection."""
        self._connected = False

    def _send_request(
        self, method: str, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Send API request.

        Args:
            method: HTTP method
            path: API endpoint path
            params: Request parameters

        Returns:
            Response data if successful
        """
        try:
            # Check rate limits
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded")
                return None

            # Build request
            url = f"{self.base_url}{path}"
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": str(int(time.time() * 1000)),
                "X-BAPI-RECV-WINDOW": "5000",
            }

            # Add signature
            params = params or {}
            signature_payload = (
                headers["X-BAPI-TIMESTAMP"]
                + self.api_key
                + headers["X-BAPI-RECV-WINDOW"]
            )
            if params:
                signature_payload += str(sorted(params.items()))

            signature = hmac.new(
                self.api_secret.encode(), signature_payload.encode(), hashlib.sha256
            ).hexdigest()
            headers["X-BAPI-SIGN"] = signature

            # Send request
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            else:
                response = requests.post(url, headers=headers, json=params)

            # Update rate limit tracking
            self._update_rate_limits(response.headers)

            # Handle response
            if response.status_code != 200:
                logger.error(f"Request failed: {response.status_code} {response.text}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def _check_rate_limit(self) -> bool:
        """Check if within rate limits.

        Returns:
            True if request can proceed
        """
        now = time.time()

        # Reset counters after 1 minute
        if now - self._last_request_reset >= 60:
            self._request_weight = 0
            self.rate_limits["order_rate_remaining"] = self.rate_limits[
                "orders_per_minute"
            ]
            self.rate_limits["weight_remaining"] = self.rate_limits["weight_per_minute"]
            self._last_request_reset = now

        return self._request_weight < self.rate_limits["weight_per_minute"]

    def _update_rate_limits(self, headers: Dict[str, str]) -> None:
        """Update rate limit tracking from response headers.

        Args:
            headers: Response headers
        """
        # Update request weight
        weight = int(headers.get("X-RateLimit-Used", "1"))
        self._request_weight += weight

        # Update remaining limits
        self.rate_limits["weight_remaining"] = int(
            headers.get("X-RateLimit-Remaining", "0")
        )

        # Update order rate limit if order endpoint
        if "order" in headers.get("X-RateLimit-Scope", ""):
            self.rate_limits["order_rate_remaining"] = int(
                headers.get("X-RateLimit-Order-Remaining", "0")
            )

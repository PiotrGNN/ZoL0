"""Exchange data types and enums."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


class OrderSide(Enum):
    """Order side enum."""

    BUY = "Buy"
    SELL = "Sell"


class OrderType(Enum):
    """Order type enum."""

    MARKET = "Market"
    LIMIT = "Limit"
    STOP_MARKET = "StopMarket"
    STOP_LIMIT = "StopLimit"
    TAKE_PROFIT = "TakeProfit"
    TAKE_PROFIT_LIMIT = "TakeProfitLimit"


class TimeInForce(Enum):
    """Time in force enum."""

    GOOD_TILL_CANCEL = "GTC"
    IMMEDIATE_OR_CANCEL = "IOC"
    FILL_OR_KILL = "FOK"
    POST_ONLY = "PostOnly"


class OrderStatus(Enum):
    """Order status enum."""

    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELED = "Canceled"
    REJECTED = "Rejected"
    EXPIRED = "Expired"


@dataclass
class Position:
    """Position details."""

    symbol: str
    size: float
    entry_price: float
    leverage: float
    liquidation_price: float
    unrealized_pnl: float
    margin: float
    entry_time: datetime
    current_price: Optional[float] = None
    strategy: Optional[str] = None


@dataclass
class Order:
    """Order details."""

    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float]
    status: str
    time_in_force: TimeInForce
    reduce_only: bool
    created_time: datetime
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    last_update_time: Optional[datetime] = None
    trigger_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: Optional[float] = None


@dataclass
class Trade:
    """Completed trade details."""

    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    side: str
    pnl: float
    fees: float
    id: Optional[str] = None
    tags: Optional[dict] = None


@dataclass
class TickerData:
    """Real-time ticker data."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime
    open_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    price_24h_change: float
    price_1h_change: float


@dataclass
class FundingInfo:
    """Funding rate information."""

    symbol: str
    funding_rate: float
    next_funding_time: datetime
    predicted_funding_rate: float
    next_funding_interval: int
    mark_price: float
    index_price: float


@dataclass
class AccountInfo:
    """Account information."""

    total_equity: float
    available_balance: float
    used_margin: float
    order_margin: float
    position_margin: float
    unrealized_pnl: float
    realized_pnl: float
    currency: str
    leverage: float
    maintenance_margin: float
    timestamp: datetime


@dataclass
class ExchangeInfo:
    """Exchange market information."""

    symbol: str
    base_asset: str
    quote_asset: str
    price_decimals: int
    qty_decimals: int
    min_order_size: float
    max_order_size: float
    min_price: float
    max_price: float
    tick_size: float
    step_size: float
    maker_fee: float
    taker_fee: float
    leverage_tiers: list
    trading: bool
    margin_trading: bool
    timestamp: datetime

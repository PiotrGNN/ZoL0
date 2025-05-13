"""Database management system."""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import sqlite3
import pandas as pd
from config import get_logger
from ..exchange.types import Position, Trade

logger = get_logger()


class DatabaseManager:
    """Manages database operations and persistence."""

    def __init__(self, db_path: str):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection = None

        # Initialize database
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database tables."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Create positions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    leverage REAL NOT NULL,
                    liquidation_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    margin REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    current_price REAL,
                    strategy TEXT,
                    metadata TEXT
                )
            """
            )

            # Create trades table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    size REAL NOT NULL,
                    side TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    fees REAL NOT NULL,
                    tags TEXT,
                    metadata TEXT
                )
            """
            )

            # Create daily stats table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date DATE PRIMARY KEY,
                    start_value REAL NOT NULL,
                    high_value REAL NOT NULL,
                    low_value REAL NOT NULL,
                    current_value REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    metadata TEXT
                )
            """
            )

            # Create performance metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    timestamp TIMESTAMP PRIMARY KEY,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    total_pnl REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    metadata TEXT
                )
            """
            )

            conn.commit()

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection.

        Returns:
            SQLite connection
        """
        if not self._connection:
            try:
                self._connection = sqlite3.connect(
                    self.db_path, detect_types=sqlite3.PARSE_DECLTYPES
                )
                self._connection.row_factory = sqlite3.Row
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                raise

        return self._connection

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def is_connected(self) -> bool:
        """Check if database is connected.

        Returns:
            True if connected
        """
        try:
            if not self._connection:
                return False

            # Test connection
            self._connection.cursor().execute("SELECT 1")
            return True

        except Exception:
            return False

    def update_position(self, symbol: str, position: Position) -> None:
        """Update or insert position.

        Args:
            symbol: Trading pair symbol
            position: Position details
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO positions (
                    symbol, size, entry_price, leverage,
                    liquidation_price, unrealized_pnl, margin,
                    entry_time, current_price, strategy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol,
                    position.size,
                    position.entry_price,
                    position.leverage,
                    position.liquidation_price,
                    position.unrealized_pnl,
                    position.margin,
                    position.entry_time,
                    position.current_price,
                    position.strategy,
                ),
            )

            conn.commit()

        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
            conn.rollback()

    def delete_position(self, symbol: str) -> None:
        """Delete position.

        Args:
            symbol: Trading pair symbol
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))

            conn.commit()

        except Exception as e:
            logger.error(f"Error deleting position for {symbol}: {e}")
            conn.rollback()

    def get_open_positions(self) -> List[Position]:
        """Get all open positions.

        Returns:
            List of positions
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM positions")
            rows = cursor.fetchall()

            return [
                Position(
                    symbol=row["symbol"],
                    size=row["size"],
                    entry_price=row["entry_price"],
                    leverage=row["leverage"],
                    liquidation_price=row["liquidation_price"],
                    unrealized_pnl=row["unrealized_pnl"],
                    margin=row["margin"],
                    entry_time=row["entry_time"],
                    current_price=row["current_price"],
                    strategy=row["strategy"],
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []

    def save_trade(self, trade: Trade) -> None:
        """Save completed trade.

        Args:
            trade: Trade details
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO trades (
                    symbol, strategy, entry_time, exit_time,
                    entry_price, exit_price, size, side,
                    pnl, fees, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.symbol,
                    trade.strategy,
                    trade.entry_time,
                    trade.exit_time,
                    trade.entry_price,
                    trade.exit_price,
                    trade.size,
                    trade.side,
                    trade.pnl,
                    trade.fees,
                    json.dumps(trade.tags) if trade.tags else None,
                ),
            )

            conn.commit()

        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            conn.rollback()

    def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Trade]:
        """Get historical trades.

        Args:
            symbol: Filter by symbol
            strategy: Filter by strategy
            start_time: Start time
            end_time: End time
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = "SELECT * FROM trades WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)

            if start_time:
                query += " AND entry_time >= ?"
                params.append(start_time)

            if end_time:
                query += " AND exit_time <= ?"
                params.append(end_time)

            query += " ORDER BY exit_time DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                Trade(
                    symbol=row["symbol"],
                    strategy=row["strategy"],
                    entry_time=row["entry_time"],
                    exit_time=row["exit_time"],
                    entry_price=row["entry_price"],
                    exit_price=row["exit_price"],
                    size=row["size"],
                    side=row["side"],
                    pnl=row["pnl"],
                    fees=row["fees"],
                    id=row["id"],
                    tags=json.loads(row["tags"]) if row["tags"] else None,
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

    def save_daily_stats(self, stats: Dict[str, float]) -> None:
        """Save daily performance statistics.

        Args:
            stats: Daily statistics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO daily_stats (
                    date, start_value, high_value, low_value,
                    current_value, realized_pnl, unrealized_pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().date(),
                    stats["start_value"],
                    stats["high_value"],
                    stats["low_value"],
                    stats["current_value"],
                    stats["realized_pnl"],
                    stats["unrealized_pnl"],
                ),
            )

            conn.commit()

        except Exception as e:
            logger.error(f"Error saving daily stats: {e}")
            conn.rollback()

    def get_daily_stats(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get daily performance statistics.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Daily statistics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = "SELECT * FROM daily_stats WHERE 1=1"
            params = []

            if start_date:
                query += " AND date >= ?"
                params.append(start_date.date())

            if end_date:
                query += " AND date <= ?"
                params.append(end_date.date())

            query += " ORDER BY date DESC LIMIT 1"

            cursor.execute(query, params)
            row = cursor.fetchone()

            if not row:
                return {}

            return {
                "start_value": row["start_value"],
                "high_value": row["high_value"],
                "low_value": row["low_value"],
                "current_value": row["current_value"],
                "realized_pnl": row["realized_pnl"],
                "unrealized_pnl": row["unrealized_pnl"],
            }

        except Exception as e:
            logger.error(f"Error getting daily stats: {e}")
            return {}

    def get_performance_metrics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get historical performance metrics.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with metrics
        """
        try:
            conn = self._get_connection()

            query = "SELECT * FROM performance_metrics WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp"

            return pd.read_sql_query(
                query, conn, params=params, parse_dates=["timestamp"]
            ).set_index("timestamp")

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return pd.DataFrame()

    def save_performance_metrics(
        self, metrics: Dict[str, Any], timestamp: Optional[datetime] = None
    ) -> None:
        """Save performance metrics.

        Args:
            metrics: Performance metrics
            timestamp: Metrics timestamp
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if not timestamp:
                timestamp = datetime.now()

            cursor.execute(
                """
                INSERT INTO performance_metrics (
                    timestamp, total_trades, winning_trades,
                    total_pnl, sharpe_ratio, max_drawdown,
                    win_rate, profit_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    metrics["total_trades"],
                    metrics["winning_trades"],
                    metrics["total_pnl"],
                    metrics["sharpe_ratio"],
                    metrics["max_drawdown"],
                    metrics["win_rate"],
                    metrics["profit_factor"],
                ),
            )

            conn.commit()

        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            conn.rollback()

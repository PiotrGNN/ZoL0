"""Database manager for handling all database operations."""

import os
import sqlite3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class DatabaseManager:
    """Handles all database operations for the trading system."""

    def __init__(self, db_path: str):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self.init_db()
        
    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def init_db(self) -> None:
        """Initialize database with schema."""
        try:
            # Read schema file
            schema_path = os.path.join(os.path.dirname(__file__), 'db_schema.sql')
            with open(schema_path, 'r') as f:
                schema = f.read()
            
            # Execute schema
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema)
            logging.info(f"Database initialized successfully at {self.db_path}")
        except Exception as e:
            logging.error(f"Failed to initialize database: {str(e)}")
            raise

    def record_trade(self, trade_data: Dict[str, Any]) -> int:
        """Record a new trade in the database."""
        query = """
        INSERT INTO trades (
            symbol, side, entry_price, quantity, stop_loss, 
            take_profit, status, strategy
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (
                    trade_data['symbol'],
                    trade_data['side'],
                    trade_data['entry_price'],
                    trade_data['quantity'],
                    trade_data.get('stop_loss'),
                    trade_data.get('take_profit'),
                    'OPEN',
                    trade_data.get('strategy')
                ))
                return cursor.lastrowid
        except Exception as e:
            logging.error(f"Failed to record trade: {str(e)}")
            raise

    def update_trade(self, trade_id: int, update_data: Dict[str, Any]) -> None:
        """Update an existing trade."""
        valid_fields = {'exit_price', 'status', 'pnl', 'closed_at'}
        update_fields = {k: v for k, v in update_data.items() if k in valid_fields}
        
        if not update_fields:
            return
            
        query = f"""
        UPDATE trades SET {', '.join(f'{k} = ?' for k in update_fields.keys())}
        WHERE id = ?
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(query, [*update_fields.values(), trade_id])
        except Exception as e:
            logging.error(f"Failed to update trade {trade_id}: {str(e)}")
            raise

    def save_market_data(self, data: List[Dict[str, Any]]) -> None:
        """Save market data to database."""
        query = """
        INSERT OR REPLACE INTO market_data (
            symbol, timestamp, open, high, low, close, volume
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(query, [
                    (d['symbol'], d['timestamp'], d['open'], d['high'],
                     d['low'], d['close'], d['volume'])
                    for d in data
                ])
        except Exception as e:
            logging.error(f"Failed to save market data: {str(e)}")
            raise

    def record_model_prediction(self, prediction_data: Dict[str, Any]) -> None:
        """Record a model prediction."""
        query = """
        INSERT INTO model_predictions (
            model_name, symbol, timestamp, prediction, confidence
        ) VALUES (?, ?, ?, ?, ?)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(query, (
                    prediction_data['model_name'],
                    prediction_data['symbol'],
                    prediction_data['timestamp'],
                    prediction_data['prediction'],
                    prediction_data.get('confidence')
                ))
        except Exception as e:
            logging.error(f"Failed to record model prediction: {str(e)}")
            raise

    def log_system_event(self, event_type: str, description: str, 
                        severity: str = 'INFO') -> None:
        """Log a system event."""
        query = """
        INSERT INTO system_events (event_type, description, severity)
        VALUES (?, ?, ?)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(query, (event_type, description, severity))
        except Exception as e:
            logging.error(f"Failed to log system event: {str(e)}")
            raise

    def update_portfolio(self, portfolio_data: Dict[str, Any]) -> None:
        """Update portfolio history."""
        query = """
        INSERT INTO portfolio_history (
            timestamp, total_equity, available_balance, 
            used_margin, currency
        ) VALUES (?, ?, ?, ?, ?)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(query, (
                    int(datetime.now().timestamp()),
                    portfolio_data['total_equity'],
                    portfolio_data['available_balance'],
                    portfolio_data['used_margin'],
                    portfolio_data.get('currency', 'USDT')
                ))
        except Exception as e:
            logging.error(f"Failed to update portfolio: {str(e)}")
            raise

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all open trades."""
        query = "SELECT * FROM trades WHERE status = 'OPEN'"
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Failed to get open trades: {str(e)}")
            raise

    def get_trading_history(self, symbol: Optional[str] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get trading history."""
        query = """
        SELECT * FROM trades WHERE status = 'CLOSED'
        {}
        ORDER BY closed_at DESC LIMIT ?
        """.format("AND symbol = ?" if symbol else "")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    query, 
                    (symbol, limit) if symbol else (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Failed to get trading history: {str(e)}")
            raise
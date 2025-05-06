"""
Tests for database management functionality
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from data.db_manager import DatabaseManager
from data.models import Base, Trade, Strategy, StrategyMetrics

@pytest.fixture
def db():
    """Create test database"""
    return DatabaseManager("sqlite:///:memory:")

@pytest.fixture
def sample_strategy(db):
    """Create a sample strategy"""
    strategy_data = {
        "name": "Test Strategy",
        "description": "Strategy for testing",
        "parameters": '{"param1": 1, "param2": 2}'
    }
    with db.get_session() as session:
        strategy = Strategy(**strategy_data)
        session.add(strategy)
        session.commit()
        # Make a copy of attributes to avoid detached instance issues
        strategy_copy = Strategy(
            id=strategy.id,
            name=strategy.name,
            description=strategy.description,
            parameters=strategy.parameters
        )
    return strategy_copy

@pytest.mark.unit
class TestDatabaseManager:
    def test_init_db(self, db):
        """Test database initialization"""
        # Verify tables are created using inspect()
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert "trades" in tables
        assert "strategies" in tables
        assert "strategy_metrics" in tables
        assert "market_data" in tables
        assert "model_metadata" in tables

    def test_add_strategy(self, db):
        """Test adding a strategy"""
        strategy_data = {
            "name": "Test Strategy",
            "description": "Test description",
            "parameters": '{"param1": "value1"}'
        }
        strategy = db.add_strategy(strategy_data)
        assert strategy.id is not None
        assert strategy.name == strategy_data["name"]
        assert strategy.description == strategy_data["description"]

    def test_add_trade(self, db, sample_strategy):
        """Test adding a trade"""
        trade_data = {
            "symbol": "BTC/USDT",
            "direction": "LONG",
            "entry_price": 50000.0,
            "quantity": 0.1,
            "strategy_id": sample_strategy.id
        }
        trade = db.add_trade(trade_data)
        assert trade.id is not None
        assert trade.symbol == trade_data["symbol"]
        assert trade.entry_price == trade_data["entry_price"]
        assert trade.strategy_id == sample_strategy.id

    def test_update_trade(self, db, sample_strategy):
        """Test updating a trade"""
        # Create initial trade
        trade_data = {
            "symbol": "BTC/USDT",
            "direction": "LONG",
            "entry_price": 50000.0,
            "quantity": 0.1,
            "strategy_id": sample_strategy.id
        }
        trade = db.add_trade(trade_data)
        
        # Update trade
        update_data = {
            "exit_price": 52000.0,
            "status": "CLOSED",
            "profit_loss": 200.0
        }
        updated_trade = db.update_trade(trade.id, update_data)
        assert updated_trade.exit_price == update_data["exit_price"]
        assert updated_trade.status == update_data["status"]
        assert updated_trade.profit_loss == update_data["profit_loss"]

    def test_add_strategy_metrics(self, db, sample_strategy):
        """Test adding strategy metrics"""
        metrics_data = {
            "strategy_id": sample_strategy.id,
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.15,
            "total_trades": 100,
            "profitable_trades": 65
        }
        metrics = db.add_strategy_metrics(metrics_data)
        assert metrics.id is not None
        assert metrics.strategy_id == sample_strategy.id
        assert metrics.win_rate == metrics_data["win_rate"]
        assert metrics.profit_factor == metrics_data["profit_factor"]

    def test_get_trades(self, db, sample_strategy):
        """Test getting trades with filters"""
        # Add multiple trades
        trades_data = [
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 50000.0,
                "quantity": 0.1,
                "strategy_id": sample_strategy.id,
                "status": "OPEN"
            },
            {
                "symbol": "ETH/USDT",
                "direction": "SHORT",
                "entry_price": 3000.0,
                "quantity": 1.0,
                "strategy_id": sample_strategy.id,
                "status": "CLOSED"
            }
        ]
        for trade_data in trades_data:
            db.add_trade(trade_data)
            
        # Test filtering
        open_trades = db.get_trades(status="OPEN")
        assert len(open_trades) == 1
        assert open_trades[0].symbol == "BTC/USDT"
        
        closed_trades = db.get_trades(status="CLOSED")
        assert len(closed_trades) == 1
        assert closed_trades[0].symbol == "ETH/USDT"
        
        strategy_trades = db.get_trades(strategy_id=sample_strategy.id)
        assert len(strategy_trades) == 2

    @pytest.mark.parametrize("invalid_id", [-1, 9999])
    def test_invalid_trade_update(self, db, invalid_id):
        """Test updating non-existent trade"""
        update_data = {"exit_price": 52000.0}
        result = db.update_trade(invalid_id, update_data)
        assert result is None
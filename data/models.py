"""
Database models for the trading system
"""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from enum import Enum

Base = declarative_base()

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"

class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    PENDING = "PENDING"

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(SQLEnum(TradeDirection), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, nullable=False)
    status = Column(SQLEnum(TradeStatus), default=TradeStatus.OPEN)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)
    profit_loss = Column(Float)
    commission = Column(Float)
    strategy_id = Column(Integer, ForeignKey('strategies.id'))
    
    strategy = relationship("Strategy", back_populates="trades")

class Strategy(Base):
    __tablename__ = 'strategies'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    parameters = Column(String(1000))  # JSON string of parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    trades = relationship("Trade", back_populates="strategy")
    metrics = relationship("StrategyMetrics", back_populates="strategy")

class StrategyMetrics(Base):
    __tablename__ = 'strategy_metrics'

    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    total_trades = Column(Integer)
    profitable_trades = Column(Integer)
    
    strategy = relationship("Strategy", back_populates="metrics")

class MarketData(Base):
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float)

class ModelMetadata(Base):
    __tablename__ = 'model_metadata'

    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    parameters = Column(String(1000))  # JSON string of parameters
    training_date = Column(DateTime, default=datetime.utcnow)
    metrics = Column(String(500))  # JSON string of metrics
    file_path = Column(String(200))  # Path to saved model file
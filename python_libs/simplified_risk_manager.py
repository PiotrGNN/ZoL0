"""
simplified_risk_manager.py
-----------------------
Module for managing trading risk.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class SimplifiedRiskManager:
    def __init__(self, max_risk: float = 0.05, max_position_size: float = 0.2, max_drawdown: float = 0.1):
        """Initialize risk manager with risk parameters.
        
        Args:
            max_risk (float): Maximum risk per trade (default: 5%)
            max_position_size (float): Maximum position size as percentage of portfolio (default: 20%)
            max_drawdown (float): Maximum allowed drawdown before stopping trading (default: 10%)
        """
        self.max_risk = max_risk
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.current_drawdown = 0.0
        self.peak_value = None
        self.positions = {}
        self.risk_levels = {
            "low": 0.03,
            "medium": 0.05,
            "high": 0.07
        }
        self.last_update = datetime.now()
        logging.info(f"Initialized RiskManager with max_risk={max_risk}, max_position_size={max_position_size}, max_drawdown={max_drawdown}")
    
    def check_risk_level(self, position_size: float, stop_loss: float, entry_price: float) -> Dict[str, Any]:
        """Check if a trade meets risk parameters."""
        try:
            risk_amount = abs(entry_price - stop_loss) * position_size
            risk_percentage = risk_amount / entry_price
            
            if risk_percentage > self.max_risk:
                return {
                    "allowed": False,
                    "reason": f"Risk too high: {risk_percentage:.2%} > {self.max_risk:.2%}",
                    "risk_amount": risk_amount,
                    "risk_percentage": risk_percentage
                }
            
            if position_size > self.max_position_size:
                return {
                    "allowed": False,
                    "reason": f"Position size too large: {position_size:.2%} > {self.max_position_size:.2%}",
                    "risk_amount": risk_amount,
                    "risk_percentage": risk_percentage
                }
            
            return {
                "allowed": True,
                "risk_amount": risk_amount,
                "risk_percentage": risk_percentage
            }
        except Exception as e:
            logging.error(f"Error in check_risk_level: {e}")
            return {
                "allowed": False,
                "reason": f"Error checking risk level: {str(e)}",
                "risk_amount": 0,
                "risk_percentage": 0
            }
    
    def update_drawdown(self, current_value: float) -> Dict[str, Any]:
        """Update and check drawdown levels."""
        try:
            if self.peak_value is None:
                self.peak_value = current_value
            
            if current_value > self.peak_value:
                self.peak_value = current_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_value - current_value) / self.peak_value
            
            self.last_update = datetime.now()
            
            return {
                "current_drawdown": self.current_drawdown,
                "peak_value": self.peak_value,
                "max_drawdown": self.max_drawdown,
                "trading_allowed": self.current_drawdown < self.max_drawdown
            }
        except Exception as e:
            logging.error(f"Error in update_drawdown: {e}")
            return {
                "error": str(e),
                "trading_allowed": False
            }
    
    def add_position(self, symbol: str, position_data: Dict[str, Any]) -> bool:
        """Add a new position to track."""
        try:
            self.positions[symbol] = {
                **position_data,
                "timestamp": datetime.now()
            }
            return True
        except Exception as e:
            logging.error(f"Error adding position: {e}")
            return False
    
    def remove_position(self, symbol: str) -> bool:
        """Remove a tracked position."""
        try:
            if symbol in self.positions:
                del self.positions[symbol]
            return True
        except Exception as e:
            logging.error(f"Error removing position: {e}")
            return False
    
    def get_position_risk(self, symbol: str) -> Dict[str, Any]:
        """Get risk information for a specific position."""
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                return {
                    "symbol": symbol,
                    "risk_amount": position.get("risk_amount", 0),
                    "risk_percentage": position.get("risk_percentage", 0),
                    "entry_time": position.get("timestamp", datetime.now()).isoformat()
                }
            return {}
        except Exception as e:
            logging.error(f"Error getting position risk: {e}")
            return {"error": str(e)}
    
    def get_total_risk(self) -> Dict[str, Any]:
        """Get total risk across all positions."""
        try:
            total_risk = sum(pos.get("risk_percentage", 0) for pos in self.positions.values())
            return {
                "total_risk": total_risk,
                "max_risk": self.max_risk,
                "current_drawdown": self.current_drawdown,
                "positions_count": len(self.positions),
                "last_update": self.last_update.isoformat()
            }
        except Exception as e:
            logging.error(f"Error calculating total risk: {e}")
            return {"error": str(e)}
    
    def set_risk_level(self, level: str) -> bool:
        """Set risk level (low/medium/high)."""
        try:
            if level in self.risk_levels:
                self.max_risk = self.risk_levels[level]
                logging.info(f"Risk level set to {level} ({self.max_risk:.2%})")
                return True
            return False
        except Exception as e:
            logging.error(f"Error setting risk level: {e}")
            return False
            
    def get_risk_limits(self) -> Dict[str, float]:
        """Get current risk limits."""
        return {
            "max_risk": self.max_risk,
            "max_position_size": self.max_position_size,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown
        }

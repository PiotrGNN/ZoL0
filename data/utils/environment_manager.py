"""
environment_manager.py
--------------------
Manages environment state and transitions between environments with validation and safety checks.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnvironmentState:
    """Current environment state information"""
    is_testnet: bool
    last_switch_time: float
    active_orders: int = 0
    active_positions: int = 0
    initialized: bool = False
    switch_in_progress: bool = False
    last_switch_status: Optional[Dict[str, Any]] = None

class EnvironmentManager:
    def __init__(self, api_client=None):
        """
        Initialize environment manager.
        
        Args:
            api_client: Optional API client instance to use for environment checks
        """
        self.api_client = api_client
        self.state = EnvironmentState(
            is_testnet=True,  # Default to testnet for safety
            last_switch_time=0.0
        )
        self._load_environment()

    def _load_environment(self):
        """Load environment configuration from environment variables"""
        # Check for explicit testnet flag
        if os.getenv("BYBIT_TESTNET", "").lower() == "true":
            self.state.is_testnet = True
            logger.info("Environment set to testnet via BYBIT_TESTNET")
            return

        # Check for production environment confirmation
        if os.getenv("BYBIT_PRODUCTION_CONFIRMED", "").lower() == "true" and \
           os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true":
            self.state.is_testnet = False
            logger.warning("Production environment enabled via environment variables")
        else:
            self.state.is_testnet = True
            logger.info("Defaulting to testnet environment (production not confirmed)")

    async def validate_switch(self, target_is_testnet: bool) -> Dict[str, Any]:
        """
        Validate if environment switch is possible.
        
        Args:
            target_is_testnet: Whether switching to testnet environment
        
        Returns:
            Dict with validation results and any blockers
        """
        if self.state.switch_in_progress:
            return {
                "can_switch": False,
                "reason": "Environment switch already in progress"
            }

        # Don't allow rapid switching
        if (datetime.now().timestamp() - self.state.last_switch_time) < 60:
            return {
                "can_switch": False,
                "reason": "Please wait at least 60 seconds between environment switches",
                "remaining_time": 60 - (datetime.now().timestamp() - self.state.last_switch_time)
            }

        # Additional validation when switching to production
        if not target_is_testnet:
            # Verify production environment is properly configured
            if not os.getenv("BYBIT_PRODUCTION_CONFIRMED", "").lower() == "true" or \
               not os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true":
                return {
                    "can_switch": False,
                    "reason": "Production environment not properly confirmed. Set BYBIT_PRODUCTION_CONFIRMED=true and BYBIT_PRODUCTION_ENABLED=true"
                }

            # Check API credentials
            api_key = os.getenv("BYBIT_API_KEY", "")
            api_secret = os.getenv("BYBIT_API_SECRET", "")
            if not api_key or not api_secret or len(api_key) < 10 or len(api_secret) < 10:
                return {
                    "can_switch": False,
                    "reason": "Invalid API credentials for production environment"
                }

        return {
            "can_switch": True
        }

    async def switch_environment(self, target_is_testnet: bool) -> Dict[str, Any]:
        """
        Switch between testnet and production environments with validation.
        
        Args:
            target_is_testnet: Whether switching to testnet environment
        
        Returns:
            Dict with switch results and new environment state
        """
        # Validate switch first
        validation = await self.validate_switch(target_is_testnet)
        if not validation["can_switch"]:
            return {
                "success": False,
                "error": validation["reason"]
            }

        try:
            self.state.switch_in_progress = True
            
            # Switch environment in API client if provided
            if self.api_client:
                api_result = await self.api_client.switch_environment(target_is_testnet)
                if not api_result["success"]:
                    raise Exception(f"API client environment switch failed: {api_result.get('error')}")

            # Update environment state
            self.state.is_testnet = target_is_testnet
            self.state.last_switch_time = datetime.now().timestamp()
            self.state.last_switch_status = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "environment": "testnet" if target_is_testnet else "production"
            }

            logger.info(f"Successfully switched to {'testnet' if target_is_testnet else 'production'} environment")
            
            return {
                "success": True,
                "environment": "testnet" if target_is_testnet else "production",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Environment switch failed: {str(e)}"
            logger.error(error_msg)
            self.state.last_switch_status = {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
            return {
                "success": False,
                "error": error_msg
            }

        finally:
            self.state.switch_in_progress = False

    def get_environment_status(self) -> Dict[str, Any]:
        """Get current environment status"""
        return {
            "environment": "testnet" if self.state.is_testnet else "production",
            "initialized": self.state.initialized,
            "switch_in_progress": self.state.switch_in_progress,
            "last_switch": self.state.last_switch_status,
            "active_orders": self.state.active_orders,
            "active_positions": self.state.active_positions,
            "production_enabled": os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true",
            "production_confirmed": os.getenv("BYBIT_PRODUCTION_CONFIRMED", "").lower() == "true"
        }
"""
api_client.py
--------------
Centralized API client with improved error handling, rate limiting, and environment management.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

import requests
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def rate_limit(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._handle_rate_limit()
        return func(self, *args, **kwargs)
    return wrapper

class ApiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        use_testnet: Optional[bool] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize API client with environment-aware configuration.
        
        Args:
            api_key: API key for authentication
            api_secret: API secret for request signing
            use_testnet: Whether to use testnet environment
            base_url: Base URL for API endpoints
        """
        # Environment configuration
        self.use_testnet = self._determine_environment(use_testnet)
        self.api_key = api_key or os.getenv("BYBIT_API_KEY")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET")
        self.base_url = base_url or (
            "https://api-testnet.bybit.com" if self.use_testnet else "https://api.bybit.com"
        )

        # Rate limiting configuration
        self.last_request_time = 0
        self.rate_limit_exceeded = False
        self.rate_limit_reset_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        self.rate_limit_window = 60.0  # 1 minute window
        self.max_requests_per_window = 50 if self.use_testnet else 20

        # Request tracking
        self.request_count = 0
        self.request_timestamps = []

        # Connection state
        self._initialized = False
        self._last_server_time = 0
        
        logger.info(f"ApiClient initialized - Environment: {'testnet' if self.use_testnet else 'production'}")
        
        # Validate production environment
        if not self.use_testnet:
            self._validate_production_config()

    def _determine_environment(self, use_testnet: Optional[bool]) -> bool:
        """
        Determine which environment to use based on configuration hierarchy.
        """
        if use_testnet is not None:
            return use_testnet
        
        # Check environment variables in order of precedence
        if os.getenv("BYBIT_TESTNET", "").lower() == "true":
            return True
        if os.getenv("BYBIT_PRODUCTION_CONFIRMED", "").lower() == "true" and \
           os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true":
            return False
        
        # Default to testnet for safety
        logger.warning("No environment explicitly configured - defaulting to testnet for safety")
        return True

    def _validate_production_config(self):
        """
        Validate production environment configuration.
        """
        if not self.api_key or not self.api_secret or \
           len(self.api_key) < 10 or len(self.api_secret) < 10:
            raise ValueError("Invalid API credentials for production environment")

        if not os.getenv("BYBIT_PRODUCTION_CONFIRMED", "").lower() == "true" or \
           not os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true":
            logger.critical("\n" + "!"*80)
            logger.critical("PRODUCTION API ACCESS REQUIRES EXPLICIT CONFIRMATION")
            logger.critical("Set both environment variables:")
            logger.critical("1. BYBIT_PRODUCTION_CONFIRMED=true")
            logger.critical("2. BYBIT_PRODUCTION_ENABLED=true")
            logger.critical("!"*80 + "\n")
            raise ValueError("Production API access not explicitly confirmed")

    def _handle_rate_limit(self):
        """
        Handle rate limiting with exponential backoff.
        """
        current_time = time.time()
        
        # Clean up old request timestamps
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts < self.rate_limit_window]
        
        # Check if we're rate limited
        if self.rate_limit_exceeded:
            if current_time < self.rate_limit_reset_time:
                raise Exception(f"Rate limit exceeded. Reset in {self.rate_limit_reset_time - current_time:.1f}s")
            self.rate_limit_exceeded = False
            self.request_timestamps = []
        
        # Check request count in current window
        if len(self.request_timestamps) >= self.max_requests_per_window:
            self.rate_limit_exceeded = True
            self.rate_limit_reset_time = current_time + self.rate_limit_window
            raise Exception(f"Rate limit exceeded. Reset in {self.rate_limit_window:.1f}s")
        
        # Enforce minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = current_time
        self.request_timestamps.append(current_time)

    def _sign_request(self, method: str, endpoint: str, params: Dict = None) -> Dict[str, str]:
        """
        Sign a request using HMAC authentication.
        """
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        params = params or {}
        params_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())]) if params else ''
        
        sign_str = f"{timestamp}{self.api_key}{recv_window}{params_str}"
        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            sign_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }

    @rate_limit
    def request(self, method: str, endpoint: str, params: Dict = None, authenticate: bool = True) -> Dict[str, Any]:
        """
        Make an HTTP request to the API with rate limiting and authentication.
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._sign_request(method, endpoint, params) if authenticate else {}
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params if method == "GET" else None,
                json=params if method != "GET" else None,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Handle common error patterns
            if "retCode" in data:
                code = data["retCode"]
                if code == 0:
                    return {"success": True, "data": data.get("result", {})}
                elif code in [10006, 10018]:  # Rate limit error codes
                    self.rate_limit_exceeded = True
                    self.rate_limit_reset_time = time.time() + 60
                    raise Exception(f"Rate limit exceeded: {data.get('retMsg', 'Unknown error')}")
                else:
                    raise Exception(f"API error {code}: {data.get('retMsg', 'Unknown error')}")
            
            return {"success": True, "data": data}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"success": False, "error": str(e)}

    @rate_limit
    def get_server_time(self) -> Dict[str, Any]:
        """
        Get current server time.
        """
        return self.request("GET", "/v5/market/time", authenticate=False)

    @rate_limit
    def switch_environment(self, use_testnet: bool) -> Dict[str, Any]:
        """
        Switch between testnet and production environments.
        """
        if not self.use_testnet and use_testnet:
            logger.info("Switching from production to testnet")
            self.use_testnet = True
            self.base_url = "https://api-testnet.bybit.com"
            return {"success": True, "environment": "testnet"}
            
        if self.use_testnet and not use_testnet:
            logger.warning("\n" + "!"*80)
            logger.warning("ATTEMPTING TO SWITCH TO PRODUCTION ENVIRONMENT")
            logger.warning("This will affect real funds!")
            logger.warning("!"*80 + "\n")
            
            try:
                self._validate_production_config()
                self.use_testnet = False
                self.base_url = "https://api.bybit.com"
                return {"success": True, "environment": "production"}
            except ValueError as e:
                return {"success": False, "error": str(e)}
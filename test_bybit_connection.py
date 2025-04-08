
"""
Script to test Bybit API connection.
This checks if the API connection is working correctly.
"""

import os
import sys
import logging
import time
import json
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "api_test.log")),
        logging.StreamHandler()
    ]
)

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

def test_bybit_connection():
    """Test connection to Bybit API."""
    try:
        # Import after path is set
        from data.execution.bybit_connector import BybitConnector
        
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        use_testnet = os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"
        
        if not api_key or not api_secret:
            logging.error("API key or secret not found in .env file")
            return False
            
        # Initialize Bybit connector
        logging.info(f"Initializing Bybit connector (Testnet: {use_testnet})")
        connector = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet
        )
        
        # Test server time
        logging.info("Testing server time...")
        server_time = connector.get_server_time()
        logging.info(f"Server time: {server_time}")
        
        # Test account balance
        logging.info("Testing account balance...")
        try:
            balance = connector.get_account_balance()
            logging.info(f"Account balance: {json.dumps(balance, indent=2)}")
        except Exception as e:
            logging.warning(f"Could not get account balance: {e}")
            
        # Test market data
        logging.info("Testing market data...")
        try:
            klines = connector.get_klines(symbol="BTCUSDT", interval="15", limit=5)
            logging.info(f"BTCUSDT klines: {json.dumps(klines[:2], indent=2)}")
        except Exception as e:
            logging.warning(f"Could not get market data: {e}")
            
        logging.info("Bybit API connection test completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error testing Bybit connection: {e}")
        return False

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    logging.info("Starting Bybit API connection test")
    success = test_bybit_connection()
    if success:
        logging.info("✅ Bybit API connection test PASSED")
        print("\n✅ Bybit API connection test PASSED")
    else:
        logging.error("❌ Bybit API connection test FAILED")
        print("\n❌ Bybit API connection test FAILED")
        print("Check logs/api_test.log for details")

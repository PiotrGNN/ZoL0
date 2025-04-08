
import os
import json
import time
import logging
import sys
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/api_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
load_dotenv()

def test_bybit_connection():
    """Test connection to Bybit API."""
    try:
        # Import after path is set
        from data.execution.bybit_connector import BybitConnector
        
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        
        # Check for both environment variables
        use_testnet = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
        if os.getenv("BYBIT_USE_TESTNET", "").lower() == "false":
            use_testnet = False
        
        if not api_key or not api_secret:
            logging.error("API key or secret not found in .env file")
            print("ERROR: API key or secret not found in .env file")
            print("Please check your .env file and ensure BYBIT_API_KEY and BYBIT_API_SECRET are set")
            return False
            
        # Initialize Bybit connector
        logging.info(f"Initializing Bybit connector (Testnet: {use_testnet})")
        print(f"\nConnecting to Bybit API {'TESTNET' if use_testnet else 'PRODUCTION'}")
        print(f"API Key: {api_key[:4]}{'*'*(len(api_key)-8)}{api_key[-4:]}")
        
        connector = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet
        )
        
        # Test server time
        logging.info("Testing server time...")
        print("\nTesting server time...")
        server_time = connector.get_server_time()
        logging.info(f"Server time: {server_time}")
        print(f"Server time: {json.dumps(server_time, indent=2)}")
        
        # Test account balance
        logging.info("Testing account balance...")
        print("\nTesting account balance...")
        try:
            balance = connector.get_account_balance()
            logging.info(f"Account balance: {json.dumps(balance, indent=2)}")
            print(f"Account balance: {json.dumps(balance, indent=2)}")
        except Exception as e:
            logging.warning(f"Could not get account balance: {e}")
            print(f"WARNING: Could not get account balance: {e}")
            
        # Test market data
        logging.info("Testing market data...")
        print("\nTesting market data...")
        try:
            klines = connector.get_klines(symbol="BTCUSDT", interval="15", limit=5)
            logging.info(f"Successfully retrieved {len(klines)} klines")
            print(f"Successfully retrieved {len(klines)} klines")
            print(f"First kline: {json.dumps(klines[0], indent=2)}")
        except Exception as e:
            logging.warning(f"Could not get klines: {e}")
            print(f"WARNING: Could not get klines: {e}")
        
        print("\nConnection test completed successfully")
        return True
    except ImportError as ie:
        logging.error(f"Import error: {ie}")
        print(f"ERROR: Import error: {ie}")
        print("Please check if all required modules are installed:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        logging.error(f"Error during connection test: {e}")
        print(f"ERROR: Connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("\n=== Bybit API Connection Test ===\n")
    print("This script will test connectivity to Bybit API using your credentials")
    print("Testing connection...")
    result = test_bybit_connection()
    print(f"\nTest {'PASSED' if result else 'FAILED'}")
    print("\nCheck logs/api_test.log for detailed information")

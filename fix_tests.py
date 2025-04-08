
import os
import sys
import logging
import importlib
import subprocess

# Configure logging
logging.basicConfig(
    filename="fix_tests_log.txt",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_environment():
    """Set up the environment for testing."""
    # Add project root to path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root_dir)
    logging.info(f"Added {root_dir} to sys.path")
    
    # Create necessary directories
    for directory in ['logs', 'data/cache']:
        dir_path = os.path.join(root_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")
    
    # Set environment variable for testing
    os.environ["TESTING"] = "true"
    os.environ["BYBIT_TESTNET"] = "true"
    os.environ["USE_SIMULATED_DATA"] = "true"
    logging.info("Set testing environment variables")

def install_test_dependencies():
    """Install dependencies required for testing."""
    test_dependencies = ["pytest"]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + test_dependencies)
        logging.info(f"Installed test dependencies: {test_dependencies}")
    except Exception as e:
        logging.error(f"Failed to install test dependencies: {e}")
        print(f"Error installing test dependencies: {e}")

def fix_test_imports():
    """Fix import issues in test files."""
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tests")
    if not os.path.exists(test_dir):
        logging.warning(f"Test directory not found: {test_dir}")
        return

    for test_file in os.listdir(test_dir):
        if test_file.endswith(".py") and test_file.startswith("test_"):
            test_file_path = os.path.join(test_dir, test_file)
            logging.info(f"Checking test file: {test_file}")
            
            try:
                # Try to import the module to check for issues
                module_name = f"data.tests.{test_file[:-3]}"
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    logging.warning(f"Could not find module: {module_name}")
                else:
                    try:
                        module = importlib.import_module(module_name)
                        logging.info(f"Successfully imported {module_name}")
                    except ImportError as ie:
                        logging.error(f"Import error in {test_file}: {ie}")
                        print(f"Import error in {test_file}: {ie}")
            except Exception as e:
                logging.error(f"Error checking {test_file}: {e}")

def run_tests():
    """Run the tests to verify they work."""
    try:
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tests")
        if not os.path.exists(test_dir):
            logging.warning(f"Test directory not found: {test_dir}")
            return False
            
        print("\nRunning tests...")
        result = subprocess.run([sys.executable, "-m", "pytest", "-xvs", test_dir], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("Tests passed successfully")
            print("\nTests passed successfully!")
            return True
        else:
            logging.error(f"Tests failed with errors: {result.stderr}")
            print(f"\nTests failed. See error output below:")
            print(result.stderr)
            return False
    except Exception as e:
        logging.error(f"Error running tests: {e}")
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    print("Starting test fixing process...")
    logging.info("Starting test fixing process")
    
    setup_environment()
    install_test_dependencies()
    fix_test_imports()
    
    print("\nTest environment setup completed.")
    print("Would you like to run the tests now? (y/n)")
    choice = input().strip().lower()
    
    if choice in ['y', 'yes']:
        run_tests()
    else:
        print("\nSkipping test execution.")
        print("You can run tests later with: python -m pytest data/tests")
    
    logging.info("Test fixing process completed")
    print("\nTest fixing process completed")
    print("Check 'fix_tests_log.txt' for details")

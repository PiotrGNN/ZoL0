
import os
import sys
import logging
import subprocess
import importlib
import pkg_resources

# Setup logging
logging.basicConfig(
    filename='fix_imports_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_and_install_dependencies():
    """Check for required dependencies and install if missing."""
    required_packages = [
        'flask', 'requests', 'pandas', 'numpy', 'pybit', 
        'python-dotenv', 'pydantic', 'apscheduler'
    ]
    
    logging.info("Checking required packages...")
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = [pkg for pkg in required_packages if pkg.lower() not in installed_packages]
    
    if missing_packages:
        logging.info(f"Installing missing packages: {', '.join(missing_packages)}")
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            logging.info("Successfully installed missing packages.")
            print("Successfully installed missing packages.")
        except Exception as e:
            logging.error(f"Failed to install packages: {e}")
            print(f"Error installing packages: {e}")
            print("Try manually installing with:")
            print(f"pip install {' '.join(missing_packages)}")
    else:
        logging.info("All required packages already installed.")

def fix_imports():
    """Fix import issues in the project."""
    # Add project directory to the Python path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)
    logging.info(f"Added {base_dir} to sys.path")
    
    # Create necessary directories
    for directory in ['logs', 'data/cache']:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")
    
    # Check for common import issues
    try:
        # Test importing key modules
        import_tests = [
            "from data.execution.bybit_connector import BybitConnector",
            "from data.utils.cache_manager import get_cached_data, store_cached_data",
            "from dotenv import load_dotenv",
            "import flask"
        ]
        
        for test in import_tests:
            try:
                exec(test)
                logging.info(f"Successfully imported: {test}")
            except ImportError as e:
                logging.error(f"Import error with '{test}': {e}")
                print(f"Import error with '{test}': {e}")
                print("This might require manual fixing.")

        logging.info("All key imports checked.")
    except Exception as e:
        logging.error(f"Error in import testing: {e}")
        print(f"Error checking imports: {e}")

    # Fix specific import paths if needed
    print("\nNOTE: If you still experience import errors when running the application,")
    print("make sure to run the application from the project's root directory.")
    print("You can also try: python -m main")

if __name__ == "__main__":
    logging.info("Starting import fix process")
    check_and_install_dependencies()
    fix_imports()
    logging.info("Import fix process completed")
    
    print("\nImport fix process completed")
    print("Check 'fix_imports_log.txt' for details")
    print("\nNext steps:")
    print("1. Run the application: python main.py")
    print("2. If issues persist, check logs in 'logs' directory")

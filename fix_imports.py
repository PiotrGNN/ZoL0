import os
import sys
import logging

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fix_imports_log.txt"),
        logging.StreamHandler()
    ]
)

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
            "import flask",

    # Konwersje ścieżek Unix na Windows
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Konwersja ścieżek Unix na Windows
                    unix_paths = ['logs/', 'data/', 'ai_models/']
                    windows_conversion = False
                    
                    for unix_path in unix_paths:
                        if unix_path in content:
                            content = content.replace(unix_path, unix_path.replace('/', '\\'))
                            windows_conversion = True
                    
                    if windows_conversion:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        logging.info(f"Przekonwertowano ścieżki Unix na Windows w pliku: {file_path}")
                except Exception as e:
                    logging.error(f"Błąd podczas konwersji ścieżek w pliku {file_path}: {e}")

            "import matplotlib",
            "import pandas",
            "import numpy",
            "import requests"
        ]

        missing_modules = []

        for test in import_tests:
            try:
                exec(test)
                logging.info(f"Successfully imported: {test}")
            except ImportError as e:
                missing_modules.append((test.split("import ")[1].split(",")[0].strip(), str(e)))
                logging.error(f"Import error with '{test}': {e}")

        if missing_modules:
            print("\n==== MISSING MODULES ====")
            print("Run the following commands to install missing dependencies:")
            for module, error in missing_modules:
                print(f"pip install {module}")
            print("===========================\n")
        else:
            print("\n✅ All key imports checked successfully!")

        logging.info("All key imports checked.")
    except Exception as e:
        logging.error(f"Error in import testing: {e}")
        print(f"Error checking imports: {e}")

if __name__ == "__main__":
    print("Fixing imports...")
    fix_imports()
    print("Done! Check fix_imports_log.txt for details.")
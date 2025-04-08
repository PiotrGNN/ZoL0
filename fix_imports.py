
"""
Script to fix common import issues in the project.
This ensures all packages are properly available in the Python path.
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fix_imports_log.txt"),
        logging.StreamHandler()
    ]
)

def check_and_install_dependencies():
    """Check if all required packages are installed and install them if needed."""
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().splitlines()
        
        # Filter out comments and empty lines
        requirements = [r for r in requirements if r and not r.startswith("#")]
        
        for req in requirements:
            try:
                package_name = req.split("==")[0].split(">=")[0].strip()
                __import__(package_name)
                logging.info(f"Package {package_name} is already installed.")
            except ImportError:
                logging.warning(f"Package {package_name} is missing. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", req])
                logging.info(f"Successfully installed {req}")
    except Exception as e:
        logging.error(f"Error checking dependencies: {e}")

def fix_imports():
    """Fix import issues by adding project directories to Python path."""
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.append(project_root)
            logging.info(f"Added {project_root} to Python path")
        
        # Add specific subdirectories to path
        for dir_name in ["data", "ai_models", "templates"]:
            dir_path = os.path.join(project_root, dir_name)
            if os.path.isdir(dir_path) and dir_path not in sys.path:
                sys.path.append(dir_path)
                logging.info(f"Added {dir_path} to Python path")
        
        # Create __init__.py files where needed
        dirs_to_check = [
            os.path.join(project_root, "data"),
            os.path.join(project_root, "ai_models"),
            os.path.join(project_root, "templates"),
        ]
        
        for dir_path in dirs_to_check:
            if not os.path.isdir(dir_path):
                continue
                
            init_file = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write('"""Package initialization."""\n')
                logging.info(f"Created {init_file}")
                
            # Check subdirectories too
            for subdir in os.listdir(dir_path):
                subdir_path = os.path.join(dir_path, subdir)
                if os.path.isdir(subdir_path):
                    init_file = os.path.join(subdir_path, "__init__.py")
                    if not os.path.exists(init_file):
                        with open(init_file, "w") as f:
                            f.write('"""Package initialization."""\n')
                        logging.info(f"Created {init_file}")
        
        logging.info("Import paths fixed successfully")
    except Exception as e:
        logging.error(f"Error fixing imports: {e}")

if __name__ == "__main__":
    logging.info("Starting import fix process")
    check_and_install_dependencies()
    fix_imports()
    logging.info("Import fix process completed")

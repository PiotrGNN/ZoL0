
"""
Script to fix and run tests in the project.
This ensures all tests can be correctly executed.
"""

import os
import sys
import importlib
import logging
import unittest
import pytest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_errors.log"),
        logging.StreamHandler()
    ]
)

def fix_test_imports():
    """Fix import issues in test files."""
    try:
        # Ensure project root is in path
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.append(project_root)
            logging.info(f"Added {project_root} to Python path")
            
        # Add data directory to path if it exists
        data_dir = os.path.join(project_root, "data")
        if os.path.isdir(data_dir) and data_dir not in sys.path:
            sys.path.append(data_dir)
            logging.info(f"Added {data_dir} to Python path")
            
        # Find test directories
        test_dirs = []
        for root, dirs, files in os.walk(project_root):
            if "tests" in dirs:
                test_dir = os.path.join(root, "tests")
                test_dirs.append(test_dir)
                
        if not test_dirs:
            logging.warning("No test directories found")
            
        # Fix each test directory
        for test_dir in test_dirs:
            if test_dir not in sys.path:
                sys.path.append(test_dir)
                logging.info(f"Added {test_dir} to Python path")
                
            # Ensure __init__.py exists
            init_file = os.path.join(test_dir, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write('"""Test package initialization."""\n')
                logging.info(f"Created {init_file}")
                
            # Check test files
            for file in os.listdir(test_dir):
                if file.startswith("test_") and file.endswith(".py"):
                    logging.info(f"Found test file: {file}")
        
        logging.info("Test imports fixed successfully")
    except Exception as e:
        logging.error(f"Error fixing test imports: {e}")

def run_tests():
    """Run all tests using pytest."""
    try:
        logging.info("Running tests...")
        # Add the current directory to the path
        sys.path.insert(0, os.getcwd())
        
        # Run pytest - captures output
        result = pytest.main(["-v", "data/tests"])
        return result
    except Exception as e:
        logging.error(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    logging.info("Starting test fix process")
    fix_test_imports()
    exit_code = run_tests()
    logging.info(f"Test process completed with exit code: {exit_code}")
    sys.exit(exit_code)

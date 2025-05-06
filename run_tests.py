#!/usr/bin/env python3
"""Simple test runner."""
import os
import sys
import pytest

if __name__ == "__main__":
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Run tests
    pytest.main([
        "data/tests/test_execution.py",
        "data/tests/test_data_processing.py",
        "data/tests/test_risk_management.py",
        "-v",
        "--cov=data",
        "--cov-report=term-missing"
    ])
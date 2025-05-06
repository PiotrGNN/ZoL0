"""Test package initialization."""

import os
import sys
import logging
from typing import Dict, Any, List

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Basic imports needed by test modules
from .base_test import BaseTestCase
from .test_utils import MockDB, MockCache, MockNotifier, MockTestCase

__all__ = [
    'BaseTestCase',
    'MockTestCase',
    'MockDB',
    'MockCache',
    'MockNotifier'
]

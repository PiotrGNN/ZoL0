import unittest
import os

class BaseTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create any necessary test directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('config', exist_ok=True)
        
    def tearDown(self):
        """Clean up test environment."""
        # Clean up any test files
        test_files = [
            'config/test_config.yaml',
            'logs/test.log'
        ]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
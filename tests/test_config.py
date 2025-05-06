"""
test_config.py
-------------
Tests for configuration handling and validation.
"""

import os
import yaml
import pytest
from typing import Dict, Any

from config.config_loader import ConfigLoader
from tests import BaseTestCase

class TestConfiguration(BaseTestCase):
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.config_loader = ConfigLoader()
        self.test_config = {
            'trading': {
                'exchange': 'bybit',
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'timeframes': ['1h', '4h'],
                'risk': {
                    'max_position_size': 1000,
                    'stop_loss_pct': 2.0,
                    'take_profit_pct': 4.0
                }
            },
            'models': {
                'anomaly_detector': {
                    'threshold': 0.95,
                    'window_size': 100
                },
                'sentiment_analyzer': {
                    'model_type': 'bert',
                    'confidence_threshold': 0.8
                }
            },
            'logging': {
                'level': 'INFO',
                'file_path': 'logs/trading.log'
            }
        }

    def test_config_loading(self):
        """Test configuration loading from file."""
        # Write test config
        with open('config/test_config.yaml', 'w') as f:
            yaml.dump(self.test_config, f)

        config = self.config_loader.load_config('config/test_config.yaml')
        self.assertEqual(config, self.test_config)

    def test_config_validation(self):
        """Test configuration validation."""
        required_fields = [
            'trading.exchange',
            'trading.symbols',
            'trading.risk.max_position_size',
            'models.anomaly_detector.threshold',
            'logging.level'
        ]

        for field in required_fields:
            config = self._deep_copy_dict(self.test_config)
            self._delete_nested_key(config, field)
            
            with self.assertRaises(ValueError):
                self.config_loader.validate_config(config)

    def test_config_types(self):
        """Test configuration value types."""
        type_tests = [
            ('trading.exchange', str),
            ('trading.symbols', list),
            ('trading.risk.max_position_size', (int, float)),
            ('trading.risk.stop_loss_pct', float),
            ('models.anomaly_detector.threshold', float),
            ('models.sentiment_analyzer.confidence_threshold', float)
        ]

        for path, expected_type in type_tests:
            value = self._get_nested_value(self.test_config, path)
            self.assertIsInstance(
                value, 
                expected_type, 
                f"Config value at {path} should be of type {expected_type}"
            )

    def test_config_ranges(self):
        """Test configuration value ranges."""
        range_tests = [
            ('trading.risk.stop_loss_pct', 0, 100),
            ('trading.risk.take_profit_pct', 0, 100),
            ('models.anomaly_detector.threshold', 0, 1),
            ('models.sentiment_analyzer.confidence_threshold', 0, 1)
        ]

        for path, min_val, max_val in range_tests:
            value = self._get_nested_value(self.test_config, path)
            self.assertTrue(
                min_val <= value <= max_val,
                f"Config value at {path} should be between {min_val} and {max_val}"
            )

    def test_config_dependencies(self):
        """Test configuration value dependencies."""
        # Test that take_profit is greater than stop_loss
        tp = self.test_config['trading']['risk']['take_profit_pct']
        sl = self.test_config['trading']['risk']['stop_loss_pct']
        self.assertGreater(tp, sl, "Take profit should be greater than stop loss")

    def test_config_environment_override(self):
        """Test environment variable configuration override."""
        env_overrides = {
            'TRADING_EXCHANGE': 'binance',
            'RISK_MAX_POSITION_SIZE': '2000',
            'LOGGING_LEVEL': 'DEBUG'
        }

        # Set environment variables
        for key, value in env_overrides.items():
            os.environ[key] = value

        config = self.config_loader.load_config_with_env_override('config/test_config.yaml')

        self.assertEqual(config['trading']['exchange'], 'binance')
        self.assertEqual(config['trading']['risk']['max_position_size'], 2000)
        self.assertEqual(config['logging']['level'], 'DEBUG')

        # Clean up environment
        for key in env_overrides:
            del os.environ[key]

    def test_config_update(self):
        """Test configuration update functionality."""
        updates = {
            'trading': {
                'risk': {
                    'max_position_size': 1500
                }
            },
            'models': {
                'anomaly_detector': {
                    'threshold': 0.98
                }
            }
        }

        updated_config = self.config_loader.update_config(self.test_config, updates)

        self.assertEqual(updated_config['trading']['risk']['max_position_size'], 1500)
        self.assertEqual(updated_config['models']['anomaly_detector']['threshold'], 0.98)
        # Verify other values remain unchanged
        self.assertEqual(updated_config['trading']['exchange'], self.test_config['trading']['exchange'])

    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of a dictionary."""
        return {
            key: self._deep_copy_dict(value) if isinstance(value, dict) else value
            for key, value in d.items()
        }

    def _delete_nested_key(self, d: Dict[str, Any], path: str) -> None:
        """Delete a nested dictionary key using dot notation."""
        keys = path.split('.')
        for key in keys[:-1]:
            d = d[key]
        del d[keys[-1]]

    def _get_nested_value(self, d: Dict[str, Any], path: str) -> Any:
        """Get a nested dictionary value using dot notation."""
        for key in path.split('.'):
            d = d[key]
        return d

    def tearDown(self):
        """Clean up test resources."""
        super().tearDown()
        if os.path.exists('config/test_config.yaml'):
            os.remove('config/test_config.yaml')
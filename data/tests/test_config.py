"""
test_config.py
-------------
Tests for configuration loading and validation.
"""

import os
import tempfile
import yaml
from data.tests import BaseTestCase
from config.config_loader import ConfigLoader

class TestConfiguration(BaseTestCase):
    """Test configuration functionality."""

    def setUp(self):
        """Initialize test resources."""
        super().setUp()
        self.config_data = {
            "environment": {
                "mode": "test",
                "debug": True,
                "log_level": "INFO"
            },
            "trading": {
                "max_positions": 5,
                "risk_per_trade": 0.02,
                "max_leverage": 3.0,
                "allowed_symbols": ["BTCUSDT", "ETHUSDT"]
            },
            "ai_models": {
                "model_path": "models/",
                "update_interval": 3600,
                "confidence_threshold": 0.75
            },
            "api": {
                "base_url": "https://api.test.com",
                "timeout": 30,
                "retry_attempts": 3
            }
        }
        self.temp_config = self.create_temp_file(suffix=".yaml")
        with open(self.temp_config, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.config_loader = ConfigLoader(config_path=self.temp_config)

    def test_config_loading(self):
        """Test configuration loading and validation."""
        # Test basic loading
        config = self.config_loader.load_config()
        self.assertEqual(config["environment"]["mode"], "test")
        self.assertEqual(config["trading"]["max_positions"], 5)
        
        # Test nested access
        self.assertIsInstance(config["trading"]["allowed_symbols"], list)
        self.assertEqual(len(config["trading"]["allowed_symbols"]), 2)
        
        # Test type validation
        self.assertIsInstance(config["trading"]["risk_per_trade"], float)
        self.assertIsInstance(config["api"]["timeout"], int)
        
        # Test value ranges
        self.assertGreater(config["trading"]["max_leverage"], 0)
        self.assertLess(config["trading"]["risk_per_trade"], 1.0)
        
        # Test required fields
        required_sections = ["environment", "trading", "ai_models", "api"]
        for section in required_sections:
            self.assertIn(section, config)

    def test_config_validation(self):
        """Test configuration validation rules."""
        # Test invalid trading parameters
        invalid_configs = [
            {
                "change": {"trading": {"risk_per_trade": 1.5}},
                "error": ValueError,
                "message": "Risk per trade must be < 1"
            },
            {
                "change": {"trading": {"max_leverage": -1.0}},
                "error": ValueError,
                "message": "Max leverage must be positive"
            },
            {
                "change": {"trading": {"max_positions": 0}},
                "error": ValueError,
                "message": "Max positions must be positive"
            }
        ]
        
        for test_case in invalid_configs:
            with self.subTest(test_case["message"]):
                invalid_data = self.config_data.copy()
                for section, values in test_case["change"].items():
                    invalid_data[section].update(values)
                    
                with open(self.temp_config, 'w') as f:
                    yaml.dump(invalid_data, f)
                    
                with self.assertRaises(test_case["error"]):
                    self.config_loader.load_config()

    def test_environment_override(self):
        """Test environment variable overrides."""
        # Test environment variable override
        os.environ["TRADING_MAX_POSITIONS"] = "10"
        os.environ["API_TIMEOUT"] = "60"
        
        config = self.config_loader.load_config()
        self.assertEqual(config["trading"]["max_positions"], 10)
        self.assertEqual(config["api"]["timeout"], 60)
        
        # Clean up
        del os.environ["TRADING_MAX_POSITIONS"]
        del os.environ["API_TIMEOUT"]

    def test_config_persistence(self):
        """Test configuration persistence and reloading."""
        # Modify and save config
        config = self.config_loader.load_config()
        config["trading"]["max_positions"] = 8
        
        save_path = self.create_temp_file(suffix=".yaml")
        self.config_loader.save_config(config, save_path)
        
        # Load modified config
        new_loader = ConfigLoader(config_path=save_path)
        loaded_config = new_loader.load_config()
        self.assertEqual(loaded_config["trading"]["max_positions"], 8)
        
        # Test config merging
        override_config = {
            "trading": {"max_leverage": 2.0},
            "api": {"timeout": 45}
        }
        merged_config = self.config_loader.merge_configs(config, override_config)
        self.assertEqual(merged_config["trading"]["max_leverage"], 2.0)
        self.assertEqual(merged_config["api"]["timeout"], 45)
        self.assertEqual(
            merged_config["trading"]["max_positions"],
            config["trading"]["max_positions"]
        )

    def test_config_security(self):
        """Test configuration security measures."""
        # Test sensitive field masking
        sensitive_config = {
            "api": {
                "key": "secret_api_key",
                "secret": "very_secret_value"
            }
        }
        
        masked_config = self.config_loader.mask_sensitive_data(sensitive_config)
        self.assertEqual(masked_config["api"]["key"], "********")
        self.assertEqual(masked_config["api"]["secret"], "********")
        
        # Test file permissions
        if os.name != 'nt':  # Skip on Windows
            secure_config = self.create_temp_file(suffix=".yaml")
            with open(secure_config, 'w') as f:
                yaml.dump(sensitive_config, f)
            
            # Check file permissions (only owner should have read access)
            file_permissions = oct(os.stat(secure_config).st_mode)[-3:]
            self.assertEqual(file_permissions, "600")

    def test_config_validation_rules(self):
        """Test custom configuration validation rules."""
        validation_cases = [
            {
                "rule": "trading.risk_per_trade <= trading.max_risk_per_symbol",
                "config": {
                    "trading": {
                        "risk_per_trade": 0.02,
                        "max_risk_per_symbol": 0.01
                    }
                },
                "should_pass": False
            },
            {
                "rule": "trading.allowed_symbols subset of api.supported_symbols",
                "config": {
                    "trading": {"allowed_symbols": ["BTCUSDT", "UNKNOWN"]},
                    "api": {"supported_symbols": ["BTCUSDT", "ETHUSDT"]}
                },
                "should_pass": False
            },
            {
                "rule": "ai_models.confidence_threshold in [0.5, 1.0]",
                "config": {
                    "ai_models": {"confidence_threshold": 0.3}
                },
                "should_pass": False
            }
        ]
        
        for case in validation_cases:
            with self.subTest(rule=case["rule"]):
                config_data = self.config_data.copy()
                for section, values in case["config"].items():
                    config_data[section].update(values)
                
                with open(self.temp_config, 'w') as f:
                    yaml.dump(config_data, f)
                
                if case["should_pass"]:
                    try:
                        self.config_loader.load_config()
                    except ValueError:
                        self.fail(f"Validation rule '{case['rule']}' failed unexpectedly")
                else:
                    with self.assertRaises(ValueError):
                        self.config_loader.load_config()

    def test_config_defaults(self):
        """Test configuration defaults and fallbacks."""
        # Remove some values to test defaults
        minimal_config = {
            "environment": {"mode": "test"},
            "trading": {"allowed_symbols": ["BTCUSDT"]}
        }
        
        with open(self.temp_config, 'w') as f:
            yaml.dump(minimal_config, f)
        
        config = self.config_loader.load_config()
        
        # Check default values
        self.assertEqual(config["environment"].get("log_level", "INFO"), "INFO")
        self.assertEqual(config["api"].get("timeout", 30), 30)
        self.assertEqual(config["trading"].get("max_positions", 5), 5)
        
        # Test layered defaults
        self.assertIsInstance(config["trading"].get("risk_limits", {}), dict)
        self.assertGreater(
            config["trading"].get("risk_limits", {}).get("max_drawdown", 0.5),
            0
        )
"""
test_security.py
--------------
Security tests for the trading system.
"""

import os
import json
import hashlib
import pytest
from pathlib import Path
from typing import Dict, Any
import secrets
import base64

from data.tests import BaseTestCase
from config.config_loader import ConfigLoader
from data.execution.order_execution import OrderExecution
from data.utils.encryption import encrypt_data, decrypt_data

class TestSecurity(BaseTestCase):
    """Test security measures in the trading system."""

    def setUp(self):
        """Set up security test resources."""
        super().setUp()
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config()
        self.test_api_key = "test_api_key"
        self.test_api_secret = "test_api_secret"

    @pytest.mark.security
    def test_config_file_permissions(self):
        """Test configuration file security."""
        if os.name == 'nt':  # Skip on Windows
            self.skipTest("File permission tests not applicable on Windows")
            
        config_file = Path(self.config_loader.config_path)
        
        # Check file exists
        self.assertTrue(config_file.exists())
        
        # Check file permissions (only owner should have read access)
        mode = oct(config_file.stat().st_mode)[-3:]
        self.assertEqual(
            mode,
            "600",
            f"Config file has unsafe permissions: {mode}"
        )
        
        # Verify sensitive data is not stored in plaintext
        with open(config_file) as f:
            content = f.read()
            self.assertNotIn(self.test_api_key, content)
            self.assertNotIn(self.test_api_secret, content)

    @pytest.mark.security
    def test_api_key_handling(self):
        """Test secure API key handling."""
        # Test key encryption
        encrypted_key = encrypt_data(self.test_api_key)
        self.assertNotEqual(encrypted_key, self.test_api_key)
        
        # Test key decryption
        decrypted_key = decrypt_data(encrypted_key)
        self.assertEqual(decrypted_key, self.test_api_key)
        
        # Test key storage
        key_file = Path("api_credentials.enc")
        try:
            # Save encrypted credentials
            credentials = {
                "api_key": encrypted_key,
                "api_secret": encrypt_data(self.test_api_secret)
            }
            with open(key_file, "w") as f:
                json.dump(credentials, f)
            
            # Check file permissions
            if os.name != 'nt':
                self.assertEqual(
                    oct(key_file.stat().st_mode)[-3:],
                    "600"
                )
            
            # Verify data is encrypted
            with open(key_file) as f:
                stored_data = f.read()
                self.assertNotIn(self.test_api_key, stored_data)
                self.assertNotIn(self.test_api_secret, stored_data)
        finally:
            if key_file.exists():
                key_file.unlink()

    @pytest.mark.security
    def test_order_validation(self):
        """Test order validation and sanitization."""
        execution = OrderExecution()
        
        # Test SQL injection in symbol
        malicious_orders = [
            {
                "symbol": "BTCUSDT'; DROP TABLE orders;--",
                "side": "BUY",
                "quantity": 0.001
            },
            {
                "symbol": "ETHUSDT\x00",  # Null byte injection
                "side": "SELL",
                "quantity": 0.001
            },
            {
                "symbol": "<script>alert('xss')</script>",
                "side": "BUY",
                "quantity": 0.001
            }
        ]
        
        for order in malicious_orders:
            with self.assertRaises(ValueError):
                execution.validate_order(order)

        # Test numeric overflow
        with self.assertRaises(ValueError):
            execution.validate_order({
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": float('inf')
            })

    @pytest.mark.security
    def test_input_sanitization(self):
        """Test input sanitization for data processing."""
        malicious_inputs = [
            "../../../etc/passwd",  # Path traversal
            "data/*;rm -rf /",     # Command injection
            "data\x00hidden",      # Null byte injection
            "<script>alert(1)</script>",  # XSS
            "data/**/;touch /tmp/pwned",  # Command injection
        ]
        
        for bad_input in malicious_inputs:
            # Test file path sanitization
            with self.assertRaises(ValueError):
                self.config_loader.validate_path(bad_input)
            
            # Test command sanitization
            with self.assertRaises(ValueError):
                self.config_loader.validate_command(bad_input)

    @pytest.mark.security
    def test_session_management(self):
        """Test session security measures."""
        # Test session token generation
        token = self._generate_session_token()
        self.assertEqual(len(token), 32)  # 256 bits
        self.assertTrue(self._is_random_enough(token))
        
        # Test session timeout
        session = {
            "token": token,
            "created_at": time.time() - 3700  # 1 hour + 100 seconds ago
        }
        self.assertTrue(
            self._is_session_expired(session),
            "Session should expire after 1 hour"
        )

    @pytest.mark.security
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        execution = OrderExecution()
        max_requests = 10
        window_seconds = 1
        
        # Test normal operation
        for _ in range(max_requests):
            self.assertTrue(execution.check_rate_limit())
        
        # Test rate limit exceeded
        self.assertFalse(execution.check_rate_limit())
        
        # Test rate limit reset
        time.sleep(window_seconds)
        self.assertTrue(execution.check_rate_limit())

    @pytest.mark.security
    def test_data_integrity(self):
        """Test data integrity measures."""
        test_data = self.generate_test_data()
        
        # Calculate checksum of original data
        original_hash = self._calculate_checksum(test_data)
        
        # Modify data
        test_data.loc[0, "close"] *= 1.1
        modified_hash = self._calculate_checksum(test_data)
        
        # Verify checksum detects changes
        self.assertNotEqual(original_hash, modified_hash)

    def _generate_session_token(self) -> str:
        """Generate a secure session token."""
        return secrets.token_hex(16)  # 256 bits of randomness

    def _is_random_enough(self, token: str) -> bool:
        """Check if a token has sufficient entropy."""
        # Convert hex string to bytes
        token_bytes = bytes.fromhex(token)
        
        # Calculate entropy
        entropy = self._calculate_entropy(token_bytes)
        return entropy > 3.0  # Require at least 3 bits of entropy per byte

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0
            
        entropy = 0
        for x in range(256):
            p_x = data.count(x) / len(data)
            if p_x > 0:
                entropy += -p_x * math.log2(p_x)
        return entropy

    def _is_session_expired(self, session: Dict[str, Any]) -> bool:
        """Check if a session is expired."""
        max_age = 3600  # 1 hour
        return time.time() - session["created_at"] > max_age

    def _calculate_checksum(self, df) -> str:
        """Calculate SHA-256 checksum of DataFrame."""
        return hashlib.sha256(
            df.to_json().encode()
        ).hexdigest()

"""Security and end-to-end tests for the trading system."""

import pytest
import pandas as pd
import numpy as np
import json
import requests
from typing import Dict, Any
import logging
from unittest.mock import patch
from .base_test import BaseTestCase

class TestSecurity(BaseTestCase):
    """Test suite for security-related functionality."""
    
    def setUp(self):
        """Set up test resources."""
        super().setUp()
        self.test_data = self.generate_test_data(periods=100)
        
    @pytest.mark.security
    def test_input_validation(self):
        """Test input validation and sanitization."""
        from ai_models.model_training import ModelTrainer
        trainer = ModelTrainer()
        
        # Test SQL injection attempts
        malicious_data = pd.DataFrame({
            'feature': ["1.0; DROP TABLE users;--", "1.0' UNION SELECT * FROM secrets;--"],
            'target': [1.0, 1.0]
        })
        
        with self.assertRaises(ValueError):
            trainer.validate_data(malicious_data)
            
        # Test numeric overflow attempts
        overflow_data = pd.DataFrame({
            'feature': [1e308, -1e308],  # Very large values
            'target': [1.0, 1.0]
        })
        
        with self.assertRaises(ValueError):
            trainer.validate_data(overflow_data)
            
    @pytest.mark.security
    def test_api_security(self):
        """Test API endpoint security."""
        from dashboard_api import app
        client = app.test_client()
        
        # Test unauthorized access
        response = client.get('/api/models/sensitive')
        self.assertEqual(response.status_code, 401)
        
        # Test invalid tokens
        response = client.get(
            '/api/models/sensitive',
            headers={'Authorization': 'Bearer invalid_token'}
        )
        self.assertEqual(response.status_code, 401)
        
        # Test CORS headers
        response = client.options('/api/models/predict')
        self.assertIn('Access-Control-Allow-Origin', response.headers)
        
    @pytest.mark.security
    def test_model_security(self):
        """Test model security features."""
        from ai_models.model_loader import ModelLoader
        loader = ModelLoader()
        
        # Test model signature verification
        with self.assertRaises(ValueError):
            loader.load_model("tampered_model.pkl")
            
        # Test input bounds checking
        model = loader.load_model("valid_model.pkl")
        with self.assertRaises(ValueError):
            model.predict(np.array([[1e9]]))  # Out of bounds input
            
class TestE2E(BaseTestCase):
    """End-to-end test suite."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test-wide resources."""
        super().setUpClass()
        from dashboard_api import app
        cls.app = app.test_client()
        cls.api_base = "http://localhost:5000/api"
        
    @pytest.mark.e2e
    def test_full_trading_cycle(self):
        """Test complete trading cycle from data ingestion to order execution."""
        # 1. Data ingestion
        market_data = self.generate_test_data(periods=100)
        response = self.app.post(
            f"{self.api_base}/data/ingest",
            json=market_data.to_dict(orient="records")
        )
        self.assertEqual(response.status_code, 200)
        
        # 2. Feature engineering
        response = self.app.post(
            f"{self.api_base}/features/generate",
            json={"symbol": "BTCUSDT", "timeframe": "1h"}
        )
        self.assertEqual(response.status_code, 200)
        features = response.get_json()
        self.assertIsNotNone(features)
        
        # 3. Model prediction
        response = self.app.post(
            f"{self.api_base}/models/predict",
            json={"features": features}
        )
        self.assertEqual(response.status_code, 200)
        prediction = response.get_json()
        self.assertIn("prediction", prediction)
        self.assertIn("confidence", prediction)
        
        # 4. Risk check
        response = self.app.post(
            f"{self.api_base}/risk/check",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "price": 50000.0
            }
        )
        self.assertEqual(response.status_code, 200)
        risk_check = response.get_json()
        self.assertTrue(risk_check["approved"])
        
        # 5. Order execution
        if risk_check["approved"]:
            response = self.app.post(
                f"{self.api_base}/orders/execute",
                json={
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "MARKET",
                    "quantity": 0.001
                }
            )
            self.assertEqual(response.status_code, 200)
            order = response.get_json()
            self.assertEqual(order["status"], "FILLED")
            
    @pytest.mark.e2e
    def test_error_handling_flow(self):
        """Test system behavior under error conditions."""
        # 1. Test invalid data handling
        response = self.app.post(
            f"{self.api_base}/data/ingest",
            json=[{"invalid": "data"}]
        )
        self.assertEqual(response.status_code, 400)
        
        # 2. Test system recovery
        # Send valid data after invalid
        valid_data = self.generate_test_data(periods=10)
        response = self.app.post(
            f"{self.api_base}/data/ingest",
            json=valid_data.to_dict(orient="records")
        )
        self.assertEqual(response.status_code, 200)
        
        # 3. Test concurrent request handling
        import concurrent.futures
        import random
        
        def make_request():
            endpoint = random.choice([
                "data/ingest",
                "features/generate",
                "models/predict"
            ])
            return self.app.get(f"{self.api_base}/{endpoint}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                self.assertIn(response.status_code, [200, 400, 429])  # Include rate limit
                
    @pytest.mark.e2e
    def test_model_retraining_flow(self):
        """Test model retraining and update process."""
        # 1. Get current model version
        response = self.app.get(f"{self.api_base}/models/status")
        self.assertEqual(response.status_code, 200)
        initial_status = response.get_json()
        
        # 2. Trigger retraining
        response = self.app.post(
            f"{self.api_base}/models/retrain",
            json={"mode": "full"}
        )
        self.assertEqual(response.status_code, 200)
        
        # 3. Monitor training progress
        max_wait = 60  # Maximum wait time in seconds
        while max_wait > 0:
            response = self.app.get(f"{self.api_base}/models/status")
            status = response.get_json()
            if status["status"] == "ready":
                break
            max_wait -= 1
            import time
            time.sleep(1)
            
        self.assertGreater(max_wait, 0, "Model retraining timed out")
        
        # 4. Verify model improvement
        response = self.app.get(f"{self.api_base}/models/metrics")
        self.assertEqual(response.status_code, 200)
        metrics = response.get_json()
        self.assertGreaterEqual(
            metrics["accuracy"],
            initial_status["metrics"]["accuracy"]
        )
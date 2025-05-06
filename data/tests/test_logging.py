"""
test_logging.py
-------------
Tests for logging and monitoring functionality.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import pytest
from unittest.mock import patch

from data.tests import BaseTestCase
from data.utils.logger import setup_logging, LogHandler
from data.monitoring.system_monitor import SystemMonitor
from data.monitoring.alerts import AlertManager
from data.monitoring.metrics import MetricsCollector

class TestLogging(BaseTestCase):
    """Test logging and monitoring functionality."""

    def setUp(self):
        """Initialize test resources."""
        super().setUp()
        self.log_dir = Path(tempfile.mkdtemp())
        self.log_file = self.log_dir / "test.log"
        self.metrics_file = self.log_dir / "metrics.json"
        
        # Setup logging for tests
        self.logger = setup_logging(
            log_file=str(self.log_file),
            level=logging.DEBUG
        )
        
        # Initialize monitoring components
        self.monitor = SystemMonitor()
        self.alert_manager = AlertManager()
        self.metrics_collector = MetricsCollector(
            metrics_file=str(self.metrics_file)
        )

    def test_log_levels(self):
        """Test logging at different levels."""
        test_messages = {
            "debug": "Debug message test",
            "info": "Info message test",
            "warning": "Warning message test",
            "error": "Error message test",
            "critical": "Critical message test"
        }
        
        # Log messages at different levels
        self.logger.debug(test_messages["debug"])
        self.logger.info(test_messages["info"])
        self.logger.warning(test_messages["warning"])
        self.logger.error(test_messages["error"])
        self.logger.critical(test_messages["critical"])
        
        # Verify log contents
        with open(self.log_file) as f:
            log_content = f.read()
            
        for level, message in test_messages.items():
            self.assertIn(
                message,
                log_content,
                f"Log message '{message}' not found in log file"
            )
            
        # Verify log format
        log_lines = log_content.splitlines()
        for line in log_lines:
            self.assertRegex(
                line,
                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[\w+\] .*"
            )

    def test_error_logging(self):
        """Test error logging and stack traces."""
        error_message = "Test error message"
        
        try:
            raise ValueError(error_message)
        except ValueError:
            self.logger.exception("Caught test exception")
        
        # Verify error log
        with open(self.log_file) as f:
            log_content = f.read()
            
        self.assertIn(error_message, log_content)
        self.assertIn("Traceback", log_content)
        self.assertIn("ValueError", log_content)

    def test_log_rotation(self):
        """Test log file rotation."""
        max_size = 1024  # 1KB
        backup_count = 3
        
        # Configure rotating handler
        rotating_logger = setup_logging(
            log_file=str(self.log_file),
            max_bytes=max_size,
            backup_count=backup_count
        )
        
        # Generate logs until rotation occurs
        long_message = "x" * (max_size // 10)
        for _ in range(20):  # Should cause multiple rotations
            rotating_logger.info(long_message)
        
        # Verify rotated files exist
        for i in range(1, backup_count + 1):
            rotated_file = Path(f"{self.log_file}.{i}")
            self.assertTrue(
                rotated_file.exists(),
                f"Rotated log file {rotated_file} not found"
            )

    def test_system_monitoring(self):
        """Test system monitoring and metrics collection."""
        # Collect system metrics
        metrics = self.monitor.collect_metrics()
        
        # Verify required metrics
        required_metrics = [
            "cpu_usage",
            "memory_usage",
            "disk_usage",
            "network_latency"
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
        # Test metrics persistence
        self.metrics_collector.save_metrics(metrics)
        loaded_metrics = self.metrics_collector.load_metrics()
        
        self.assertEqual(metrics, loaded_metrics)

    def test_alert_generation(self):
        """Test alert generation and notification."""
        # Define test alerts
        test_alerts = [
            {
                "level": "warning",
                "message": "High CPU usage detected",
                "metric": "cpu_usage",
                "threshold": 80,
                "value": 85
            },
            {
                "level": "critical",
                "message": "Memory usage exceeded limit",
                "metric": "memory_usage",
                "threshold": 90,
                "value": 95
            }
        ]
        
        # Test alert processing
        with patch.object(self.alert_manager, 'send_notification') as mock_notify:
            for alert in test_alerts:
                self.alert_manager.process_alert(alert)
                
                # Verify notification was sent
                mock_notify.assert_called_with(
                    level=alert["level"],
                    message=alert["message"]
                )
                
                # Verify alert was logged
                with open(self.log_file) as f:
                    log_content = f.read()
                    self.assertIn(alert["message"], log_content)

    def test_performance_logging(self):
        """Test performance metric logging."""
        # Test execution time logging
        with LogHandler.measure_time() as timer:
            # Simulate some work
            result = sum(range(1000000))
        
        execution_time = timer.execution_time
        self.assertGreater(execution_time, 0)
        
        # Verify timing log
        with open(self.log_file) as f:
            log_content = f.read()
            self.assertIn(
                f"Execution time: {execution_time:.3f}s",
                log_content
            )

    def test_structured_logging(self):
        """Test structured logging format."""
        # Log structured data
        test_data = {
            "event": "trade_executed",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.001,
            "price": 50000.0
        }
        
        self.logger.info("Trade executed", extra=test_data)
        
        # Verify JSON formatting
        with open(self.log_file) as f:
            last_line = f.readlines()[-1]
            log_entry = json.loads(last_line)
            
            self.assertEqual(log_entry["event"], "trade_executed")
            self.assertEqual(log_entry["symbol"], "BTCUSDT")
            self.assertIsInstance(log_entry["timestamp"], str)

    def test_log_filtering(self):
        """Test log filtering and searching."""
        # Generate test logs
        test_logs = [
            ("info", "Normal operation message"),
            ("warning", "Resource usage high"),
            ("error", "Connection failed"),
            ("info", "Resource usage normal"),
            ("critical", "System shutdown required")
        ]
        
        for level, message in test_logs:
            getattr(self.logger, level)(message)
        
        # Test log searching
        log_handler = LogHandler(str(self.log_file))
        
        # Search by level
        error_logs = log_handler.filter_logs(level="error")
        self.assertEqual(len(error_logs), 1)
        self.assertIn("Connection failed", error_logs[0])
        
        # Search by keyword
        resource_logs = log_handler.filter_logs(keyword="Resource")
        self.assertEqual(len(resource_logs), 2)
        
        # Search by time range
        recent_logs = log_handler.filter_logs(
            minutes=5
        )
        self.assertEqual(len(recent_logs), len(test_logs))

    def tearDown(self):
        """Clean up test resources."""
        super().tearDown()
        # Remove log directory and files
        for file in self.log_dir.glob("*"):
            file.unlink()
        self.log_dir.rmdir()
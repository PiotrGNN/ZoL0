"""
test_interfaces.py
---------------
Tests to verify component interfaces and contracts.
"""

import inspect
from typing import Any, Dict, List, Type
import pytest
from abc import ABC, abstractmethod

from data.tests import BaseTestCase
from data.ai_models.model_loader import ModelLoader
from data.ai_models.model_recognition import ModelRecognizer
from data.ai_models.anomaly_detection import AnomalyDetector
from data.ai_models.sentiment_ai import SentimentAnalyzer
from data.data.historical_data import HistoricalDataManager
from data.execution.order_execution import OrderExecution

class TestInterfaces(BaseTestCase):
    """Test component interfaces and contracts."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.required_interfaces = {
            'ModelLoader': {
                'methods': [
                    'load_available_models',
                    'load_model',
                    'get_model_info'
                ],
                'properties': ['models_dir']
            },
            'ModelRecognizer': {
                'methods': [
                    'predict',
                    'fit',
                    'recognize_pattern'
                ],
                'properties': ['model_type', 'confidence_threshold']
            },
            'AnomalyDetector': {
                'methods': [
                    'detect_anomalies',
                    'fit',
                    'predict'
                ],
                'properties': ['threshold', 'model_type']
            },
            'SentimentAnalyzer': {
                'methods': [
                    'analyze_sentiment',
                    'preprocess_text',
                    'predict'
                ],
                'properties': ['model_type']
            },
            'HistoricalDataManager': {
                'methods': [
                    'load_data',
                    'save_data',
                    'query_database',
                    'load_to_database'
                ],
                'properties': ['db_path', 'csv_path']
            },
            'OrderExecution': {
                'methods': [
                    'send_order',
                    'cancel_order',
                    'get_order_status',
                    'get_position'
                ],
                'properties': ['exchange', 'api_key']
            }
        }

    def test_required_interfaces(self):
        """Test that components implement required interfaces."""
        components = {
            'ModelLoader': ModelLoader,
            'ModelRecognizer': ModelRecognizer,
            'AnomalyDetector': AnomalyDetector,
            'SentimentAnalyzer': SentimentAnalyzer,
            'HistoricalDataManager': HistoricalDataManager,
            'OrderExecution': OrderExecution
        }
        
        for component_name, component_class in components.items():
            interface = self.required_interfaces[component_name]
            
            # Check required methods
            for method_name in interface['methods']:
                self.assertTrue(
                    hasattr(component_class, method_name),
                    f"{component_name} missing required method: {method_name}"
                )
                method = getattr(component_class, method_name)
                self.assertTrue(
                    callable(method),
                    f"{component_name}.{method_name} is not callable"
                )
            
            # Check required properties
            for prop_name in interface['properties']:
                self.assertTrue(
                    hasattr(component_class, prop_name),
                    f"{component_name} missing required property: {prop_name}"
                )

    def test_method_signatures(self):
        """Test method signatures match expected interfaces."""
        expected_signatures = {
            'ModelLoader.load_model': {
                'params': ['self', 'model_name'],
                'return': Any
            },
            'ModelRecognizer.predict': {
                'params': ['self', 'features'],
                'return': Any
            },
            'AnomalyDetector.detect_anomalies': {
                'params': ['self', 'data'],
                'return': List
            },
            'SentimentAnalyzer.analyze_sentiment': {
                'params': ['self', 'text'],
                'return': Dict
            },
            'OrderExecution.send_order': {
                'params': ['self', 'symbol', 'side', 'type', 'quantity'],
                'return': Dict
            }
        }
        
        for method_path, expected in expected_signatures.items():
            class_name, method_name = method_path.split('.')
            component_class = globals()[class_name]
            method = getattr(component_class, method_name)
            
            # Get method signature
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            
            # Verify parameters
            for param in expected['params']:
                self.assertIn(
                    param,
                    params,
                    f"{method_path} missing parameter: {param}"
                )
            
            # Verify return type if specified
            if expected['return'] is not None:
                self.assertTrue(
                    hasattr(method, '__annotations__') and
                    'return' in method.__annotations__,
                    f"{method_path} missing return type annotation"
                )

    def test_component_initialization(self):
        """Test component initialization with various parameters."""
        test_cases = [
            {
                'class': ModelLoader,
                'params': {'models_dir': 'test_models'},
                'expected_attrs': {'models_dir'}
            },
            {
                'class': ModelRecognizer,
                'params': {'confidence_threshold': 0.8},
                'expected_attrs': {'confidence_threshold'}
            },
            {
                'class': AnomalyDetector,
                'params': {'threshold': 0.95},
                'expected_attrs': {'threshold'}
            },
            {
                'class': SentimentAnalyzer,
                'params': {'model_type': 'bert'},
                'expected_attrs': {'model_type'}
            }
        ]
        
        for case in test_cases:
            # Initialize component
            component = case['class'](**case['params'])
            
            # Check attributes were set correctly
            for attr in case['expected_attrs']:
                self.assertTrue(
                    hasattr(component, attr),
                    f"{case['class'].__name__} missing attribute: {attr}"
                )
                self.assertEqual(
                    getattr(component, attr),
                    case['params'][attr],
                    f"{case['class'].__name__}.{attr} not set correctly"
                )

    def test_error_handling(self):
        """Test error handling in component interfaces."""
        test_cases = [
            {
                'component': ModelLoader(),
                'method': 'load_model',
                'args': ['nonexistent_model'],
                'expected_error': ValueError
            },
            {
                'component': OrderExecution(),
                'method': 'send_order',
                'args': ['INVALID', 'BUY', 'MARKET', -1.0],
                'expected_error': ValueError
            },
            {
                'component': HistoricalDataManager(db_path=':memory:', csv_path='test.csv'),
                'method': 'query_database',
                'args': ['INVALID SQL'],
                'expected_error': Exception
            }
        ]
        
        for case in test_cases:
            with self.assertRaises(
                case['expected_error'],
                msg=f"Expected {case['expected_error']} not raised"
            ):
                method = getattr(case['component'], case['method'])
                method(*case['args'])

    def test_interface_consistency(self):
        """Test consistency of interface behavior across instances."""
        # Test ModelLoader interface consistency
        loader1 = ModelLoader()
        loader2 = ModelLoader()
        
        models1 = loader1.load_available_models()
        models2 = loader2.load_available_models()
        
        self.assertEqual(
            set(models1.keys()),
            set(models2.keys()),
            "ModelLoader interface inconsistent across instances"
        )
        
        # Test ModelRecognizer interface consistency
        recognizer1 = ModelRecognizer()
        recognizer2 = ModelRecognizer()
        
        self.assertEqual(
            recognizer1.confidence_threshold,
            recognizer2.confidence_threshold,
            "ModelRecognizer interface inconsistent across instances"
        )
        
        # Test AnomalyDetector interface consistency
        detector1 = AnomalyDetector()
        detector2 = AnomalyDetector()
        
        self.assertEqual(
            detector1.threshold,
            detector2.threshold,
            "AnomalyDetector interface inconsistent across instances"
        )

    def test_interface_documentation(self):
        """Test presence and format of interface documentation."""
        components = [
            ModelLoader,
            ModelRecognizer,
            AnomalyDetector,
            SentimentAnalyzer,
            HistoricalDataManager,
            OrderExecution
        ]
        
        for component in components:
            # Check class documentation
            self.assertIsNotNone(
                component.__doc__,
                f"{component.__name__} missing class documentation"
            )
            
            # Check method documentation
            for name, method in inspect.getmembers(component, inspect.isfunction):
                if not name.startswith('_'):  # Skip private methods
                    self.assertIsNotNone(
                        method.__doc__,
                        f"{component.__name__}.{name} missing method documentation"
                    )

    def tearDown(self):
        """Clean up test resources."""
        super().tearDown()
        # Clean up any test instances
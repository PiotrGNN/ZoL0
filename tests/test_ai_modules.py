import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.model_recognition import ModelRecognizer
from ai_models.sentiment_ai import SentimentAnalyzer
from ai_models.anomaly_detection import AnomalyDetector

class TestAIModules(unittest.TestCase):
    def setUp(self):
        """Initialize test objects"""
        self.model_recognizer = ModelRecognizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.anomaly_detector = AnomalyDetector()

    def test_model_recognizer_initialization(self):
        """Test if ModelRecognizer initializes correctly"""
        self.assertIsNotNone(self.model_recognizer)
        
    def test_sentiment_analyzer_initialization(self):
        """Test if SentimentAnalyzer initializes correctly"""
        self.assertIsNotNone(self.sentiment_analyzer)
        
    def test_anomaly_detector_initialization(self):
        """Test if AnomalyDetector initializes correctly"""
        self.assertIsNotNone(self.anomaly_detector)

    # TODO: Add more specific tests for each module
    
if __name__ == '__main__':
    unittest.main()
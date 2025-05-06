"""End-to-end tests validating complete system workflows"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from data.data.data_preprocessing import preprocess_pipeline
from python_libs.bybit_v5_connector import BybitV5Connector
from ai_models.sentiment_ai import SentimentAnalyzer
from ai_models.model_recognition import ModelRecognizer
from ai_models.anomaly_detection import AnomalyDetector

@pytest.mark.e2e
class TestTradingSystemE2E:
    """End-to-end tests for complete trading system workflows"""
    
    @pytest.fixture(autouse=True)
    def setup(self, test_credentials):
        """Setup test environment with all required components"""
        self.connector = BybitV5Connector(**test_credentials)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.pattern_recognizer = ModelRecognizer()
        self.anomaly_detector = AnomalyDetector()
        self.symbol = "BTCUSDT"
        
    def get_test_data(self) -> pd.DataFrame:
        """Get historical data for testing"""
        data = self.connector.get_klines(
            symbol=self.symbol,
            interval="5",
            limit=1000
        )
        return pd.DataFrame(data)
        
    @pytest.mark.e2e
    def test_complete_trading_workflow(self):
        """Test complete trading workflow from data to execution"""
        # 1. Fetch market data
        df = self.get_test_data()
        assert len(df) > 0, "Failed to fetch market data"
        
        # 2. Preprocess data
        df_processed = preprocess_pipeline(df.copy())
        assert not df_processed.empty, "Data preprocessing failed"
        
        # 3. Run pattern recognition
        pattern = self.pattern_recognizer.identify_model_type(df_processed)
        assert pattern is not None, "Pattern recognition failed"
        
        # 4. Check for anomalies
        anomalies = self.anomaly_detector.detect(df_processed)
        assert isinstance(anomalies, (pd.Series, np.ndarray)), "Anomaly detection failed"
        
        # 5. Analyze market sentiment
        sentiment = self.sentiment_analyzer.analyze("BTC showing strong momentum")
        assert isinstance(sentiment, (float, dict)), "Sentiment analysis failed"
        
        # 6. Generate and validate trading signal
        signal = self.generate_trading_signal(pattern, anomalies, sentiment)
        assert self.validate_signal(signal), "Invalid trading signal"
        
        # 7. Execute test trade
        result = self.execute_test_trade(signal)
        assert result["success"], f"Trade execution failed: {result.get('error')}"
        
    def generate_trading_signal(self, pattern: Dict, anomalies: Any, sentiment: Any) -> Dict:
        """Generate trading signal from analysis results"""
        return {
            "symbol": self.symbol,
            "side": "Buy" if sentiment > 0.5 and pattern["confidence"] > 0.7 else "Sell",
            "order_type": "Limit",
            "qty": 0.001,
            "price": None  # Will be set at execution time
        }
        
    def validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal"""
        required_fields = ["symbol", "side", "order_type", "qty"]
        return all(field in signal for field in required_fields)
        
    def execute_test_trade(self, signal: Dict) -> Dict:
        """Execute test trade with small quantity"""
        try:
            # Get current market price
            market_data = self.connector.get_market_data(signal["symbol"])
            current_price = float(market_data["last_price"])
            
            # Set limit price slightly above/below market
            signal["price"] = current_price * (1.01 if signal["side"] == "Buy" else 0.99)
            
            # Place order
            order = self.connector.place_order(**signal)
            if not order.get("success"):
                return {"success": False, "error": "Order placement failed"}
                
            # Cancel order after test
            if order.get("order_id"):
                self.connector.cancel_order(order["order_id"])
                
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    @pytest.mark.e2e
    def test_recovery_workflow(self):
        """Test system recovery from common failure scenarios"""
        # 1. Test network disconnection recovery
        self.connector.disconnect()
        assert self.connector.initialize(), "Failed to recover from disconnection"
        
        # 2. Test invalid order recovery
        invalid_order = {
            "symbol": self.symbol,
            "side": "Buy",
            "order_type": "Limit",
            "qty": 0.00001  # Too small quantity
        }
        result = self.execute_test_trade(invalid_order)
        assert not result["success"], "Should fail with invalid order"
        
        # 3. Test rate limit handling
        for _ in range(10):
            self.connector.get_market_data(self.symbol)
        # Should not raise rate limit exception
        data = self.connector.get_market_data(self.symbol)
        assert data is not None, "Rate limit handling failed"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
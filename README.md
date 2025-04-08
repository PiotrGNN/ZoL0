
# ZoL0-1 Trading System

## 📊 Overview

ZoL0-1 is an advanced algorithmic trading system designed for cryptocurrency markets, with a primary focus on ByBit integration. The system includes AI-powered market analysis, risk management, and automated trading strategies.

## 🚀 Quick Start

### Running on Replit

1. Click the **Run** button at the top of the Replit interface
2. Wait for the system to initialize
3. Access the web dashboard at the opened URL

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## 🔑 Environment Configuration

Create a `.env` file in the root directory with the following parameters:

```
# API Bybit - Configuration
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_USE_TESTNET=true

# Application modes
DEBUG=True
LOG_LEVEL=INFO
IS_PRODUCTION=False
```

## 🏗️ System Architecture

The system consists of several components:

1. **API Connector** - Interface with ByBit API
2. **Data Processor** - Market data processing and analysis
3. **Trading Engine** - Core trading logic and strategy execution
4. **Risk Manager** - Position sizing and risk assessment
5. **AI Models** - Market prediction and anomaly detection

## 📈 Features

- **Advanced Trading Strategies** - Multiple built-in strategies including trend following, mean reversion, and breakout
- **Risk Management** - Sophisticated position sizing and drawdown protection
- **Sentiment Analysis** - Multi-source market sentiment tracking
- **Anomaly Detection** - ML-based market anomaly identification
- **Portfolio Optimization** - Asset allocation and portfolio rebalancing
- **Performance Monitoring** - Real-time trading and system performance metrics

## 📁 Project Structure

```
├── ai_models/            # AI and ML models
├── data/                 # Data processing and strategy implementations
│   ├── execution/        # Exchange connectors and order execution
│   ├── indicators/       # Technical and market indicators
│   ├── optimization/     # Strategy and portfolio optimization
│   ├── risk_management/  # Risk assessment and position sizing
│   ├── strategies/       # Trading strategies
│   └── utils/            # Utility functions and helpers
├── logs/                 # System logs
├── python_libs/          # Local library modules
├── static/               # Web frontend static assets
├── templates/            # HTML templates
├── main.py               # Main application entry point
└── requirements.txt      # Project dependencies
```

## 📝 License

This project is proprietary software. All rights reserved.

## 🔧 Troubleshooting

If you encounter any issues:

1. Check the logs in the `logs/` directory
2. Ensure your API keys are correctly configured
3. Verify that all dependencies are installed
4. For more complex issues, see the system status indicators on the dashboard

## 🛠️ Development

### Adding a New Strategy

Create a new strategy file in `data/strategies/` with the following structure:

```python
class MyNewStrategy:
    def __init__(self, params):
        self.params = params
        
    def analyze(self, data):
        # Implement strategy logic
        return signals
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

## 📊 Performance Metrics

The system tracks key performance indicators:

- Win Rate
- Profit Factor
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio

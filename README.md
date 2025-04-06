
# 🤖 Advanced Trading System with AI

## 📋 Overview

Advanced trading system using AI for market analysis, prediction, and automated execution. The system features:

- 🔌 Multiple exchange connections (Binance, Bybit)
- 🧠 AI-powered market analysis and decision making
- 📊 Backtesting and simulation capabilities
- ⚙️ Real-time trading with market monitoring
- 🚨 Risk management and anomaly detection

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (installed automatically)

### Installation

1. Clone the repository
2. Create and configure `.env` file (use `.env.example` as template)
3. Install dependencies:

```
pip install -r requirements.txt
```

### Running the System

```
python main.py
```

The system will guide you through:
1. Choosing environment (Production/Testnet)
2. Selecting exchange (Binance/Bybit)
3. Starting trading and AI analysis

## 🧩 Project Structure

```
├── ai_models/            # AI and ML models
├── config/               # Configuration files
├── data/                 # Data processing and strategy modules
│   ├── data/             # Data acquisition and preprocessing
│   ├── execution/        # Order execution logic
│   ├── indicators/       # Technical indicators
│   ├── logging/          # Logging utilities
│   ├── optimization/     # Strategy optimization
│   ├── risk_management/  # Risk control modules
│   ├── strategies/       # Trading strategies
│   ├── tests/            # Unit tests
│   └── utils/            # Utility functions
├── logs/                 # Log files
├── saved_models/         # Serialized trained models
└── main.py               # Main entry point
```

## ⚙️ Configuration

All configuration is done through:
- `.env` file for environment variables
- `config/settings.yml` for application settings

## 🛠️ Development

### Testing

Run tests with:
```
pytest data/tests/
```

### Code Style

Format code with:
```
black .
```

Check code style with:
```
flake8
```

## 📄 License

This project is proprietary software.

## 🤝 Contributing

Please read the CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

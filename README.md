
# 🤖 Inteligentny System Tradingowy

## 📝 Project Description
Advanced trading system with real-time market analysis, risk management, and intelligent order execution. Utilizes machine learning algorithms and market sentiment analysis for automated decision-making.

## 🚀 Features
- Connection to Bybit API (testnet and production)
- Technical analysis and algorithmic trading strategies
- Automatic adjustment to API limits (exponential backoff)
- AI models for price movement prediction
- Transaction monitoring and notification system
- Interactive dashboard (Flask)

## 📋 System Requirements
- Python 3.10+ (3.10 recommended)
- Active Bybit account with API keys
- Internet connection
- Windows 10/11

## ⚙️ Local Installation on Windows

```cmd
# Clone repository (if using Git)
git clone <repository_URL>
cd intelligent-trading-system

# Alternatively, after downloading .zip archive
# 1. Extract the ZIP file
# 2. Open command prompt (cmd) in the extracted project location

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configuration
copy .env.example .env
# Edit the .env file with your API keys
```

## 🔧 Configuration
1. Create API keys in your Bybit account panel
2. Fill in the `.env` file with your keys
3. Set `BYBIT_TESTNET=true` for test environment or `BYBIT_TESTNET=false` for production

### ⚠️ Production Environment
When `BYBIT_TESTNET=false`, you are operating with real funds! The system will apply additional safeguards.

## 🏃 Running on Windows
```cmd
# Standard execution
python main.py

# Alternatively, use the batch file
run_windows.bat

# Alternative execution with full logging
python -u main.py > logs\app_output.log 2>&1
```

## 📊 Dashboard
Access the dashboard at: `http://localhost:5000`

## 🧠 AI Models
The system contains various AI models for market analysis:
- Anomaly detection
- Sentiment analysis
- Price prediction
- Reinforcement learning for trading strategies

## 📊 Dashboard
Access the web panel at `http://localhost:5000` to monitor:
- Current positions
- Account status
- Performance metrics
- Market sentiment and anomalies

## 🔧 Testing Bybit Connection
To test your connection to Bybit API:
```cmd
python test_bybit_connection.py
```

## 🔧 Troubleshooting on Windows

### API limit issues (403/429 Errors)
If you encounter API rate limit errors (403/429):

1. Set `BYBIT_TESTNET=true` in the `.env` file
2. Wait 5-10 minutes before trying again
3. Set `USE_SIMULATED_DATA=true` for testing without making API calls

### Dependency issues
If you have problems installing dependencies, try installing them individually:
```cmd
pip install flask requests pandas numpy python-dotenv pybit
```

### Import errors
If you encounter import errors:
```cmd
python fix_imports.py
```

### Tests
To fix and run tests:
```cmd
python fix_tests.py
```

## 📜 License
This project is distributed under the MIT license.

## 📁 Project Structure
- `main.py` - Main application entry point
- `data/` - Data processing and API integration
  - `execution/` - Modules for exchange interaction
  - `indicators/` - Technical indicators and analysis
  - `risk_management/` - Risk management modules
  - `strategies/` - Trading strategies
  - `utils/` - Utility functions
- `ai_models/` - AI/ML models for market analysis
- `static/` - Frontend resources
- `templates/` - HTML templates
- `logs/` - Application logs

## 🔧 Technologies
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Data Analysis**: Pandas, NumPy
- **Exchange Integration**: ByBit API
- **Data Storage**: JSON caching, cache management

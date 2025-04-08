# Trading System with AI Integration

## 📈 Overview
A comprehensive trading system with AI-powered market analysis, technical indicators, and automated execution features. The system uses ByBit API for trading operations.  This system includes a dashboard for real-time portfolio monitoring and management of automated trading strategies on cryptocurrency exchanges, with a focus on ByBit integration.

## 🚀 Quick Start

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/username/trading-system.git
   cd trading-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the `.env.example` file to `.env` and configure your settings:
   ```bash
   cp .env.example .env
   ```
4. Run the application:
   ```bash
   python main.py
   ```
5. Access the application: Open your browser and go to: `http://localhost:5000`


## 🔧 Configuration
Configure the system through the `.env` file or environment variables:
- `BYBIT_API_KEY`: Your ByBit API key
- `BYBIT_API_SECRET`: Your ByBit API secret
- `BYBIT_USE_TESTNET`: Set to "true" to use the testnet environment (recommended for testing)
- `USE_SIMULATED_DATA`: Set to "true" to use simulated data instead of real API calls
- `RISK_LEVEL` - Level of risk: `low`, `medium`, `high`
- `MAX_POSITION_SIZE` - Maximum position size as % of portfolio
- `API_RATE_LIMIT` - Enable smart API request limiting
- `API_CACHE_ENABLED` - Enable API response caching


## 🧠 AI Models
The system includes various AI models for market analysis:
- Anomaly detection
- Sentiment analysis
- Price prediction
- Reinforcement learning for trading strategies

## 📊 Dashboard
Access the web dashboard at `http://localhost:5000` to monitor:
- Current positions
- Account balance
- Performance metrics
- Market sentiment and anomalies

## 🔧 Troubleshooting

### Problemy z limitami API (403/429 Errors)
If you encounter rate limit errors (403/429):

1. Set `BYBIT_USE_TESTNET=true` in your `.env` file
2. Wait 5-10 minutes before retrying
3. Set `USE_SIMULATED_DATA=true` for testing without API calls
4. Consider using a different IP address (e.g., via VPN or proxy).
5. For testing, use simulation mode - set `USE_SIMULATED_DATA=true` in `.env`


### Problemy z zależnościami
If you have dependency conflicts, try installing without dependencies:
```bash
pip install -r requirements.txt --no-deps
```

Then install missing packages manually.

### Błędy importu
If you encounter import errors:
```bash
python fix_imports.py
```

### Testy
To fix and run tests:
```bash
python fix_tests.py
```


## 📁 Project Structure
- `main.py` - Main application entry point
- `data/` - Data processing and API integration
  - `execution/` - Exchange interaction modules
  - `indicators/` - Technical indicators and analysis
  - `risk_management/` - Risk management modules
  - `strategies/` - Trading strategies
  - `utils/` - Utility functions
- `ai_models/` - AI/ML models for market analysis
- `static/` - Frontend assets
- `templates/` - HTML templates
- `logs/` - Application logs

## 🔧 Technologie
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Analiza danych**: Pandas, NumPy
- **Integracja z giełdą**: ByBit API
- **Przechowywanie danych**: Buforowanie JSON, cache-management

## 📄 Licencja

This project is licensed under the MIT License.
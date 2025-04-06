
# AI Trading Bot

## Overview
Advanced trading system using AI for market analysis, strategy optimization, and automated trading.

## Features
- Dynamic configuration loading and API key management
- Environment selection (Testnet/Production)
- Trading module initialization (Binance, Bybit)
- Backtesting and paper trading simulation
- Real-time trading with market monitoring
- AI-powered market analysis and trend prediction
- Automatic strategy optimization through AI
- Centralized logging and exception handling

## Getting Started
1. Run the application by clicking the "Run" button
2. Select your environment (Production or Testnet)
3. Choose your exchange (Binance or Bybit)
4. The system will initialize and start trading

## Configuration
API keys and secrets should be stored as environment variables:
- `BINANCE_API_KEY` - Your Binance API key
- `BINANCE_API_SECRET` - Your Binance API secret
- `BYBIT_API_KEY` - Your Bybit API key
- `BYBIT_API_SECRET` - Your Bybit API secret

## Modules
- `ai_models/` - AI components for market analysis
- `config/` - Configuration and settings
- `data/` - Data processing, execution, and strategies
- `logs/` - Application logs
- `saved_models/` - Saved AI models

## License
This project is licensed under the MIT License.

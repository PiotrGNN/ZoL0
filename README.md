# Trading System

Advanced trading system with AI-powered analysis, real-time monitoring, and automated execution.

## Features

- Real-time market data processing
- Multiple AI models for market analysis and prediction
- Interactive dashboard with live trading metrics
- Automated trading execution
- System health monitoring
- Performance analytics
- Backtesting capabilities

## Installation

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-system.git
cd trading-system
```

2. Build and run using Docker Compose:
```bash
docker-compose up --build
```

The system will be available at:
- Dashboard: http://localhost:8501
- API: http://localhost:5000
- Monitoring: http://localhost:3000

### Manual Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python -c "from data.db_manager import DatabaseManager; DatabaseManager('sqlite:///data/db/trading.db').init_db()"
```

5. Start the services:
```bash
# Start API server
python main.py &
# Start trading system
python run.py &
# Start dashboard
streamlit run dashboard.py
```

## Configuration

The system can be configured through `config/config.yaml`. Key configuration sections:

- `environment`: Development/production settings
- `trading`: Trading parameters and risk limits
- `ai_models`: AI model configuration
- `api`: API server settings
- `logging`: Logging configuration

## Project Structure

- `ai_models/`: AI model implementations
- `config/`: Configuration files
- `data/`: Data management and storage
- `docs/`: Documentation
- `tests/`: Test suite
- `utils/`: Utility functions
- `dashboard.py`: Interactive dashboard
- `main.py`: API server
- `run.py`: Trading system entry point

## Development

### Setting Up Development Environment

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest tests/
```

### Code Style

This project follows:
- PEP 8 for Python code style
- Type hints for improved code clarity
- Comprehensive docstrings
- Unit tests for all new features

### Adding New Features

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Implement your changes
3. Add tests
4. Run the test suite
5. Create a pull request

## Testing

Run different test suites:

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/ -m unit

# Run integration tests
pytest tests/ -m integration

# Generate coverage report
pytest --cov=./ tests/
```

## Monitoring

The system includes comprehensive monitoring:

- System health metrics
- Trading performance analytics
- AI model performance tracking
- Resource usage monitoring

Access monitoring dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- System Dashboard: http://localhost:8501

## Troubleshooting

Common issues and solutions:

1. Database Connection Issues:
   - Check database file permissions
   - Verify connection string in config

2. AI Model Loading Errors:
   - Ensure model files exist in saved_models/
   - Check model version compatibility

3. API Connection Problems:
   - Verify API key configuration
   - Check network connectivity
   - Review API logs in logs/api.log

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Technical indicators from TA-Lib
- Machine learning implementations using scikit-learn
- Real-time visualization with Streamlit
- Monitoring with Grafana

#!/bin/bash
set -e

# Initialize database
python -c "from data.db_manager import DatabaseManager; DatabaseManager('sqlite:///data/db/trading.db').init_db()"

# Start services based on the command
case "$1" in
    "dashboard")
        echo "Starting dashboard..."
        streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
        ;;
    "api")
        echo "Starting API server..."
        python main.py
        ;;
    "trading")
        echo "Starting trading system..."
        python run.py
        ;;
    *)
        echo "Starting all services..."
        python run.py & 
        python main.py &
        streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
        ;;
esac
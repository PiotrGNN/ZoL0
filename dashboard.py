"""
Trading system dashboard with real-time visualization
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, Any

from data.utils.database_manager import DatabaseManager
from data.data.market_data_fetcher import MarketDataFetcher
from utils.monitoring import TradingMetrics, HealthCheck
from utils.error_handling import handle_errors

# Initialize components
db = DatabaseManager("trading.db")  # Using direct path instead of URI format
market_data_fetcher = MarketDataFetcher(output_mode="db")
trading_metrics = TradingMetrics()
health_check = HealthCheck()

def initialize_session_state():
    """Initialize session state variables"""
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = '1h'
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

def create_candlestick_chart(data: pd.DataFrame):
    """Create interactive candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=data['timestamp'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close']
    )])
    fig.update_layout(
        title='Market Price Action',
        yaxis_title='Price',
        xaxis_title='Time'
    )
    return fig

def display_system_health():
    """Display system health metrics"""
    health_status = health_check.check_health()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "CPU Usage",
            f"{health_status['metrics']['cpu_percent']}%",
            delta=None,
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Memory Usage",
            f"{health_status['metrics']['memory_percent']}%",
            delta=None,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Disk Usage",
            f"{health_status['metrics']['disk_usage_percent']}%",
            delta=None,
            delta_color="inverse"
        )
    
    if health_status['warnings']:
        st.warning("System Warnings: " + ", ".join(health_status['warnings']))

def display_trading_metrics():
    """Display trading performance metrics"""
    try:
        metrics = trading_metrics.calculate_metrics()
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        metrics = {
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Win Rate",
            f"{metrics.get('win_rate', 0):.2%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Profit Factor",
            f"{metrics.get('profit_factor', 0):.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics.get('max_drawdown', 0):.2%}",
            delta=None,
            delta_color="inverse"
        )

def display_active_trades():
    """Display active trades table"""
    trades = db.get_trades(status="OPEN")
    if trades:
        trades_data = []
        for trade in trades:
            trades_data.append({
                'Symbol': trade['symbol'],
                'Direction': trade.get('direction', trade.get('side', 'UNKNOWN')),
                'Entry Price': f"${float(trade['entry_price']):,.2f}",
                'Current P/L': f"${float(trade.get('profit_loss', trade.get('pnl', 0))):,.2f}",
                'Entry Time': datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d %H:%M:%S") if trade['entry_time'] else 'N/A'
            })
        st.dataframe(pd.DataFrame(trades_data))
    else:
        st.info("No active trades")

def display_ai_models():
    """Display AI models status"""
    models = db.get_model_metadata()  # Remove the model_name parameter
    if models:
        models_data = []
        for model in models:
            metrics = json.loads(model['metrics']) if model.get('metrics') else {}
            models_data.append({
                'Model': model['model_name'],
                'Type': model['model_type'],
                'Accuracy': f"{metrics.get('accuracy', 0):.2%}",
                'Last Training': model['training_date'].strftime("%Y-%m-%d %H:%M:%S") if model['training_date'] else 'N/A'
            })
        st.dataframe(pd.DataFrame(models_data))
    else:
        st.info("No AI models found")

def main():
    st.set_page_config(
        page_title="Trading Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title("Settings")
    st.sidebar.selectbox(
        "Timeframe",
        ["1h", "4h", "1d", "1w"],
        key="selected_timeframe"
    )
    st.sidebar.checkbox("Auto-refresh", key="auto_refresh")
    
    # Main content
    st.title("Trading System Dashboard")
    
    # System health section
    st.subheader("System Health")
    display_system_health()
    
    # Trading metrics section
    st.subheader("Trading Performance")
    display_trading_metrics()
    
    # Market data visualization
    st.subheader("Market Data")
    try:
        market_data = market_data_fetcher.fetch_data(
            symbol="BTCUSDT",
            interval=st.session_state.selected_timeframe,
            limit=100
        )
        if not market_data.empty:
            st.plotly_chart(create_candlestick_chart(market_data), use_container_width=True)
        else:
            st.info("No market data available")
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
    
    # Active trades
    st.subheader("Active Trades")
    display_active_trades()
    
    # AI Models
    st.subheader("AI Models Status")
    display_ai_models()
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        st.experimental_rerun()

if __name__ == "__main__":
    main()

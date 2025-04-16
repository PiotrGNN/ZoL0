
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta

# Upewnij się, że katalogi istnieją
for directory in ["logs", "data/cache", "static/img"]:
    os.makedirs(directory, exist_ok=True)

# Dodanie lokalnych bibliotek do ścieżki Pythona
LOCAL_LIBS_DIR = "python_libs"
sys.path.insert(0, LOCAL_LIBS_DIR)

# Konfiguracja strony
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styl niestandardowy
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-18e3th9 {
        padding-top: 1rem;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 4px;
        padding: 10px 20px;
        margin-right: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
    }
    .css-18e3th9 h2 {
        margin-bottom: 10px;
    }
    .indicator-card {
        background-color: #1e2130; 
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .positive-value {
        color: #4CAF50;
        font-weight: bold;
    }
    .negative-value {
        color: #F44336;
        font-weight: bold;
    }
    .neutral-value {
        color: #FFC107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Funkcje pomocnicze
def get_api_data(endpoint, default_data=None):
    """Pobiera dane z API"""
    try:
        # Jeśli w Replit mamy własny serwer Flask na porcie 5000, to używamy go
        url = f"http://127.0.0.1:5000{endpoint}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Błąd API: {response.status_code} dla {endpoint}")
            return default_data
    except Exception as e:
        # W trybie developmentu używamy zapasowych danych
        logger.error(f"Błąd podczas pobierania {endpoint}: {e}")
        return default_data

# Nagłówek
st.title("🚀 Trading System Dashboard")

# Pasek boczny
with st.sidebar:
    st.header("Konfiguracja")
    
    # Sekcja - wybór pary walutowej
    st.subheader("Para walutowa")
    symbol = st.selectbox(
        "Wybierz parę",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
    )
    
    # Sekcja - wybór interwału
    st.subheader("Interwał")
    timeframe = st.selectbox(
        "Wybierz interwał",
        ["1m", "5m", "15m", "1h", "4h", "1d"]
    )
    
    # Sekcja - wybór strategii
    st.subheader("Strategia")
    strategy = st.selectbox(
        "Wybierz strategię",
        ["trend_following", "mean_reversion", "breakout", "ml_prediction"]
    )
    
    # Przyciski akcji
    with st.expander("Akcje", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Uruchom backtesting"):
                st.session_state.run_backtest = True
        with col2:
            if st.button("Symulacja handlu"):
                st.session_state.run_simulation = True
    
    # Status komponentów
    st.subheader("Status systemu")
    component_status = get_api_data("/api/component-status", {
        "api_status": "offline",
        "trading_status": "offline"
    })
    
    api_status = component_status.get("api_status", "offline")
    trading_status = component_status.get("trading_status", "offline")
    
    col1, col2 = st.columns(2)
    col1.metric("API", api_status, delta=None, delta_color="off")
    col2.metric("Trading", trading_status, delta=None, delta_color="off")

# Główne zakładki
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Wykresy", "🤖 AI Models", "📝 Raporty", "⚙️ Diagnostyka"])

# Zakładka 1: Dashboard
with tab1:
    # Pobieranie danych z API
    portfolio_data = get_api_data("/api/portfolio", {
        "balance": {"USDT": 1000},
        "positions": [],
        "equity": 1000,
        "available": 1000,
        "unrealized_pnl": 0
    })
    
    # Wyświetl dane portfela
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Całkowita wartość", f"${portfolio_data.get('equity', 0):.2f}")
    col2.metric("Dostępne środki", f"${portfolio_data.get('available', 0):.2f}")
    col3.metric("Niezrealizowany P/L", f"${portfolio_data.get('unrealized_pnl', 0):.2f}")
    
    # Wykres equity
    st.subheader("Wykres kapitału")
    
    # Symulowane dane dla wykresu
    chart_dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    chart_values = np.linspace(1000, portfolio_data.get('equity', 1000), len(chart_dates))
    chart_values = chart_values + np.random.normal(0, 20, size=len(chart_values)).cumsum()
    
    df_equity = pd.DataFrame({
        'date': chart_dates,
        'equity': chart_values
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_equity['date'], y=df_equity['equity'], 
                            mode='lines', name='Equity', 
                            line=dict(color='#4CAF50', width=2)))
    
    fig.update_layout(
        title='Historia kapitału',
        xaxis_title='Data',
        yaxis_title='Wartość ($)',
        template='plotly_dark',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Otwarte pozycje
    st.subheader("Otwarte pozycje")
    
    positions = portfolio_data.get('positions', [])
    if positions:
        position_data = []
        for pos in positions:
            position_data.append({
                "Symbol": pos.get('symbol'),
                "Typ": pos.get('side'),
                "Wielkość": pos.get('size'),
                "Cena wejścia": pos.get('entry_price'),
                "Aktualna cena": pos.get('mark_price'),
                "P/L": pos.get('unrealized_pnl'),
                "P/L %": pos.get('roe')
            })
        
        df_positions = pd.DataFrame(position_data)
        st.dataframe(df_positions, use_container_width=True)
    else:
        st.info("Brak otwartych pozycji")
    
    # Ostatnie transakcje
    st.subheader("Ostatnie transakcje")
    trades = get_api_data("/api/trades", {"trades": []}).get("trades", [])
    
    if trades:
        trade_data = []
        for trade in trades[:10]:  # Pokaż tylko 10 najnowszych
            trade_data.append({
                "Symbol": trade.get('symbol'),
                "Strona": trade.get('side'),
                "Wielkość": trade.get('quantity'),
                "Cena wejścia": trade.get('entry_price'),
                "Cena wyjścia": trade.get('exit_price', '-'),
                "P/L": trade.get('profit_loss', '-'),
                "P/L %": f"{trade.get('profit_loss_percent', '-')}%",
                "Status": trade.get('status')
            })
        
        df_trades = pd.DataFrame(trade_data)
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("Brak transakcji do wyświetlenia")

# Zakładka 2: Wykresy
with tab2:
    st.subheader(f"Wykres {symbol} ({timeframe})")
    
    # Pobierz dane wykresu
    chart_data = get_api_data(f"/api/chart-data?symbol={symbol}&timeframe={timeframe}", {
        "candles": [],
        "indicators": {}
    })
    
    # Utwórz wykres świecowy
    if "candles" in chart_data and chart_data["candles"]:
        df_candles = pd.DataFrame(chart_data["candles"])
        
        fig = go.Figure()
        
        # Dodaj świece
        fig.add_trace(go.Candlestick(
            x=df_candles['timestamp'],
            open=df_candles['open'],
            high=df_candles['high'],
            low=df_candles['low'],
            close=df_candles['close'],
            name='Świece'
        ))
        
        # Dodaj wskaźniki jeśli dostępne
        indicators = chart_data.get("indicators", {})
        for indicator_name, indicator_data in indicators.items():
            if indicator_data:
                df_indicator = pd.DataFrame(indicator_data)
                fig.add_trace(go.Scatter(
                    x=df_indicator['timestamp'],
                    y=df_indicator['value'],
                    mode='lines',
                    name=indicator_name
                ))
        
        # Aktualizuj układ wykresu
        fig.update_layout(
            title=f'{symbol} ({timeframe})',
            xaxis_title='Data',
            yaxis_title='Cena',
            template='plotly_dark',
            height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Brak danych dla wykresu")
    
    # Analiza techniczna
    st.subheader("Analiza techniczna")
    
    # Pobierz analizę rynku
    market_analysis = get_api_data(f"/api/market/analyze?symbol={symbol}&interval={timeframe}&strategy={strategy}", {
        "signal": "NEUTRAL",
        "confidence": 0.5,
        "indicators": {}
    })
    
    # Sygnał i pewność
    col1, col2 = st.columns(2)
    
    signal_text = market_analysis.get("signal", "NEUTRAL")
    signal_color = "green" if signal_text == "BUY" else "red" if signal_text == "SELL" else "gray"
    
    col1.markdown(f"<h3 style='color:{signal_color}'>Sygnał: {signal_text}</h3>", unsafe_allow_html=True)
    
    confidence = market_analysis.get("confidence", 0.5) * 100
    col2.progress(confidence / 100)
    col2.text(f"Pewność: {confidence:.1f}%")
    
    # Wskaźniki
    indicators = market_analysis.get("indicators", {})
    if indicators:
        st.markdown("### Wskaźniki")
        
        # Podziel na kolumny
        cols = st.columns(3)
        i = 0
        
        for indicator, value in indicators.items():
            col_idx = i % 3
            with cols[col_idx]:
                st.markdown(f"<div class='indicator-card'><b>{indicator}:</b> {value:.4f}</div>", unsafe_allow_html=True)
            i += 1
    
    # Sentiment
    st.subheader("Analiza sentymentu")
    sentiment_data = get_api_data("/api/sentiment", {
        "value": 0,
        "analysis": "Neutralny",
        "sources": {
            "twitter": 0,
            "news": 0,
            "forum": 0
        }
    })
    
    sentiment_value = sentiment_data.get("value", 0)
    sentiment_text = sentiment_data.get("analysis", "Neutralny")
    sentiment_color = "green" if sentiment_value > 0.2 else "red" if sentiment_value < -0.2 else "#FFC107"
    
    st.markdown(f"<h3 style='color:{sentiment_color}'>Sentyment: {sentiment_text} ({sentiment_value:.2f})</h3>", unsafe_allow_html=True)
    
    # Źródła sentymentu
    sources = sentiment_data.get("sources", {})
    if sources:
        cols = st.columns(len(sources))
        for i, (source, value) in enumerate(sources.items()):
            cols[i].metric(f"Sentyment - {source.capitalize()}", f"{value:.2f}")

# Zakładka 3: AI Models
with tab3:
    st.subheader("Modele AI")
    
    # Pobierz status modeli AI
    ai_models = get_api_data("/api/ai-models-status", {"models": []}).get("models", [])
    
    if ai_models:
        # Podziel modele na kolumny
        cols = st.columns(3)
        for i, model in enumerate(ai_models):
            col_idx = i % 3
            with cols[col_idx]:
                status_color = "green" if model.get("status") == "active" else "red"
                accuracy = model.get("accuracy", 0)
                accuracy_color = "green" if accuracy > 70 else "orange" if accuracy > 50 else "red"
                
                st.markdown(f"""
                <div style="background-color:#1e2130; padding:15px; border-radius:5px; margin-bottom:10px;">
                    <h4>{model.get("name")}</h4>
                    <p>Typ: {model.get("type", "Nieznany")}</p>
                    <p>Status: <span style="color:{status_color};">{model.get("status", "nieaktywny")}</span></p>
                    <p>Dokładność: <span style="color:{accuracy_color};">{accuracy}%</span></p>
                    <p>Ostatnia aktywność: {model.get("last_active", "Brak danych")}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Brak dostępnych modeli AI")
    
    # Wyniki uczenia
    st.subheader("Status uczenia AI")
    learning_status = get_api_data("/api/ai/learning-status", {
        "models_in_training": 0,
        "last_trained": "",
        "best_accuracy": 0,
        "training_progress": 0
    })
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Modele w treningu", learning_status.get("models_in_training", 0))
    col2.metric("Najlepsza dokładność", f"{learning_status.get('best_accuracy', 0)}%")
    
    # Postęp treningu
    training_progress = learning_status.get("training_progress", 0)
    st.progress(training_progress / 100)
    st.text(f"Postęp treningu: {training_progress}%")
    
    # Myśli AI
    st.subheader("Analiza AI")
    ai_thoughts = get_api_data("/api/ai/thoughts", {
        "thoughts": ["Brak danych analitycznych od modeli AI."]
    }).get("thoughts", ["Brak danych"])
    
    for thought in ai_thoughts:
        st.markdown(f"""
        <div style="background-color:#1e2130; padding:15px; border-radius:5px; margin-bottom:10px; font-style:italic;">
            "{thought}"
        </div>
        """, unsafe_allow_html=True)

# Zakładka 4: Raporty
with tab4:
    st.subheader("Wyniki symulacji")
    simulation_results = get_api_data("/api/simulation-results", {"results": []}).get("results", [])
    
    if simulation_results:
        # Wybór raportu
        selected_report = st.selectbox(
            "Wybierz raport",
            [result.get("id") for result in simulation_results]
        )
        
        # Wyświetl szczegóły wybranego raportu
        selected_data = next((result for result in simulation_results if result.get("id") == selected_report), None)
        
        if selected_data:
            # Podsumowanie
            st.markdown("### Podsumowanie symulacji")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Początkowy kapitał", f"${selected_data.get('initial_capital', 0):.2f}")
            col2.metric("Końcowy kapitał", f"${selected_data.get('final_capital', 0):.2f}")
            
            profit = selected_data.get('profit', 0)
            profit_pct = selected_data.get('profit_percentage', 0)
            profit_color = "normal" if profit >= 0 else "inverse"
            
            col3.metric("Zysk/Strata", f"${profit:.2f}", f"{profit_pct:.2f}%", delta_color=profit_color)
            col4.metric("Max Drawdown", f"{selected_data.get('max_drawdown', 0):.2f}%")
            
            # Statystyki
            st.markdown("### Statystyki")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Całkowite transakcje", selected_data.get('total_trades', 0))
            col2.metric("Wygrane", selected_data.get('winning_trades', 0))
            col3.metric("Przegrane", selected_data.get('losing_trades', 0))
            col4.metric("Win Rate", f"{selected_data.get('win_rate', 0)*100:.2f}%")
            
            # Wykres
            st.markdown("### Wykres symulacji")
            chart_path = selected_data.get('chart_path', '')
            
            if chart_path and os.path.exists(chart_path):
                st.image(chart_path)
            else:
                st.warning("Wykres nie jest dostępny")
            
            # Szczegóły transakcji
            st.markdown("### Transakcje")
            trades = selected_data.get('trades', [])
            
            if trades:
                trade_data = []
                for trade in trades:
                    trade_data.append({
                        "Timestamp": trade.get('timestamp'),
                        "Akcja": trade.get('action'),
                        "Cena": trade.get('price'),
                        "Wielkość": trade.get('size', 0),
                        "PnL": trade.get('pnl', '-'),
                        "Kapitał": trade.get('capital', 0)
                    })
                
                df_sim_trades = pd.DataFrame(trade_data)
                st.dataframe(df_sim_trades, use_container_width=True)
            else:
                st.info("Brak transakcji w raporcie")
    else:
        st.info("Brak dostępnych raportów symulacji")

# Zakładka 5: Diagnostyka
with tab5:
    st.subheader("Logi systemowe")
    
    # Pobierz logi
    logs = get_api_data("/api/logs", {"logs": []}).get("logs", [])
    
    # Filtrowanie logów
    log_level = st.selectbox("Poziom logów", ["ALL", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    # Filtruj logi według poziomu
    filtered_logs = logs
    if log_level != "ALL":
        filtered_logs = [log for log in logs if log.get("level") == log_level]
    
    # Wyświetl logi
    if filtered_logs:
        for log in filtered_logs:
            level = log.get("level", "INFO")
            color = {
                "INFO": "white",
                "WARNING": "orange",
                "ERROR": "red",
                "CRITICAL": "darkred"
            }.get(level, "white")
            
            st.markdown(f"""
            <div style="background-color:#1e2130; padding:10px; border-radius:5px; margin-bottom:5px;">
                <span style="color:{color};">[{level}]</span> {log.get('timestamp')}: {log.get('message')}
                {f"<div style='margin-top:5px; font-size:0.9em; color:#aaa;'>{log.get('details')}</div>" if log.get('details') else ""}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Brak logów do wyświetlenia")
    
    # Diagnostyka połączeń
    st.subheader("Status komponentów")
    
    # Pobierz status komponentów
    components = get_api_data("/api/status", {
        "status": "offline",
        "components": {
            "bybit_api": "offline",
            "binance_api": "offline",
            "ccxt": "offline",
            "strategy_manager": "offline",
            "model_recognizer": "offline",
            "anomaly_detector": "offline",
            "sentiment_analyzer": "offline"
        }
    }).get("components", {})
    
    # Wyświetl status komponentów
    col1, col2 = st.columns(2)
    
    for i, (component, status) in enumerate(components.items()):
        color = "green" if status == "online" else "red"
        if i % 2 == 0:
            col1.markdown(f"{component}: <span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
        else:
            col2.markdown(f"{component}: <span style='color:{color};'>{status}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    # To jest uruchamiane gdy uruchamiamy skrypt bezpośrednio z Streamlit
    # Streamlit automatycznie uruchamia tę aplikację, więc nie musimy
    # nic dodawać tutaj
    pass

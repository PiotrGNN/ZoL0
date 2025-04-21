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
import logging
from ai_models.sentiment_ai import SentimentAnalyzer
from ai_models.anomaly_detection import AnomalyDetector
from ai_models.model_recognition import ModelRecognizer

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Inicjalizacja komponentów
sentiment_analyzer = SentimentAnalyzer()
anomaly_detector = AnomalyDetector()
model_recognizer = ModelRecognizer()

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

# Styl niestandardowy z naprawionym menu przewijalnym oraz poprawionymi zakładkami
st.markdown("""
<style>
    /* Podstawowe style dla całego dashboardu */
    .main {
        background-color: #0e1117;
    }
    
    /* ===== NAPRAWIONE MENU Z PRZEWIJANIEM ===== */
    section[data-testid="stSidebar"] {
        position: relative;
        height: 100vh !important;
        background-color: #1e2130;
        z-index: 1;
        overflow-y: auto;
    }
    
    section[data-testid="stSidebar"] > div {
        height: 100vh !important;
        overflow-y: auto;
        padding-bottom: 5rem !important;
    }
    
    /* ===== NAPRAWIONE ZAKŁADKI ===== */
    /* Ulepszenie zakładek - poprawne wyświetlanie bez chowania w kontenerze */
    div.stTabs {
        background-color: #0e1117;
        width: 100%;
        overflow-x: auto !important;
        display: flex;
        flex-direction: column;
    }
    
    div.stTabs > div:first-child {
        width: 100%;
        overflow-x: auto !important;
        max-width: 100% !important;
        padding-bottom: 5px; /* Dodaje odstęp na dole */
    }
    
    /* Ustawienie minimalnej szerokości dla zakładek */
    button[role="tab"] {
        min-width: 120px !important;
        white-space: nowrap !important;
        text-align: center;
        padding: 10px 15px !important;
        margin-right: 5px !important;
        background-color: #1e2130;
    }
    
    /* Aktywna zakładka */
    button[role="tab"][aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    /* Lista zakładek */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        padding-bottom: 5px;
    }
    
    /* Poprawa wyglądu elementów w menu */
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Zapewnia, że wszystkie przyciski w menu są widoczne */
    section[data-testid="stSidebar"] button {
        margin-bottom: 0.5rem;
        width: 100%;
    }
    
    /* Poprawia wyświetlanie głównego kontenera */
    .main .block-container {
        padding-top: 2rem !important;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* ===== NAPRAWA WYŚWIETLANIA WYKRESÓW ===== */
    /* Upewnia się, że wykresy są w pełni widoczne */
    .stPlotlyChart {
        width: 100% !important;
    }
    
    .js-plotly-plot, .plotly, .plot-container {
        width: 100% !important;
    }
    
    /* Poprawa stylów dla tabeli danych */
    .dataframe-container {
        width: 100% !important;
        overflow-x: auto !important;
    }
    
    .stDataFrame {
        width: 100% !important;
    }
    
    /* ===== STYLE DLA FORMULARZA LOGOWANIA ===== */
    .login-form-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 2rem;
        background-color: #1e2130;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Style dla pozostałych elementów UI */
    .card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Style dla karty statusu */
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .status-online {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .status-offline {
        color: #F44336;
        font-weight: bold;
    }
    
    .status-warning {
        color: #FF9800;
        font-weight: bold;
    }
    
    /* Naprawa responsywności */
    @media (max-width: 992px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        button[role="tab"] {
            min-width: 100px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Funkcje pomocnicze
def make_api_request(method, url, data=None, headers=None, max_retries=3, retry_delay=1):
    """
    Wykonuje żądanie API z obsługą powtórzeń i błędów
    """
    import time
    from requests.exceptions import ConnectionError, RequestException
    
    headers = headers or {}
    attempt = 0
    
    while attempt < max_retries:
        try:
            if method.lower() == 'get':
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, json=data, headers=headers)
                
            response.raise_for_status()
            return response
            
        except ConnectionError as e:
            attempt += 1
            if attempt == max_retries:
                logger.error(f"Nie można połączyć się z API po {max_retries} próbach: {e}")
                raise
            logger.warning(f"Próba połączenia {attempt}/{max_retries} nie powiodła się, ponawiam za {retry_delay}s")
            time.sleep(retry_delay)
            
        except RequestException as e:
            logger.error(f"Błąd podczas żądania API: {e}")
            raise

# Zmodyfikuj istniejącą funkcję get_api_data
def get_api_data(endpoint, default_data=None):
    """Pobiera dane z API z obsługą błędów i domyślnymi danymi w przypadku niepowodzenia"""
    try:
        headers = {'Authorization': f'Bearer {st.session_state.get("token")}'}
        response = make_api_request('get', f'http://localhost:5002{endpoint}', headers=headers)
        return response.json()
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych z API {endpoint}: {e}")
        return default_data

# Dodaj funkcję do wykonywania żądań POST
def post_api_data(endpoint, data):
    """Wysyła dane do API z obsługą błędów"""
    try:
        headers = {'Authorization': f'Bearer {st.session_state.get("token")}'}
        response = make_api_request('post', f'http://localhost:5002{endpoint}', data=data, headers=headers)
        return response.json()
    except Exception as e:
        logger.error(f"Błąd podczas wysyłania danych do API {endpoint}: {e}")
        return {"success": False, "error": str(e)}

def format_trade_data(trades):
    """Formatuje dane o transakcjach do wyświetlenia"""
    trade_data = []
    for trade in trades:
        # Konwersja wartości NA na None dla pandas
        exit_price = None if trade.get('exit_price', '-') == '-' else trade.get('exit_price')
        pnl = None if trade.get('profit_loss', '-') == '-' else float(trade.get('profit_loss', 0))
        pnl_percent = None if trade.get('profit_loss_percent', '-') == '-' else float(trade.get('profit_loss_percent', 0))
        
        trade_data.append({
            "Symbol": trade.get('symbol'),
            "Strona": trade.get('side'),
            "Wielkość": float(trade.get('quantity', 0)),
            "Cena wejścia": float(trade.get('entry_price', 0)),
            "Cena wyjścia": exit_price,
            "P/L": pnl,
            "P/L %": f"{pnl_percent}%" if pnl_percent is not None else '-',
            "Status": trade.get('status', 'Nieznany')
        })
    
    return pd.DataFrame(trade_data)

# Nagłówek
st.title("🚀 Trading System Dashboard")

# Dodaj system logowania
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <style>
    .main-container {
        max-width: 500px !important;
        margin: 0 auto !important;
        padding-top: 2rem;
    }
    .login-container {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        margin-top: 50px;
    }
    .css-1avcm0n {
        background-color: #4CAF50 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<h2 style='text-align: center;'>Logowanie do systemu</h2>", unsafe_allow_html=True)
        
        username = st.text_input("Nazwa użytkownika", key="login_username")
        password = st.text_input("Hasło", type="password", key="login_password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Zaloguj"):
                # Zmodyfikowana logika logowania, która nie wymaga API
                if username == "admin" and password == "admin":
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.token = "mock_token_for_development"  # Dodajemy sztuczny token
                    st.session_state.token_expires = time.time() + 3600  # Ważny przez godzinę
                    st.rerun()
                else:
                    st.error("Nieprawidłowe dane logowania!")
                    
        with col2:
            if st.button("Rejestracja"):
                st.session_state.show_register = True
        
        if st.session_state.get("show_register", False):
            st.markdown("<h3 style='text-align: center; margin-top: 20px;'>Rejestracja nowego konta</h3>", unsafe_allow_html=True)
            
            new_username = st.text_input("Nowa nazwa użytkownika", key="reg_username")
            new_email = st.text_input("Adres email", key="reg_email")
            new_password = st.text_input("Nowe hasło", type="password", key="reg_password")
            confirm_password = st.text_input("Potwierdź hasło", type="password", key="reg_confirm")
            
            if st.button("Utwórz konto"):
                if not new_username or not new_email or not new_password:
                    st.error("Wszystkie pola są wymagane!")
                elif new_password != confirm_password:
                    st.error("Hasła nie są identyczne!")
                else:
                    # Tutaj dodać rzeczywistą rejestrację
                    st.success("Konto utworzone pomyślnie! Możesz się teraz zalogować.")
                    st.session_state.show_register = False
    
    st.stop()

# Dodanie tokenu JWT po zalogowaniu - zastąpienie oryginalnego kodu, który próbuje się łączyć z API
if "token" not in st.session_state and st.session_state.authenticated:
    # Używamy lokalnego tokenu zamiast próbować połączyć się z API
    st.session_state.token = "mock_token_for_development"
    st.session_state.token_expires = time.time() + 3600  # Ważny przez godzinę
    
# Funkcja pomocnicza do wykonywania zapytań API z tokenem
def api_request(endpoint, method="GET", data=None):
    """Wykonuje zapytanie API z tokenem JWT"""
    url = f"http://localhost:5002{endpoint}"
    headers = {}
    
    if "token" in st.session_state:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, headers=headers)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            return None
            
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            # Token wygasł - odśwież
            if "token" in st.session_state:
                refresh_token()
                return api_request(endpoint, method, data)  # Spróbuj ponownie po odświeżeniu
        
        return None
    except Exception as e:
        logger.error(f"Błąd podczas wykonywania zapytania API: {e}")
        return None

def refresh_token():
    """Odświeża token JWT (wersja deweloperska)"""
    try:
        # Zamiast łączyć się z API, po prostu odświeżamy lokalny token
        st.session_state.token = "mock_token_for_development"
        st.session_state.token_expires = time.time() + 3600
        return True
    except Exception as e:
        logger.error(f"Błąd podczas odświeżania tokenu: {e}")
        return False

# Reszta kodu dashboardu wykonuje się tylko dla zalogowanych użytkowników
# Dodaj przycisk wylogowania w prawym górnym rogu
logout_placeholder = st.sidebar.empty()
if logout_placeholder.button("Wyloguj"):
    st.session_state.authenticated = False
    if "token" in st.session_state:
        del st.session_state.token
    st.rerun()

# Pokaż nazwę użytkownika
st.sidebar.markdown(f"**Zalogowany jako:** {st.session_state.get('username', 'Użytkownik')}")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Dashboard", "📈 Wykresy", "🤖 AI Models", 
    "📝 Raporty", "⚙️ Diagnostyka", "🎯 Risk Manager",
    "🔍 Portfolio Analytics", "🔔 Powiadomienia", "🧠 Autonomiczny AI"
])

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
    col4.metric("Max Drawdown", "0.00%")  # Wartość domyślna zamiast odwołania do selected_data
    
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
        df_trades = format_trade_data(trades[:10])  # Pokaż tylko 10 najnowszych
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
    col1.metric("Modele w treningu", learning_status.get('models_in_training', 0))
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
                    # Konwersja wartości do odpowiednich typów
                    pnl = str(trade.get('pnl', 0)) if trade.get('pnl') is not None else '-'
                    trade_data.append({
                        "Timestamp": trade.get('timestamp'),
                        "Akcja": trade.get('action'),
                        "Cena": float(trade.get('price', 0)),
                        "Wielkość": float(trade.get('size', 0)),
                        "PnL": pnl,
                        "Kapitał": float(trade.get('capital', 0))
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

# Zakładka 6: Risk Manager
with tab6:
    st.subheader("Zarządzanie Ryzykiem")
    
    # Metryki ryzyka
    risk_metrics = get_api_data("/api/risk/metrics", {
        "metrics": []
    }).get("metrics", [])
    
    if risk_metrics and len(risk_metrics) > 0:
        latest_metrics = risk_metrics[-1]
        
        # Wskaźniki w kartach
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Sharpe Ratio", 
            f"{latest_metrics.get('sharpe_ratio', 0):.2f}",
            delta=None
        )
        col2.metric(
            "Sortino Ratio",
            f"{latest_metrics.get('sortino_ratio', 0):.2f}",
            delta=None
        )
        col3.metric(
            "Max Drawdown",
            f"{latest_metrics.get('max_drawdown', 0):.2%}",
            delta=None
        )
        col4.metric(
            "Win Rate",
            f"{latest_metrics.get('win_rate', 0):.1%}",
            delta=None
        )
        
        # Wykres metryk w czasie
        st.subheader("Historia metryk")
        metrics_df = pd.DataFrame(risk_metrics)
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        
        fig = go.Figure()
        
        # Dodaj linie dla każdej metryki
        metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']
        for metric in metrics_to_plot:
            fig.add_trace(go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df[metric],
                name=metric.replace('_', ' ').title(),
                mode='lines'
            ))
            
        fig.update_layout(
            title='Historia metryk ryzyka',
            xaxis_title='Data',
            yaxis_title='Wartość',
            template='plotly_dark',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Brak dostępnych danych metryk ryzyka")
    
    # Limity ryzyka
    st.subheader("Limity ryzyka")
    
    risk_limits = get_api_data("/api/risk/limits", {
        "limits": {}
    }).get("limits", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_stop_loss = st.number_input(
            "Stop Loss Portfela (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(risk_limits.get('portfolio_stop_loss', 10.0)),
            step=0.1
        )
        
        max_position_size = st.number_input(
            "Maksymalny rozmiar pozycji (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(risk_limits.get('max_position_size', 5.0)),
            step=0.1
        )
        
    with col2:
        max_daily_trades = st.number_input(
            "Maksymalna liczba transakcji dziennie",
            min_value=1,
            max_value=100,
            value=int(risk_limits.get('max_daily_trades', 10))
        )
        
        max_daily_drawdown = st.number_input(
            "Maksymalny dzienny drawdown (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(risk_limits.get('max_daily_drawdown', 5.0)),
            step=0.1
        )
    
    if st.button("Zapisz limity"):
        new_limits = {
            'portfolio_stop_loss': portfolio_stop_loss,
            'max_position_size': max_position_size,
            'max_daily_trades': max_daily_trades,
            'max_daily_drawdown': max_daily_drawdown
        }
        
        response = requests.post(
            'http://localhost:5000/api/risk/limits',
            json=new_limits,
            headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
        )
        
        if response.status_code == 200:
            st.success("Limity zostały zaktualizowane")
        else:
            st.error("Błąd podczas aktualizacji limitów")
    
    # Kalkulator pozycji
    st.subheader("Kalkulator pozycji")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.selectbox(
            "Symbol",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
        )
        entry_price = st.number_input("Cena wejścia", min_value=0.0, value=0.0)
        
    with col2:
        stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0)
        risk_per_trade = st.number_input(
            "Ryzyko na transakcję (%)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        
    with col3:
        current_capital = st.number_input(
            "Dostępny kapitał",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )
        
    if st.button("Oblicz pozycję"):
        calc_data = {
            'symbol': symbol,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk_per_trade': risk_per_trade / 100,
            'current_capital': current_capital
        }
        
        response = requests.post(
            'http://localhost:5000/api/risk/position-calculator',
            json=calc_data,
            headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
        )
        
        if response.status_code == 200:
            result = response.json()['calculation']
            if result['success']:
                st.info(f"""
                Zalecany rozmiar pozycji: {result['position_size']:.4f}
                Wartość pozycji: ${result['position_value']:.2f}
                Ryzyko: ${result['risk_amount']:.2f}
                """)
            else:
                st.error(result['error'])
        else:
            st.error("Błąd podczas obliczania pozycji")

    # Rebalancing portfela
    st.subheader("Rebalancing portfela")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Aktualna alokacja")
        current_allocation = get_api_data("/api/portfolio/allocation", {
            "allocation": {}
        }).get("allocation", {})
        
        if current_allocation:
            # Wykres kołowy aktualnej alokacji
            fig = go.Figure(data=[go.Pie(
                labels=list(current_allocation.keys()),
                values=list(current_allocation.values()),
                hole=.3
            )])
            fig.update_layout(
                title='Aktualna alokacja',
                template='plotly_dark',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brak danych o alokacji portfela")
    
    with col2:
        st.markdown("### Optymalna alokacja")
        risk_profile = st.selectbox(
            "Profil ryzyka",
            ["konserwatywny", "umiarkowany", "agresywny"]
        )
        
        if st.button("Oblicz optymalną alokację"):
            optimal_allocation = get_api_data(
                f"/api/portfolio/optimal-allocation?risk_profile={risk_profile}",
                {"allocation": {}}
            ).get("allocation", {})
            
            if optimal_allocation:
                fig = go.Figure(data=[go.Pie(
                    labels=list(optimal_allocation.keys()),
                    values=list(optimal_allocation.values()),
                    hole=.3
                )])
                fig.update_layout(
                    title='Rekomendowana alokacja',
                    template='plotly_dark',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Zastosuj rekomendowaną alokację"):
                    response = requests.post(
                        'http://localhost:5000/api/portfolio/allocation',
                        json={"allocation": optimal_allocation},
                        headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                    )
                    if response.status_code == 200:
                        st.success("Zaktualizowano alokację portfela")
                    else:
                        st.error("Błąd podczas aktualizacji alokacji")
            else:
                st.warning("Nie można obliczyć optymalnej alokacji")
    
    # Korelacje aktywów
    st.subheader("Analiza korelacji")
    correlation_data = get_api_data("/api/portfolio/correlation", {
        "correlation_matrix": {},
        "high_correlation_pairs": []
    })
    
    if correlation_data.get("correlation_matrix"):
        # Mapa cieplna korelacji
        corr_matrix = pd.DataFrame(correlation_data["correlation_matrix"])
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu'
        ))
        fig.update_layout(
            title='Mapa korelacji aktywów',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Wyświetl pary o wysokiej korelacji
        if correlation_data["high_correlation_pairs"]:
            st.warning("Wykryto silnie skorelowane pary aktywów:")
            for pair in correlation_data["high_correlation_pairs"]:
                st.markdown(f"""
                * {pair['symbol1']} - {pair['symbol2']}: {pair['correlation']:.2f}
                """)
    else:
        st.info("Brak danych do analizy korelacji")
    
    # Szczegółowe statystyki
    st.subheader("Szczegółowe statystyki tradingowe")
    period = st.selectbox(
        "Okres analizy",
        ["7 dni", "30 dni", "90 dni", "180 dni", "365 dni"],
        index=1
    )
    
    days = int(period.split()[0])
    stats = get_api_data(f"/api/trading/statistics?days={days}", {})
    
    if stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Średni czas trzymania", f"{stats.get('avg_holding_time_hours', 0):.1f}h")
            st.metric("Największy zysk", f"${stats.get('largest_profit', 0):.2f}")
        
        with col2:
            st.metric("Średni zysk", f"${stats.get('avg_profit', 0):.2f}")
            st.metric("Średnia strata", f"${stats.get('avg_loss', 0)::.2f}")
        
        with col3:
            st.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
            st.metric("Całkowite transakcje", stats.get('total_trades', 0))
        
        # Najlepsze/najgorsze transakcje
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Najlepsze transakcje")
            best_trades = stats.get('best_trades', [])
            for trade in best_trades:
                st.markdown(f"""
                * {trade['symbol']}: ${trade['pnl']:.2f} ({trade['date']})
                """)
        
        with col2:
            st.markdown("### Najgorsze transakcje")
            worst_trades = stats.get('worst_trades', [])
            for trade in worst_trades:
                st.markdown(f"""
                * {trade['symbol']}: ${trade['pnl']:.2f} ({trade['date']})
                """)
    else:
        st.info("Brak danych statystycznych dla wybranego okresu")

# Zakładka 7: Portfolio Analytics
with tab7:
    st.subheader("Zaawansowana Analiza Portfela")
    
    # Metryki dywersyfikacji
    st.markdown("### Metryki Dywersyfikacji")
    diversification_metrics = get_api_data("/api/portfolio/analytics/diversification", {"metrics": {}})
    
    if diversification_metrics.get("metrics"):
        metrics = diversification_metrics["metrics"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Indeks Herfindahla", 
                f"{metrics['herfindahl_index']:.4f}",
                help="Miara koncentracji portfela. Niższe wartości oznaczają lepszą dywersyfikację."
            )
        
        with col2:
            st.metric(
                "Efektywna Liczba Aktywów",
                f"{metrics['effective_n']:.1f}",
                help="Efektywna liczba niezależnych pozycji w portfelu."
            )
        
        with col3:
            st.metric(
                "Różnorodność Klas Aktywów",
                f"{metrics['asset_class_diversity']:.2f}",
                help="Miara różnorodności klas aktywów w portfelu."
            )
    
    # Historia alokacji
    st.markdown("### Historia Alokacji")
    days = st.slider("Okres analizy (dni)", 7, 90, 30)
    allocation_history = get_api_data(f"/api/portfolio/analytics/allocation?days={days}", {"data": {}})
    
    if allocation_history.get("data"):
        df_allocation = pd.DataFrame(allocation_history["data"])
        df_allocation.index = pd.to_datetime(df_allocation.index)
        
        fig = go.Figure()
        for column in df_allocation.columns:
            fig.add_trace(go.Scatter(
                x=df_allocation.index,
                y=df_allocation[column],
                name=column,
                stackgroup='one',
                mode='lines'
            ))
        
        fig.update_layout(
            title='Rozkład Alokacji w Czasie',
            xaxis_title='Data',
            yaxis_title='Alokacja (%)',
            template='plotly_dark',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Ekspozycja na ryzyko
    st.markdown("### Ekspozycja na Ryzyko")
    risk_exposure = get_api_data("/api/portfolio/analytics/risk", {"risk_metrics": {}})
    
    if risk_exposure.get("risk_metrics"):
        metrics = risk_exposure["risk_metrics"]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Ryzyko Rynkowe",
                f"{metrics['market_risk']*100:.1f}%",
                help="Annualizowana zmienność portfela"
            )
            st.metric(
                "VaR (95%)",
                f"${metrics['var_95']:,.2f}",
                help="Value at Risk przy 95% poziomie ufności"
            )
        
        with col2:
            st.metric(
                "CVaR (95%)",
                f"${metrics['cvar_95']:,.2f}",
                help="Conditional Value at Risk"
            )
            st.metric(
                "Ryzyko Koncentracji",
                f"{metrics['concentration_risk']*100:.1f}%",
                help="Miara koncentracji ryzyka w portfelu"
            )
        
        with col3:
            st.metric(
                "Ryzyko Systematyczne",
                f"{metrics['systematic_risk']*100:.1f}%",
                help="Ryzyko związane z rynkiem"
            )
            st.metric(
                "Ryzyko Niesystematyczne",
                f"{metrics['unsystematic_risk']*100:.1f}%",
                help="Ryzyko specyficzne dla aktywów"
            )
    
    # Rotacja kapitału
    st.markdown("### Analiza Rotacji Kapitału")
    period_days = st.selectbox("Okres analizy", [30, 60, 90, 180, 365], index=0)
    turnover_metrics = get_api_data(f"/api/portfolio/analytics/turnover?days={period_days}", {"metrics": {}})
    
    if turnover_metrics.get("metrics"):
        metrics = turnover_metrics["metrics"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Wskaźnik Obrotu",
                f"{metrics['turnover_ratio']*100:.1f}%",
                help="Stosunek wartości transakcji do średniej wartości portfela"
            )
        
        with col2:
            st.metric(
                "Koszty Transakcyjne",
                f"${metrics['trading_costs']:,.2f}",
                help="Szacowane koszty transakcyjne w okresie"
            )
        
        with col3:
            st.metric(
                "Średnia Wartość Portfela",
                f"${metrics['portfolio_value']:,.2f}",
                help="Średnia wartość portfela w analizowanym okresie"
            )
    
    # Optymalizacja portfela
    st.markdown("### Optymalizacja Portfela")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_tolerance = st.slider(
            "Tolerancja ryzyka",
            min_value=0.01,
            max_value=0.10,
            value=0.02,
            step=0.01,
            format="%.2f",
            help="Docelowy poziom ryzyka portfela (odchylenie standardowe)"
        )
    
    with col2:
        if st.button("Optymalizuj Alokację"):
            with st.spinner("Obliczam optymalną alokację..."):
                response = requests.post(
                    'http://localhost:5000/api/portfolio/optimize',
                    json={"risk_tolerance": risk_tolerance},
                    headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("allocation"):
                        st.success("Obliczono optymalną alokację")
                        
                        # Wykres optymalnej alokacji
                        fig = go.Figure(data=[go.Pie(
                            labels=list(data["allocation"].keys()),
                            values=list(data["allocation"].values()),
                            hole=.3
                        )])
                        
                        fig.update_layout(
                            title='Rekomendowana Alokacja',
                            template='plotly_dark',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tabela z dokładnymi wartościami
                        df_allocation = pd.DataFrame([
                            {"Aktywo": k, "Alokacja (%)": f"{v*100:.1f}%"}
                            for k, v in data["allocation"].items()
                        ])
                        st.table(df_allocation)
                    else:
                        st.warning("Nie można obliczyć optymalnej alokacji")
                else:
                    st.error("Błąd podczas optymalizacji portfela")

# Zakładka 8: Powiadomienia i Alerty
with tab8:
    st.subheader("Zarządzanie Powiadomieniami")
    
    # Konfiguracja kanałów komunikacji
    st.markdown("### Kanały Komunikacji")
    
    # Pobierz aktualne ustawienia kanałów
    notification_settings = get_api_data("/api/notifications/settings", {"settings": []})
    
    if notification_settings.get("settings"):
        settings = notification_settings["settings"]
        
        # Email
        with st.expander("Konfiguracja Email", expanded=False):
            email_config = next((s for s in settings if s["channel_type"] == "email"), None)
            
            email_active = st.checkbox(
                "Aktywny",
                value=email_config["is_active"] if email_config else False,
                key="email_active"
            )
            
            email_sender = st.text_input(
                "Adres nadawcy",
                value=json.loads(email_config["config_json"])["sender"] if email_config else "",
                key="email_sender"
            )
            
            email_smtp = st.text_input(
                "Serwer SMTP",
                value=json.loads(email_config["config_json"])["smtp_server"] if email_config else "",
                key="email_smtp"
            )
            
            email_port = st.number_input(
                "Port SMTP",
                value=json.loads(email_config["config_json"])["smtp_port"] if email_config else 587,
                key="email_port"
            )
            
            email_user = st.text_input(
                "Nazwa użytkownika SMTP",
                value=json.loads(email_config["config_json"])["username"] if email_config else "",
                key="email_user",
                type="password"
            )
            
            email_pass = st.text_input(
                "Hasło SMTP",
                value=json.loads(email_config["config_json"])["password"] if email_config else "",
                key="email_pass",
                type="password"
            )
            
            if st.button("Zapisz konfigurację email"):
                email_settings = {
                    "channel_type": "email",
                    "settings": {
                        "is_active": email_active,
                        "config": {
                            "sender": email_sender,
                            "smtp_server": email_smtp,
                            "smtp_port": email_port,
                            "username": email_user,
                            "password": email_pass
                        }
                    }
                }
                
                response = requests.post(
                    'http://localhost:5000/api/notifications/settings',
                    json=email_settings,
                    headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                )
                
                if response.status_code == 200:
                    st.success("Konfiguracja email została zaktualizowana")
                else:
                    st.error("Błąd podczas aktualizacji konfiguracji email")
        
        # Telegram
        with st.expander("Konfiguracja Telegram", expanded=False):
            telegram_config = next((s for s in settings if s["channel_type"] == "telegram"), None)
            
            telegram_active = st.checkbox(
                "Aktywny",
                value=telegram_config["is_active"] if telegram_config else False,
                key="telegram_active"
            )
            
            telegram_token = st.text_input(
                "Token bota",
                value=json.loads(telegram_config["config_json"])["bot_token"] if telegram_config else "",
                key="telegram_token",
                type="password"
            )
            
            telegram_chat = st.text_input(
                "ID czatu",
                value=json.loads(telegram_config["config_json"])["chat_id"] if telegram_config else "",
                key="telegram_chat"
            )
            
            if st.button("Zapisz konfigurację Telegram"):
                telegram_settings = {
                    "channel_type": "telegram",
                    "settings": {
                        "is_active": telegram_active,
                        "config": {
                            "bot_token": telegram_token,
                            "chat_id": telegram_chat
                        }
                    }
                }
                
                response = requests.post(
                    'http://localhost:5000/api/notifications/settings',
                    json=telegram_settings,
                    headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                )
                
                if response.status_code == 200:
                    st.success("Konfiguracja Telegram została zaktualizowana")
                else:
                    st.error("Błąd podczas aktualizacji konfiguracji Telegram")
    
    # Testowanie powiadomień
    st.markdown("### Test Powiadomień")
    col1, col2 = st.columns(2)
    
    with col1:
        test_channel = st.selectbox(
            "Wybierz kanał",
            ["email", "telegram"]
        )
    
    with col2:
        if st.button("Wyślij powiadomienie testowe"):
            response = requests.post(
                'http://localhost:5000/api/notifications/test',
                json={"channel_type": test_channel},
                headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
            )
            
            if response.status_code == 200:
                st.success("Powiadomienie testowe zostało wysłane")
            else:
                st.error("Błąd podczas wysyłania powiadomienia testowego")
    
    # Historia powiadomień
    st.markdown("### Historia Powiadomień")
    
    notification_history = get_api_data("/api/notifications/history", {"history": []})
    
    if notification_history.get("history"):
        history = notification_history["history"]
        
        # Filtrowanie
        col1, col2 = st.columns(2)
        
        with col1:
            channel_filter = st.multiselect(
                "Filtruj po kanale",
                ["email", "telegram"],
                default=["email", "telegram"]
            )
        
        with col2:
            status_filter = st.multiselect(
                "Filtruj po statusie",
                ["delivered", "failed"],
                default=["delivered", "failed"]
            )
        
        # Filtruj historię
        filtered_history = [
            h for h in history 
            if h["channel"] in channel_filter and h["status"] in status_filter
        ]
        
        for notification in filtered_history:
            status_color = "green" if notification["status"] == "delivered" else "red"
            
            st.markdown(f"""
            <div style="background-color:#1e2130; padding:10px; border-radius:5px; margin-bottom:5px;">
                <span style="color:{status_color};">●</span> <b>{notification["channel"]}</b> - {notification["timestamp"]}
                <br><i>{notification["message"]}</i>
                {f'<br><span style="color:red;">Błąd: {notification["error_message"]}</span>' if notification["error_message"] else ""}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Brak historii powiadomień")
    
    # Zarządzanie alertami
    st.markdown("### Alerty")
    
    # Dodawanie nowego alertu
    with st.expander("Dodaj Nowy Alert", expanded=False):
        alert_type = st.selectbox(
            "Typ alertu",
            ["price", "pattern", "anomaly", "custom"]
        )
        
        if alert_type == "price":
            symbol = st.text_input("Symbol")
            condition = st.selectbox(
                "Warunek",
                ["above", "below", "cross_up", "cross_down"]
            )
            value = st.number_input("Wartość", step=0.0001)
            
            if st.button("Dodaj Alert Cenowy"):
                response = requests.post(
                    'http://localhost:5000/api/alerts',
                    json={
                        "type": alert_type,
                        "symbol": symbol,
                        "condition": condition,
                        "value": value
                    },
                    headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                )
                
                if response.status_code == 200:
                    st.success("Alert został dodany")
                else:
                    st.error("Błąd podczas dodawania alertu")
        
        elif alert_type == "pattern":
            symbol = st.text_input("Symbol")
            pattern = st.selectbox(
                "Wzorzec",
                ["bullish_flag", "bearish_flag", "double_top", "double_bottom"]
            )
            timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "1h", "4h", "1d"]
            )
            
            if st.button("Dodaj Alert Wzorca"):
                response = requests.post(
                    'http://localhost:5000/api/alerts/pattern',
                    json={
                        "symbol": symbol,
                        "pattern": pattern,
                        "timeframe": timeframe
                    },
                    headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                )
                
                if response.status_code == 200:
                    st.success("Alert wzorca został dodany")
                else:
                    st.error("Błąd podczas dodawania alertu wzorca")
        
        elif alert_type == "anomaly":
            symbol = st.text_input("Symbol")
            severity = st.slider(
                "Minimalna ważność",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1
            )
            
            if st.button("Dodaj Alert Anomalii"):
                response = requests.post(
                    'http://localhost:5000/api/alerts/anomaly',
                    json={
                        "symbol": symbol,
                        "min_severity": severity
                    },
                    headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                )
                
                if response.status_code == 200:
                    st.success("Alert anomalii został dodany")
                else:
                    st.error("Błąd podczas dodawania alertu anomalii")
    
    # Lista aktywnych alertów
    active_alerts = get_api_data("/api/alerts/active", {"alerts": []})
    
    if active_alerts.get("alerts"):
        alerts = active_alerts["alerts"]
        
        for alert in alerts:
            # Upewnij się, że alert jest słownikiem, a nie stringiem
            if isinstance(alert, str):
                try:
                    alert = json.loads(alert)
                except:
                    continue
                
            col1, col2 = st.columns([0.9, 0.1])
            
            with col1:
                st.markdown(f"""
                <div style="background-color:#1e2130; padding:10px; border-radius:5px;">
                    <b>{alert.get("type", "Unknown").title()}</b> - {alert.get("symbol", "N/A")}
                    <br>{alert.get("condition", "")} {alert.get("value", "")}
                    <br><small>Utworzono: {alert.get("created_at", "")}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("❌", key=f"delete_alert_{alert.get('id', '')}"):
                    response = requests.delete(
                        f'http://localhost:5000/api/alerts/{alert["id"]}',
                        headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                    )
                    
                    if response.status_code == 200:
                        st.success("Alert został usunięty")
                        st.experimental_rerun()
                    else:
                        st.error("Błąd podczas usuwania alertu")
    else:
        st.info("Brak aktywnych alertów")

# Nowa zakładka 9: Autonomiczny AI
with tab9:
    st.subheader("🧠 Autonomiczny system AI")
    
    # Status autonomii
    autonomous_status = get_api_data("/api/autonomous/status", {
        "autonomous_mode": False,
        "models": {},
        "last_decision": None
    })
    
    col1, col2 = st.columns(2)
    with col1:
        autonomous_mode = autonomous_status.get("autonomous_mode", False)
        st.markdown(f"""
        <div style="background-color: {'#1e5631' if autonomous_mode else '#561e1e'}; padding: 15px; border-radius:5px;">
            <h3 style="margin: 0;">Status: {'AKTYWNY' if autonomous_mode else 'NIEAKTYWNY'}</h3>
            <p>{'System działa w trybie w pełni autonomicznym' if autonomous_mode else 'System wymaga zatwierdzenia przez człowieka'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if autonomous_mode:
            if st.button("Wyłącz tryb autonomiczny"):
                response = requests.post(
                    'http://localhost:5000/api/autonomous/mode',
                    json={"enabled": False},
                    headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                )
                if response.status_code == 200:
                    st.success("Tryb autonomiczny został wyłączony")
                    st.experimental_rerun()
        else:
            if st.button("Włącz tryb autonomiczny"):
                response = requests.post(
                    'http://localhost:5000/api/autonomous/mode',
                    json={"enabled": True},
                    headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                )
                if response.status_code == 200:
                    st.success("Tryb autonomiczny został włączony")
                    st.experimental_rerun()
    
    # Panel decyzyjny Meta-Agenta
    st.subheader("🤖 Decyzje Meta-Agenta")
    
    # Pobierz ostatnie decyzje
    decisions = get_api_data("/api/autonomous/decisions", {"decisions": []}).get("decisions", [])
    
    if decisions:
        for decision in decisions:
            # Definiuj kolory na podstawie decyzji
            decision_color = {
                "BUY": "#4CAF50",   # zielony
                "SELL": "#F44336",  # czerwony
                "HOLD": "#FFC107"   # żółty
            }.get(decision.get("decision", "HOLD"), "#9E9E9E")  # domyślnie szary
            
            confidence = decision.get("confidence", 0) * 100
            
            st.markdown("""
                <div style="background-color: #1e2130; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="color: {decision_color};">{decision_type} - {symbol}</h4>
                    <p>Pewność: {confidence_value}%</p>
                    <div style="width: 100%; height: 10px; background-color: #333333; border-radius: 5px;">
                        <div style="width: {confidence_width}%; height: 10px; background-color: {color}; border-radius: 5px;"></div>
                    </div>
                    <p style="margin-top: 10px;"><b>Wyjaśnienie:</b><br>{explanation}</p>
                    <p style="margin-top: 10px;">Timestamp: {timestamp}</p>
                </div>
                """.format(
                    decision_color=decision_color,
                    decision_type=decision.get("decision", "UNKNOWN"),
                    symbol=decision.get("symbol", ""),
                    confidence_value=confidence,
                    confidence_width=confidence,
                    color=decision_color,
                    explanation=decision.get("explanation", "Brak wyjaśnienia").replace('\n', '<br>'),
                    timestamp=decision.get("timestamp", "")
                ), unsafe_allow_html=True)
    else:
        st.info("Brak ostatnich decyzji autonomicznego systemu")
    
    # Parametry autonomii
    st.subheader("⚙️ Parametry systemu autonomicznego")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Meta-Agent")
        
        decision_threshold = st.slider(
            "Próg pewności decyzji",
            min_value=0.5,
            max_value=1.0,
            value=0.65,
            step=0.01,
            help="Minimalna pewność wymagana do podjęcia decyzji"
        )
        
        model_weights = get_api_data("/api/autonomous/model-weights", {"weights": {}}).get("weights", {})
        st.markdown("#### Wagi modeli")
        
        for model_name, weight in model_weights.items():
            new_weight = st.slider(
                f"{model_name}",
                min_value=0.0,
                max_value=2.0,
                value=float(weight),
                step=0.1
            )
            
            if new_weight != weight:
                requests.post(
                    'http://localhost:5000/api/autonomous/model-weights',
                    json={"model": model_name, "weight": new_weight},
                    headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
                )
    
    with col2:
        st.markdown("### Zarządzanie ryzykiem")
        
        risk_params = get_api_data("/api/autonomous/risk-parameters", {"parameters": {}}).get("parameters", {})
        
        base_position_size = st.slider(
            "Bazowa wielkość pozycji (%)",
            min_value=0.01,
            max_value=0.1,
            value=float(risk_params.get("base_position_size", 0.02)),
            step=0.01,
            format="%.2f"
        )
        
        max_position_size = st.slider(
            "Maksymalna wielkość pozycji (%)",
            min_value=0.02,
            max_value=0.2,
            value=float(risk_params.get("max_position_size", 0.05)),
            step=0.01,
            format="%.2f"
        )
        
        max_drawdown = st.slider(
            "Maksymalny dozwolony drawdown (%)",
            min_value=0.05,
            max_value=0.3,
            value=float(risk_params.get("max_drawdown", 0.1)),
            step=0.01,
            format="%.2f"
        )
        
        if st.button("Zapisz parametry ryzyka"):
            risk_config = {
                "base_position_size": base_position_size,
                "max_position_size": max_position_size,
                "max_drawdown": max_drawdown
            }
            
            response = requests.post(
                'http://localhost:5000/api/autonomous/risk-parameters',
                json=risk_config,
                headers={'Authorization': f'Bearer {st.session_state.get("token")}'}
            )
            
            if response.status_code == 200:
                st.success("Zapisano parametry ryzyka")
            else:
                st.error("Błąd podczas zapisywania parametrów ryzyka")
    
    # Status uczenia i adaptacji
    st.subheader("📊 Status systemu uczenia")
    
    learning_status = get_api_data("/api/autonomous/learning-status", {"status": {}}).get("status", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Próbki w kolejce", learning_status.get("samples_in_queue", 0))
    
    with col2:
        st.metric("Ostatnia aktualizacja", learning_status.get("last_update", "Brak"))
    
    with col3:
        st.metric("Model z najwyższym wynikiem", learning_status.get("best_model", "Brak"))
    
    # Wykres wydajności modeli
    st.markdown("#### Wydajność modeli")
    
    model_performance = get_api_data("/api/autonomous/model-performance", {"performance": []}).get("performance", [])
    
    if model_performance:
        df = pd.DataFrame(model_performance)
        fig = go.Figure()
        
        # Dodaj linie dla każdego modelu
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(go.Scatter(
                x=model_data['timestamp'],
                y=model_data['score'],
                mode='lines',
                name=model
            ))
        
        fig.update_layout(
            title='Wydajność modeli w czasie',
            xaxis_title='Data',
            yaxis_title='Wynik',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Brak danych o wydajności modeli")

if __name__ == "__main__":
    # To jest uruchamiane gdy uruchamiamy skrypt bezpośrednio z Streamlit
    # Streamlit automatycznie uruchamia tę aplikację, więc nie musimy
    # nic dodawać tutaj
    pass

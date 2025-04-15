
#!/usr/bin/env python3
"""
fix_dashboard.py - Skrypt do naprawy dashboardu i interfejsu użytkownika.

Ten skrypt sprawdza i naprawia problemy związane z interfejsem dashboardu i API.
"""

import os
import re
import logging
import json
import shutil
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fix_dashboard_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories_exist():
    """Tworzy niezbędne katalogi, jeśli nie istnieją."""
    directories = [
        "static/css",
        "static/js",
        "static/img",
        "templates",
        "logs",
        "data/cache"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Upewniono się, że katalog '{directory}' istnieje")

def fix_api_routes():
    """Sprawdza i naprawia endpointy API w głównym pliku aplikacji."""
    try:
        with open("main.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Sprawdź, czy istnieje endpoint dla /api/sentiment/latest
        if "'/api/sentiment/latest'" not in content and '@app.route(\'/api/sentiment/latest\'' not in content:
            logger.info("Endpoint '/api/sentiment/latest' nie istnieje - dodaję go")
            
            # Szukaj wzorca deklaracji endpointu /api/sentiment
            pattern = r"@app\.route\('/api/sentiment'[^\n]*\n[^\n]*def get_sentiment_data\(\):"
            replacement = "@app.route('/api/sentiment', methods=['GET'])\n@app.route('/api/sentiment/latest', methods=['GET'])\ndef get_sentiment_data():"
            
            updated_content = re.sub(pattern, replacement, content)
            
            if updated_content != content:
                with open("main.py", "w", encoding="utf-8") as f:
                    f.write(updated_content)
                logger.info("Dodano endpoint '/api/sentiment/latest'")
            else:
                logger.warning("Nie udało się znaleźć miejsca do dodania endpointu '/api/sentiment/latest'")
        else:
            logger.info("Endpoint '/api/sentiment/latest' już istnieje")
    except Exception as e:
        logger.error(f"Błąd podczas naprawiania endpointów API: {e}")

def fix_dashboard_js():
    """Sprawdza i naprawia problemy w pliku dashboard.js."""
    try:
        js_file = "static/js/dashboard.js"
        if not os.path.exists(js_file):
            logger.error(f"Plik {js_file} nie istnieje! Nie można naprawić.")
            return
        
        with open(js_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Sprawdź, czy jest poprawny URL do API sentymentu
        if "'/api/sentiment/latest'" in content and "'/api/sentiment'" not in content:
            logger.info("Znaleziono odwołania do '/api/sentiment/latest' - poprawiam na '/api/sentiment'")
            content = content.replace("'/api/sentiment/latest'", "'/api/sentiment'")
            
            with open(js_file, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Poprawiono URL do API sentymentu")
        else:
            logger.info("URL do API sentymentu jest poprawny lub już został naprawiony")
    except Exception as e:
        logger.error(f"Błąd podczas naprawiania dashboard.js: {e}")

def check_templates():
    """Sprawdza i naprawia szablony HTML."""
    try:
        template_file = "templates/dashboard.html"
        if not os.path.exists(template_file):
            logger.error(f"Plik {template_file} nie istnieje! Nie można sprawdzić.")
            return
        
        with open(template_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Sprawdź, czy istnieją odpowiednie kontenery dla danych portfela
        if 'id="portfolio-container"' not in content and 'id="portfolio-data"' not in content:
            logger.warning("Brak kontenera na dane portfela w szablonie! Dodaję...")
            
            # Znajdź sekcję, gdzie powinien być kontener portfela
            portfolio_section_pattern = r'<div class="card">\s*<h2>Portfolio ByBit</h2>\s*<div[^>]*>'
            portfolio_replacement = '<div class="card">\n    <h2>Portfolio ByBit</h2>\n    <div id="portfolio-container" class="portfolio-data">'
            
            updated_content = re.sub(portfolio_section_pattern, portfolio_replacement, content)
            
            if updated_content != content:
                with open(template_file, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                logger.info("Dodano kontener na dane portfela w szablonie")
            else:
                logger.warning("Nie udało się znaleźć miejsca do dodania kontenera portfela")
        else:
            logger.info("Kontenery na dane portfela są już obecne w szablonie")
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania szablonów: {e}")

def create_backup_file(file_path):
    """Tworzy kopię zapasową pliku."""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup"
        shutil.copy2(file_path, backup_path)
        logger.info(f"Utworzono kopię zapasową pliku {file_path} -> {backup_path}")
        return True
    return False

def main():
    """Główna funkcja naprawcza."""
    logger.info("Rozpoczynam naprawę dashboardu i interfejsu...")
    
    # Tworzenie kopii zapasowych
    files_to_backup = ["main.py", "static/js/dashboard.js", "templates/dashboard.html"]
    for file_path in files_to_backup:
        create_backup_file(file_path)
    
    # Uruchomienie funkcji naprawczych
    ensure_directories_exist()
    fix_api_routes()
    fix_dashboard_js()
    check_templates()
    
    logger.info("Naprawa dashboardu i interfejsu zakończona.")
    print("Naprawa dashboardu i interfejsu zakończona. Teraz uruchom aplikację ponownie.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
fix_dashboard.py - Skrypt naprawiający dashboard i jego połączenia z API
"""

import os
import logging
import json
import shutil
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fix_dashboard_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def backup_file(filepath):
    """Tworzy kopię zapasową pliku"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup"
        shutil.copy2(filepath, backup_path)
        logger.info(f"Utworzono kopię zapasową: {backup_path}")
        return True
    return False

def ensure_directory_exists(directory):
    """Upewnia się, że katalog istnieje"""
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Upewniono się, że katalog istnieje: {directory}")

def fix_dashboard_js():
    """Naprawia plik dashboard.js"""
    js_file = "static/js/dashboard.js"
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(js_file):
        logger.error(f"Nie znaleziono pliku {js_file}")
        return False
    
    # Utwórz kopię zapasową
    backup_file(js_file)
    
    try:
        # Napraw plik
        with open(js_file, 'r') as f:
            content = f.read()
        
        # Napraw ustęp dotyczący pobierania statusu systemu
        if 'fetch(CONFIG.apiEndpoints.systemStatus)' in content:
            # Zastąp nieprawidłowy endpoint poprawnym
            content = content.replace(
                'fetch(CONFIG.apiEndpoints.systemStatus)',
                'fetch(CONFIG.apiEndpoints.componentStatus)'
            )
            logger.info("Naprawiono endpoint API statusu systemu")
        
        # Zapisz naprawiony plik
        with open(js_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Naprawiono plik {js_file}")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas naprawiania pliku {js_file}: {e}")
        return False

def fix_dashboard_html():
    """Naprawia plik dashboard.html"""
    html_file = "templates/dashboard.html"
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(html_file):
        logger.error(f"Nie znaleziono pliku {html_file}")
        return False
    
    # Utwórz kopię zapasową
    backup_file(html_file)
    
    try:
        # Napraw plik
        with open(html_file, 'r') as f:
            content = f.read()
        
        # Dodaj brakujący div do wyświetlania błędów jeśli go nie ma
        if '<div id="error-container" class="error-container">' not in content:
            error_div = '''
            <div id="error-container" class="error-container" style="display:none;">
                <div class="error-message">
                    <h3>Błąd</h3>
                    <p id="error-text"></p>
                    <button onclick="hideError()">Zamknij</button>
                </div>
            </div>
            '''
            
            # Dodaj przed znacznikiem </body>
            content = content.replace('</body>', f'{error_div}\n</body>')
            logger.info("Dodano kontener błędów do pliku HTML")
        
        # Zapisz naprawiony plik
        with open(html_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Naprawiono plik {html_file}")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas naprawiania pliku {html_file}: {e}")
        return False

def fix_api_endpoints():
    """Naprawia konfigurację endpointów API w pliku JS"""
    js_file = "static/js/dashboard.js"
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(js_file):
        logger.error(f"Nie znaleziono pliku {js_file}")
        return False
    
    try:
        # Odczytaj plik
        with open(js_file, 'r') as f:
            content = f.read()
        
        # Znajdź konfigurację API
        import re
        api_config_pattern = r'apiEndpoints:\s*\{[^}]*\}'
        api_config_match = re.search(api_config_pattern, content)
        
        if not api_config_match:
            logger.error("Nie znaleziono konfiguracji API w pliku JS")
            return False
        
        # Popraw konfigurację endpoints
        correct_endpoints = '''apiEndpoints: {
        portfolio: '/api/portfolio',
        dashboard: '/api/dashboard/data',
        componentStatus: '/api/component-status',
        chartData: '/api/chart/data',
        aiModelsStatus: '/api/ai-models-status',
        simulationResults: '/api/simulation-results',
        sentiment: '/api/sentiment',
        aiThoughts: '/api/ai/thoughts',
        learningStatus: '/api/ai/learning-status'
    }'''
        
        # Zastąp istniejącą konfigurację
        updated_content = re.sub(api_config_pattern, correct_endpoints, content)
        
        # Zapisz naprawiony plik
        with open(js_file, 'w') as f:
            f.write(updated_content)
        
        logger.info("Naprawiono konfigurację endpointów API w pliku JS")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas naprawiania konfiguracji API: {e}")
        return False

def ensure_needed_libraries():
    """Upewnia się, że wszystkie potrzebne moduły istnieją"""
    required_libs = [
        "python_libs/simplified_notification.py",
        "python_libs/simplified_trading_engine.py",
        "python_libs/simplified_risk_manager.py",
        "python_libs/simplified_strategy.py",
        "python_libs/model_tester.py",
        "python_libs/portfolio_manager.py"
    ]
    
    # Sprawdź, czy katalog python_libs istnieje
    ensure_directory_exists("python_libs")
    
    # Upewnij się, że istnieje plik __init__.py
    init_file = "python_libs/__init__.py"
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Inicjalizacja pakietu python_libs\n")
        logger.info(f"Utworzono plik {init_file}")
    
    # Sprawdź, czy wszystkie wymagane pliki istnieją
    missing_files = [lib for lib in required_libs if not os.path.exists(lib)]
    
    if missing_files:
        logger.warning(f"Brakujące pliki bibliotek: {missing_files}")
        return False
    else:
        logger.info("Wszystkie wymagane moduły bibliotek są dostępne")
        return True

def ensure_ai_models():
    """Upewnia się, że wszystkie modele AI są dostępne i poprawne"""
    required_models = [
        "ai_models/__init__.py",
        "ai_models/sentiment_ai.py",
        "ai_models/anomaly_detection.py",
        "ai_models/model_recognition.py"
    ]
    
    # Sprawdź, czy katalog ai_models istnieje
    ensure_directory_exists("ai_models")
    
    # Upewnij się, że istnieje plik __init__.py
    init_file = "ai_models/__init__.py"
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Inicjalizacja pakietu ai_models\n")
        logger.info(f"Utworzono plik {init_file}")
    
    # Sprawdź, czy wszystkie wymagane pliki istnieją
    missing_files = [model for model in required_models if not os.path.exists(model)]
    
    if missing_files:
        logger.warning(f"Brakujące pliki modeli AI: {missing_files}")
        return False
    else:
        logger.info("Wszystkie wymagane modele AI są dostępne")
        return True

def check_static_directories():
    """Sprawdza i tworzy wymagane katalogi statyczne"""
    directories = [
        "static",
        "static/css",
        "static/js",
        "static/img"
    ]
    
    for directory in directories:
        ensure_directory_exists(directory)
    
    logger.info("Sprawdzono i utworzono wymagane katalogi statyczne")
    return True

def create_example_chart():
    """Tworzy przykładowy wykres, jeśli go nie ma"""
    chart_path = "static/img/simulation_chart.png"
    
    if not os.path.exists(chart_path):
        try:
            # Spróbuj utworzyć prosty obraz wykresu
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(10, 6))
            days = 30
            x = np.arange(days)
            y = np.cumsum(np.random.normal(0.5, 1, days))
            plt.plot(x, y, 'g-')
            plt.title("Symulacja wyników tradingu")
            plt.xlabel("Dni")
            plt.ylabel("Zysk/Strata")
            plt.grid(True)
            plt.savefig(chart_path)
            plt.close()
            logger.info(f"Utworzono przykładowy wykres: {chart_path}")
            return True
        except Exception as e:
            logger.warning(f"Nie można utworzyć przykładowego wykresu: {e}")
            # Utwórz pusty plik jako placeholder
            Path(chart_path).touch()
            logger.info(f"Utworzono pusty plik jako placeholder: {chart_path}")
            return False
    else:
        logger.info(f"Plik {chart_path} już istnieje")
        return True

def main():
    """Główna funkcja naprawiająca dashboard"""
    logger.info("Rozpoczęcie procesu naprawy dashboardu")
    
    # Sprawdź i utwórz wymagane katalogi
    check_static_directories()
    
    # Upewnij się, że wszystkie wymagane moduły bibliotek istnieją
    ensure_needed_libraries()
    
    # Upewnij się, że wszystkie modele AI są dostępne
    ensure_ai_models()
    
    # Napraw plik dashboard.js
    fix_dashboard_js()
    
    # Napraw plik dashboard.html
    fix_dashboard_html()
    
    # Napraw konfigurację endpointów API
    fix_api_endpoints()
    
    # Utwórz przykładowy wykres
    create_example_chart()
    
    logger.info("Zakończono proces naprawy dashboardu")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
fix_dashboard.py - Skrypt do naprawy plików dashboardu.
"""
import os
import shutil
import logging
from typing import List, Dict, Tuple

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fix_dashboard_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory: str) -> None:
    """Upewnia się, że katalog istnieje."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Utworzono katalog {directory}")

def check_static_files() -> None:
    """Sprawdza i tworzy niezbędne pliki statyczne."""
    # Sprawdź katalogi
    static_dirs = ['static/css', 'static/js', 'static/img']
    for directory in static_dirs:
        ensure_directory_exists(directory)
    
    # Sprawdź plik CSS dashboardu
    css_file = 'static/css/styles.css'
    if not os.path.exists(css_file):
        default_css = """
body {
    font-family: 'Arial', sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f7fa;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.card {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    padding: 20px;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.header h1 {
    margin: 0;
    color: #333;
}

.tabs {
    display: flex;
    border-bottom: 1px solid #ddd;
    margin-bottom: 20px;
}

.tab {
    padding: 10px 20px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
}

.tab.active {
    border-bottom: 2px solid #4CAF50;
    color: #4CAF50;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.stats-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.stat-card {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
}

.stat-card h3 {
    margin: 0 0 10px 0;
    color: #666;
    font-size: 14px;
}

.stat-card p {
    margin: 0;
    font-size: 24px;
    font-weight: bold;
    color: #333;
}

.chart-container {
    height: 400px;
    margin-bottom: 20px;
}

.grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
}

table {
    width: 100%;
    border-collapse: collapse;
}

table th,
table td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

table th {
    background-color: #f2f2f2;
}

.btn {
    display: inline-block;
    padding: 8px 16px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    text-decoration: none;
}

.btn:hover {
    background-color: #45a049;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 12px;
}

.badge-success {
    background-color: #e6f7e6;
    color: #4CAF50;
}

.badge-warning {
    background-color: #fff3e0;
    color: #ff9800;
}

.badge-danger {
    background-color: #ffebee;
    color: #f44336;
}

.online {
    color: #4CAF50;
}

.offline {
    color: #f44336;
}

.degraded {
    color: #ff9800;
}

.strategy-card {
    border: 1px solid #eee;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 15px;
}

.strategy-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.switch {
    position: relative;
    display: inline-block;
    width: 60px;
    height: 34px;
}

.switch input {
    opacity: 0;
    width: 0;
    height: 0;
}

.slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    transition: .4s;
    border-radius: 34px;
}

.slider:before {
    position: absolute;
    content: "";
    height: 26px;
    width: 26px;
    left: 4px;
    bottom: 4px;
    background-color: white;
    transition: .4s;
    border-radius: 50%;
}

input:checked + .slider {
    background-color: #4CAF50;
}

input:focus + .slider {
    box-shadow: 0 0 1px #4CAF50;
}

input:checked + .slider:before {
    transform: translateX(26px);
}

.ai-model-card {
    border: 1px solid #eee;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 15px;
}

.ai-model-card h3 {
    margin-top: 0;
}

.progress {
    height: 8px;
    background-color: #f5f5f5;
    border-radius: 4px;
    margin-bottom: 10px;
}

.progress-bar {
    height: 100%;
    border-radius: 4px;
    background-color: #4CAF50;
}

/* AI Models Section */
.ai-models-container {
    margin-top: 20px;
}

.ai-model {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 15px;
}

.ai-model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.accuracy-indicator {
    height: 8px;
    background-color: #f5f5f5;
    border-radius: 4px;
    margin: 10px 0;
}

.accuracy-bar {
    height: 100%;
    border-radius: 4px;
}

/* AI Thoughts Section */
.thoughts-container {
    margin-top: 20px;
}

.thought-card {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 15px;
}

.thought-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.thought-model {
    font-weight: bold;
    color: #333;
}

.thought-confidence {
    font-size: 14px;
    color: #666;
}

.thought-content {
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 4px;
    margin-bottom: 10px;
}

.thought-timestamp {
    font-size: 12px;
    color: #999;
    text-align: right;
}

/* Responsywność */
@media (max-width: 768px) {
    .stats-container,
    .grid-container {
        grid-template-columns: 1fr;
    }
    
    .chart-container {
        height: 300px;
    }
}
"""
        with open(css_file, 'w', encoding='utf-8') as f:
            f.write(default_css)
        logger.info(f"Utworzono domyślny plik CSS: {css_file}")
    
    # Sprawdź plik JS dashboardu
    js_file = 'static/js/dashboard.js'
    if not os.path.exists(js_file):
        default_js = """
// Dashboard.js - Główny skrypt dashboardu

// Inicjalizacja zmiennych
let dashboardData = {};
let portfolioData = {};
let aiModelsData = [];
let aiThoughtsData = [];
let chartInstance = null;
let updateInterval = null;
let currentTab = 'dashboard-tab';

// Przy załadowaniu dokumentu
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard załadowany');
    initializeDashboard();
    
    // Inicjalizacja zakładek
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.getAttribute('data-tab');
            switchTab(tabId);
        });
    });
    
    // Przyciski akcji
    setupActionButtons();
});

// Inicjalizacja dashboardu
function initializeDashboard() {
    // Pobierz dane początkowe
    fetchDashboardData();
    fetchPortfolioData();
    fetchAIModelsStatus();
    fetchAIThoughts();
    
    // Ustaw interwał aktualizacji
    updateInterval = setInterval(() => {
        console.log('Aktualizacja danych dashboardu...');
        fetchDashboardData();
        fetchPortfolioData();
    }, 5000);
    
    // Interwał aktualizacji dla AI
    setInterval(() => {
        fetchAIModelsStatus();
        fetchAIThoughts();
    }, 30000);
}

// Przełączanie zakładek
function switchTab(tabId) {
    // Ukryj wszystkie zakładki
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Usuń aktywną klasę z wszystkich przycisków zakładek
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Aktywuj wybraną zakładkę
    document.getElementById(tabId).classList.add('active');
    document.querySelector(`.tab[data-tab="${tabId}"]`).classList.add('active');
    
    // Zapisz aktualną zakładkę
    currentTab = tabId;
    console.log(`Przełączono na zakładkę: ${tabId}`);
    
    // Specjalne akcje dla konkretnych zakładek
    if (tabId === 'ai-monitor-tab') {
        console.log('Inicjalizacja AI Monitor');
        fetchAIModelsStatus();
        fetchAIThoughts();
    }
}

// Pobieranie danych dashboardu
function fetchDashboardData() {
    fetch('/api/dashboard/data')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                dashboardData = data;
                updateDashboardUI(data);
                updateChartData();
            } else {
                console.error('Błąd podczas pobierania danych dashboardu:', data.error);
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych dashboardu:', error);
        });
}

// Pobieranie danych portfela
function fetchPortfolioData() {
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            portfolioData = data;
            updatePortfolioUI(data);
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych portfela:', error);
        });
}

// Pobieranie statusu komponentów
function fetchComponentStatus() {
    fetch('/api/component-status')
        .then(response => response.json())
        .then(data => {
            updateComponentStatusUI(data);
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statusu komponentów:', error);
        });
}

// Pobieranie statusu modeli AI
function fetchAIModelsStatus() {
    fetch('/api/ai-models-status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                aiModelsData = data.models;
                updateAIModelsUI(data.models);
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statusu modeli AI:', error);
        });
}

// Pobieranie myśli AI
function fetchAIThoughts() {
    fetch('/api/ai/thoughts')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                aiThoughtsData = data.thoughts;
                updateAIThoughtsUI(data.thoughts);
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania myśli AI:', error);
        });
}

// Aktualizacja UI dashboardu
function updateDashboardUI(data) {
    // Aktualizacja statystyk
    document.getElementById('balance-value').textContent = data.balance ? `$${data.balance.toFixed(2)}` : 'N/A';
    document.getElementById('profit-loss-value').textContent = data.profit_loss ? `$${data.profit_loss.toFixed(2)}` : 'N/A';
    document.getElementById('open-positions-value').textContent = data.open_positions || 'N/A';
    document.getElementById('win-rate-value').textContent = data.win_rate ? `${data.win_rate.toFixed(1)}%` : 'N/A';
    
    // Aktualizacja sentymentu rynkowego
    if (data.market_sentiment) {
        const sentimentElement = document.getElementById('market-sentiment-value');
        sentimentElement.textContent = data.market_sentiment;
        
        // Usunięcie poprzednich klas
        sentimentElement.classList.remove('badge-success', 'badge-warning', 'badge-danger');
        
        // Dodanie odpowiedniej klasy w zależności od sentymentu
        if (data.market_sentiment.toLowerCase().includes('bycz') || data.market_sentiment.toLowerCase().includes('pozytyw')) {
            sentimentElement.classList.add('badge-success');
        } else if (data.market_sentiment.toLowerCase().includes('niedź') || data.market_sentiment.toLowerCase().includes('negatyw')) {
            sentimentElement.classList.add('badge-danger');
        } else {
            sentimentElement.classList.add('badge-warning');
        }
    }
    
    // Aktualizacja statusu komponentów
    fetchComponentStatus();
}

// Aktualizacja UI portfela
function updatePortfolioUI(data) {
    const portfolioContainer = document.getElementById('portfolio-assets');
    if (!portfolioContainer) return;
    
    // Wyczyść poprzednie dane
    portfolioContainer.innerHTML = '';
    
    if (data.success && data.balances) {
        // Obliczanie całkowitej wartości portfela
        let totalValue = 0;
        for (const asset in data.balances) {
            totalValue += data.balances[asset].equity;
        }
        
        // Aktualizacja całkowitej wartości
        const portfolioValueElement = document.getElementById('portfolio-value');
        if (portfolioValueElement) {
            portfolioValueElement.textContent = `$${totalValue.toFixed(2)}`;
        }
        
        // Dodanie każdego aktywa do tabeli
        for (const asset in data.balances) {
            const assetData = data.balances[asset];
            const equity = assetData.equity;
            const percentOfPortfolio = (equity / totalValue * 100).toFixed(2);
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${asset}</td>
                <td>${asset === 'USDT' ? '$' + equity.toFixed(2) : equity.toFixed(8)}</td>
                <td>${percentOfPortfolio}%</td>
                <td>
                    <div class="progress">
                        <div class="progress-bar" style="width: ${percentOfPortfolio}%"></div>
                    </div>
                </td>
            `;
            
            portfolioContainer.appendChild(row);
        }
    } else {
        // Komunikat o błędzie
        portfolioContainer.innerHTML = '<tr><td colspan="4">Nie można załadować danych portfela</td></tr>';
    }
}

// Aktualizacja statusu komponentów
function updateComponentStatusUI(data) {
    const apiStatus = document.getElementById('api-status');
    const tradingStatus = document.getElementById('trading-status');
    
    if (apiStatus) {
        apiStatus.className = ''; // Usunięcie poprzednich klas
        apiStatus.classList.add(data.api);
        apiStatus.textContent = data.api.charAt(0).toUpperCase() + data.api.slice(1);
    }
    
    if (tradingStatus) {
        tradingStatus.className = ''; // Usunięcie poprzednich klas
        tradingStatus.classList.add(data.trading_engine);
        tradingStatus.textContent = data.trading_engine.charAt(0).toUpperCase() + data.trading_engine.slice(1);
    }
}

// Aktualizacja UI modeli AI
function updateAIModelsUI(models) {
    const modelsContainer = document.getElementById('ai-models-list');
    if (!modelsContainer) return;
    
    // Wyczyść poprzednie dane
    modelsContainer.innerHTML = '';
    
    if (models && models.length > 0) {
        models.forEach(model => {
            const modelElement = document.createElement('div');
            modelElement.className = 'ai-model';
            
            // Określenie klasy statusu
            let statusClass = 'badge-warning';
            if (model.status === 'Active') {
                statusClass = 'badge-success';
            } else if (model.status === 'Error') {
                statusClass = 'badge-danger';
            }
            
            modelElement.innerHTML = `
                <div class="ai-model-header">
                    <h3>${model.name}</h3>
                    <span class="badge ${statusClass}">${model.status}</span>
                </div>
                <p>Typ: ${model.type}</p>
                <p>Dokładność: ${model.accuracy ? model.accuracy.toFixed(1) + '%' : 'N/A'}</p>
                <div class="accuracy-indicator">
                    <div class="accuracy-bar" style="width: ${model.accuracy || 0}%; background-color: ${getAccuracyColor(model.accuracy)}"></div>
                </div>
                <p>Ostatnio używany: ${model.last_used || 'Nieznany'}</p>
            `;
            
            modelsContainer.appendChild(modelElement);
        });
    } else {
        modelsContainer.innerHTML = '<p>Brak dostępnych modeli AI</p>';
    }
}

// Aktualizacja UI myśli AI
function updateAIThoughtsUI(thoughts) {
    const thoughtsContainer = document.getElementById('ai-thoughts-list');
    if (!thoughtsContainer) return;
    
    // Wyczyść poprzednie dane
    thoughtsContainer.innerHTML = '';
    
    if (thoughts && thoughts.length > 0) {
        thoughts.forEach(thought => {
            const thoughtElement = document.createElement('div');
            thoughtElement.className = 'thought-card';
            
            thoughtElement.innerHTML = `
                <div class="thought-header">
                    <div class="thought-model">${thought.model}</div>
                    <div class="thought-confidence">Pewność: ${thought.confidence.toFixed(1)}%</div>
                </div>
                <div class="thought-content">${thought.thought}</div>
                <div class="thought-timestamp">${thought.timestamp}</div>
            `;
            
            thoughtsContainer.appendChild(thoughtElement);
        });
    } else {
        thoughtsContainer.innerHTML = '<p>Brak dostępnych przemyśleń AI</p>';
    }
}

// Aktualizacja danych wykresu
function updateChartData() {
    fetch('/api/chart-data')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                createOrUpdateChart(data.data);
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych wykresu:', error);
        });
}

// Tworzenie lub aktualizacja wykresu
function createOrUpdateChart(data) {
    const ctx = document.getElementById('portfolio-chart');
    if (!ctx) return;
    
    if (chartInstance) {
        // Aktualizuj istniejący wykres
        chartInstance.data = data;
        chartInstance.update();
    } else {
        // Utwórz nowy wykres
        chartInstance = new Chart(ctx, {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    }
}

// Funkcje pomocnicze

// Kolor dla paska dokładności
function getAccuracyColor(accuracy) {
    if (!accuracy) return '#ccc';
    
    if (accuracy >= 90) {
        return '#4CAF50';
    } else if (accuracy >= 75) {
        return '#8BC34A';
    } else if (accuracy >= 60) {
        return '#FFC107';
    } else {
        return '#F44336';
    }
}

// Ustawienie przycisków akcji
function setupActionButtons() {
    // Przycisk uruchomienia tradingu
    const startTradingBtn = document.getElementById('start-trading-btn');
    if (startTradingBtn) {
        startTradingBtn.addEventListener('click', function() {
            fetch('/api/trading/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Trading uruchomiony pomyślnie');
                    fetchComponentStatus();
                } else {
                    alert('Błąd podczas uruchamiania tradingu: ' + data.error);
                    console.error('Błąd podczas uruchamiania tradingu:', data);
                }
            })
            .catch(error => {
                alert('Błąd podczas uruchamiania tradingu');
                console.error('Błąd podczas uruchamiania tradingu:', error);
            });
        });
    }
    
    // Przycisk zatrzymania tradingu
    const stopTradingBtn = document.getElementById('stop-trading-btn');
    if (stopTradingBtn) {
        stopTradingBtn.addEventListener('click', function() {
            fetch('/api/trading/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Trading zatrzymany pomyślnie');
                    fetchComponentStatus();
                } else {
                    alert('Błąd podczas zatrzymywania tradingu: ' + data.error);
                    console.error('Błąd podczas zatrzymywania tradingu:', data);
                }
            })
            .catch(error => {
                alert('Błąd podczas zatrzymywania tradingu');
                console.error('Błąd podczas zatrzymywania tradingu:', error);
            });
        });
    }
    
    // Przycisk resetu systemu
    const resetSystemBtn = document.getElementById('reset-system-btn');
    if (resetSystemBtn) {
        resetSystemBtn.addEventListener('click', function() {
            if (confirm('Czy na pewno chcesz zresetować system?')) {
                fetch('/api/system/reset', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('System zresetowany pomyślnie');
                        fetchComponentStatus();
                        fetchPortfolioData();
                        fetchDashboardData();
                    } else {
                        alert('Błąd podczas resetowania systemu: ' + data.error);
                        console.error('Błąd podczas resetowania systemu:', data);
                    }
                })
                .catch(error => {
                    alert('Błąd podczas resetowania systemu');
                    console.error('Błąd podczas resetowania systemu:', error);
                });
            }
        });
    }
    
    // Przycisk uruchomienia symulacji
    const runSimulationBtn = document.getElementById('run-simulation-btn');
    if (runSimulationBtn) {
        runSimulationBtn.addEventListener('click', function() {
            const initialCapital = document.getElementById('simulation-capital').value || 10000;
            const duration = document.getElementById('simulation-duration').value || 1000;
            
            fetch('/api/simulation/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    initial_capital: parseFloat(initialCapital),
                    duration: parseInt(duration)
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    alert(`Symulacja zakończona pomyślnie. Profit: ${data.summary.profit.toFixed(2)}`);
                    // Odśwież dane symulacji
                    fetchSimulationResults();
                } else {
                    alert('Błąd podczas uruchamiania symulacji: ' + data.message);
                    console.error('Błąd podczas uruchamiania symulacji:', data);
                }
            })
            .catch(error => {
                alert('Błąd podczas uruchamiania symulacji');
                console.error('Błąd podczas uruchamiania symulacji:', error);
            });
        });
    }
}

// Pobieranie wyników symulacji
function fetchSimulationResults() {
    fetch('/api/simulation-results')
        .then(response => response.json())
        .then(data => {
            updateSimulationResultsUI(data);
        })
        .catch(error => {
            console.error('Błąd podczas pobierania wyników symulacji:', error);
        });
}

// Aktualizacja UI wyników symulacji
function updateSimulationResultsUI(data) {
    const resultsContainer = document.getElementById('simulation-results');
    if (!resultsContainer) return;
    
    if (data.status === 'success' && data.summary) {
        const summary = data.summary;
        resultsContainer.innerHTML = `
            <div class="card">
                <h3>Wyniki Symulacji</h3>
                <div class="grid-container">
                    <div class="stat-card">
                        <h3>Kapitał Początkowy</h3>
                        <p>$${summary.initial_capital.toFixed(2)}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Kapitał Końcowy</h3>
                        <p>$${summary.final_capital.toFixed(2)}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Zysk/Strata</h3>
                        <p style="color: ${summary.profit >= 0 ? '#4CAF50' : '#f44336'}">
                            $${summary.profit.toFixed(2)} (${summary.profit_percentage.toFixed(2)}%)
                        </p>
                    </div>
                    <div class="stat-card">
                        <h3>Liczba Transakcji</h3>
                        <p>${summary.trades}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Win Rate</h3>
                        <p>${summary.win_rate.toFixed(2)}%</p>
                    </div>
                    <div class="stat-card">
                        <h3>Max Drawdown</h3>
                        <p>${summary.max_drawdown.toFixed(2)}%</p>
                    </div>
                </div>
                
                <h3>Wykres Symulacji</h3>
                <div class="chart-container">
                    <img src="${data.chart_path || '/static/img/simulation_chart.png'}" alt="Wykres symulacji" style="width: 100%; height: 100%; object-fit: contain;">
                </div>
            </div>
        `;
    } else {
        resultsContainer.innerHTML = `
            <div class="card">
                <h3>Wyniki Symulacji</h3>
                <p>Brak danych symulacji lub wystąpił błąd: ${data.message || 'Nieznany błąd'}</p>
            </div>
        `;
    }
}

// Obsługa zdarzeń poza ładowaniem dokumentu

// Czyszczenie interwału przy opuszczeniu strony
window.addEventListener('beforeunload', function() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});
"""
        with open(js_file, 'w', encoding='utf-8') as f:
            f.write(default_js)
        logger.info(f"Utworzono domyślny plik JS: {js_file}")
    
    # Sprawdź plik AI monitor JS
    ai_monitor_js = 'static/js/ai_monitor.js'
    if not os.path.exists(ai_monitor_js):
        default_ai_js = """
// ai_monitor.js - Skrypt do monitora AI

document.addEventListener('DOMContentLoaded', function() {
    // Inicjalizacja monitorowania AI
    console.log('Inicjalizacja AI Monitor');
    fetchAILearningStatus();
    
    // Ustaw interwał aktualizacji
    setInterval(() => {
        fetchAILearningStatus();
    }, 10000);
    
    // Obsługa przycisku uczenia AI
    const trainAIBtn = document.getElementById('train-ai-btn');
    if (trainAIBtn) {
        trainAIBtn.addEventListener('click', function() {
            const iterations = document.getElementById('learning-iterations').value || 5;
            
            fetch('/api/simulation/learn', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    iterations: parseInt(iterations)
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    alert('Uczenie AI rozpoczęte pomyślnie');
                    fetchAILearningStatus();
                } else {
                    alert('Błąd podczas uruchamiania uczenia AI: ' + data.message);
                    console.error('Błąd podczas uruchamiania uczenia AI:', data);
                }
            })
            .catch(error => {
                alert('Błąd podczas uruchamiania uczenia AI');
                console.error('Błąd podczas uruchamiania uczenia AI:', error);
            });
        });
    }
});

// Pobieranie statusu uczenia AI
function fetchAILearningStatus() {
    fetch('/api/ai/learning-status')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                updateAILearningStatusUI(data);
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statusu uczenia AI:', error);
        });
}

// Aktualizacja UI statusu uczenia AI
function updateAILearningStatusUI(data) {
    const learningContainer = document.getElementById('ai-learning-status');
    if (!learningContainer) return;
    
    // Aktualizacja statusu uczenia
    const isTraining = data.is_training;
    const currentIteration = data.current_iteration;
    const totalIterations = data.total_iterations;
    
    let statusHTML = '';
    
    if (isTraining) {
        statusHTML += `
            <div class="card">
                <h3>Status Uczenia</h3>
                <p><strong>Status:</strong> <span class="badge badge-success">W trakcie uczenia</span></p>
                <p><strong>Postęp:</strong> Iteracja ${currentIteration} z ${totalIterations}</p>
                <div class="progress">
                    <div class="progress-bar" style="width: ${(currentIteration / totalIterations * 100).toFixed(1)}%"></div>
                </div>
            </div>
        `;
    } else {
        statusHTML += `
            <div class="card">
                <h3>Status Uczenia</h3>
                <p><strong>Status:</strong> <span class="badge badge-warning">Bezczynny</span></p>
                <p>Możesz rozpocząć nową sesję uczenia używając przycisku poniżej.</p>
            </div>
        `;
    }
    
    // Modele w trakcie uczenia
    if (data.models_training && data.models_training.length > 0) {
        statusHTML += `
            <div class="card">
                <h3>Modele w Trakcie Uczenia</h3>
                <div class="grid-container">
        `;
        
        data.models_training.forEach(model => {
            statusHTML += `
                <div class="stat-card">
                    <h3>${model.name}</h3>
                    <p><strong>Typ:</strong> ${model.type}</p>
                    <p><strong>Obecna dokładność:</strong> ${model.current_accuracy.toFixed(1)}%</p>
                    <p><strong>Pozostały czas:</strong> ${model.eta}</p>
                    <div class="progress">
                        <div class="progress-bar" style="width: ${model.progress.toFixed(1)}%"></div>
                    </div>
                </div>
            `;
        });
        
        statusHTML += `
                </div>
            </div>
        `;
    }
    
    // Wyniki uczenia
    if (data.learning_data && data.learning_data.length > 0) {
        statusHTML += `
            <div class="card">
                <h3>Wyniki Uczenia</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Iteracja</th>
                            <th>Dokładność</th>
                            <th>Win Rate</th>
                            <th>Transakcje</th>
                            <th>Zysk</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        data.learning_data.forEach(result => {
            statusHTML += `
                <tr>
                    <td>${result.iteration}</td>
                    <td>${result.accuracy.toFixed(2)}%</td>
                    <td>${result.win_rate.toFixed(2)}%</td>
                    <td>${result.trades}</td>
                    <td style="color: ${result.profit >= 0 ? '#4CAF50' : '#f44336'}">$${result.profit.toFixed(2)}</td>
                </tr>
            `;
        });
        
        statusHTML += `
                    </tbody>
                </table>
            </div>
        `;
    }
    
    learningContainer.innerHTML = statusHTML;
}
"""
        with open(ai_monitor_js, 'w', encoding='utf-8') as f:
            f.write(default_ai_js)
        logger.info(f"Utworzono domyślny plik AI monitor JS: {ai_monitor_js}")
    
    # Sprawdź plik CSS AI monitora
    ai_monitor_css = 'static/css/ai_monitor.css'
    if not os.path.exists(ai_monitor_css):
        default_ai_css = """
/* Styles dla monitora AI */

.ai-monitor-container {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.model-card {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
}

.model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.model-info {
    margin-bottom: 10px;
}

.model-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 10px;
    margin-bottom: 10px;
}

.model-metric {
    background-color: #f5f7fa;
    padding: 10px;
    border-radius: 4px;
}

.model-metric h4 {
    margin: 0 0 5px 0;
    font-size: 12px;
    color: #666;
}

.model-metric p {
    margin: 0;
    font-weight: bold;
    color: #333;
}

.model-actions {
    display: flex;
    gap: 10px;
}

/* Styles dla wykresu uczenia */
.learning-chart {
    width: 100%;
    height: 300px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 20px;
}

/* Styles dla formularza uczenia */
.learning-form {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 20px;
}

.form-group {
    margin-bottom: 15px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

.form-group input {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.form-actions {
    display: flex;
    justify-content: flex-end;
}
"""
        with open(ai_monitor_css, 'w', encoding='utf-8') as f:
            f.write(default_ai_css)
        logger.info(f"Utworzono domyślny plik AI monitor CSS: {ai_monitor_css}")
        
    # Sprawdź obrazek wykresu symulacji
    simulation_chart = 'static/img/simulation_chart.png'
    if not os.path.exists(simulation_chart):
        try:
            # Spróbuj utworzyć domyślny obrazek za pomocą matplotlib
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(10, 6))
            x = np.arange(100)
            y = np.cumsum(np.random.normal(0.1, 1, 100))
            plt.plot(x, y)
            plt.title('Symulacja Wyników Tradingu')
            plt.xlabel('Dni')
            plt.ylabel('Profit/Loss')
            plt.grid(True)
            plt.savefig(simulation_chart)
            plt.close()
            logger.info(f"Utworzono domyślny wykres symulacji: {simulation_chart}")
        except Exception as e:
            logger.warning(f"Nie można utworzyć domyślnego wykresu symulacji: {e}")
            # Utwórz pusty plik jako placeholder
            with open(simulation_chart, 'w') as f:
                pass

def check_templates():
    """Sprawdza i naprawia szablony."""
    # Sprawdź katalog szablonów
    ensure_directory_exists('templates')
    
    # Sprawdź dashboard.html
    dashboard_html = 'templates/dashboard.html'
    if not os.path.exists(dashboard_html):
        default_dashboard = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Dashboard</title>
    <link rel="stylesheet" href="/static/css/styles.css">
    <link rel="stylesheet" href="/static/css/ai_monitor.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Trading System Dashboard</h1>
            <div>
                <span>Status API: <span id="api-status" class="online">Online</span></span>
                <span>Status Silnika: <span id="trading-status" class="online">Online</span></span>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="dashboard-tab">Dashboard</div>
            <div class="tab" data-tab="portfolio-tab">Portfolio</div>
            <div class="tab" data-tab="ai-monitor-tab">AI Monitor</div>
            <div class="tab" data-tab="simulation-tab">Symulacja</div>
            <div class="tab" data-tab="settings-tab">Ustawienia</div>
        </div>
        
        <!-- Dashboard Tab -->
        <div id="dashboard-tab" class="tab-content active">
            <div class="stats-container">
                <div class="stat-card">
                    <h3>Stan Portfela</h3>
                    <p id="balance-value">$0.00</p>
                </div>
                <div class="stat-card">
                    <h3>Zysk/Strata</h3>
                    <p id="profit-loss-value">$0.00</p>
                </div>
                <div class="stat-card">
                    <h3>Otwarte Pozycje</h3>
                    <p id="open-positions-value">0</p>
                </div>
                <div class="stat-card">
                    <h3>Win Rate</h3>
                    <p id="win-rate-value">0%</p>
                </div>
                <div class="stat-card">
                    <h3>Sentyment Rynkowy</h3>
                    <p><span id="market-sentiment-value" class="badge badge-warning">Neutralny</span></p>
                </div>
            </div>
            
            <div class="card">
                <h2>Wykres Portfela</h2>
                <div class="chart-container">
                    <canvas id="portfolio-chart"></canvas>
                </div>
            </div>
            
            <div class="grid-container">
                <div class="card">
                    <h2>Aktywne Strategie</h2>
                    <div id="active-strategies">
                        {% for strategy in strategies %}
                        <div class="strategy-card">
                            <div class="strategy-header">
                                <h3>{{ strategy.name }}</h3>
                                <label class="switch">
                                    <input type="checkbox" {% if strategy.enabled %}checked{% endif %}>
                                    <span class="slider"></span>
                                </label>
                            </div>
                            <p>{{ strategy.description }}</p>
                            <p><strong>Win Rate:</strong> {{ strategy.win_rate }}%</p>
                            <p><strong>Profit Factor:</strong> {{ strategy.profit_factor }}</p>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <div class="card">
                    <h2>Ostatnie Transakcje</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Typ</th>
                                <th>Czas</th>
                                <th>Zysk/Strata</th>
                            </tr>
                        </thead>
                        <tbody id="recent-trades">
                            {% for trade in trades %}
                            <tr>
                                <td>{{ trade.symbol }}</td>
                                <td>{{ trade.type }}</td>
                                <td>{{ trade.time }}</td>
                                <td style="color: {% if trade.profit > 0 %}#4CAF50{% else %}#f44336{% endif %}">
                                    {{ trade.profit }}%
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="card">
                <h2>Przemyślenia AI</h2>
                <div id="ai-thoughts">
                    <div class="thoughts-container">
                        <div id="ai-thoughts-list">
                            <!-- Tutaj będą wyświetlane myśli AI -->
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Alerty Systemowe</h2>
                <div id="system-alerts">
                    <table>
                        <thead>
                            <tr>
                                <th>Poziom</th>
                                <th>Czas</th>
                                <th>Wiadomość</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for alert in alerts %}
                            <tr>
                                <td><span class="badge badge-{{ alert.level_class }}">{{ alert.level }}</span></td>
                                <td>{{ alert.time }}</td>
                                <td>{{ alert.message }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Portfolio Tab -->
        <div id="portfolio-tab" class="tab-content">
            <div class="card">
                <h2>Stan Portfela</h2>
                <div class="stats-container">
                    <div class="stat-card">
                        <h3>Wartość Całkowita</h3>
                        <p id="portfolio-value">$0.00</p>
                    </div>
                </div>
                
                <h3>Aktywa</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Aktywo</th>
                            <th>Ilość</th>
                            <th>% Portfela</th>
                            <th>Alokacja</th>
                        </tr>
                    </thead>
                    <tbody id="portfolio-assets">
                        <!-- Dane portfela będą tutaj -->
                    </tbody>
                </table>
            </div>
            
            <div class="grid-container">
                <div class="card">
                    <h2>Kontrola Tradingu</h2>
                    <div>
                        <button id="start-trading-btn" class="btn">Uruchom Trading</button>
                        <button id="stop-trading-btn" class="btn">Zatrzymaj Trading</button>
                        <button id="reset-system-btn" class="btn">Reset Systemu</button>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Statystyki Tradingowe</h2>
                    <div class="stats-container" id="trading-stats">
                        <!-- Statystyki tradingu będą tutaj -->
                    </div>
                </div>
            </div>
        </div>
        
        <!-- AI Monitor Tab -->
        <div id="ai-monitor-tab" class="tab-content">
            <div class="card">
                <h2>Modele AI</h2>
                <div class="ai-models-container">
                    <div id="ai-models-list">
                        <!-- Lista modeli AI będzie tutaj -->
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Uczenie AI</h2>
                <div class="learning-form">
                    <div class="form-group">
                        <label for="learning-iterations">Liczba Iteracji</label>
                        <input type="number" id="learning-iterations" value="5" min="1" max="100">
                    </div>
                    <div class="form-actions">
                        <button id="train-ai-btn" class="btn">Rozpocznij Uczenie</button>
                    </div>
                </div>
                
                <div id="ai-learning-status">
                    <!-- Status uczenia AI będzie tutaj -->
                </div>
            </div>
        </div>
        
        <!-- Simulation Tab -->
        <div id="simulation-tab" class="tab-content">
            <div class="card">
                <h2>Symulacja Tradingu</h2>
                <div class="grid-container">
                    <div class="stat-card">
                        <h3>Parametry Symulacji</h3>
                        <div class="form-group">
                            <label for="simulation-capital">Kapitał Początkowy ($)</label>
                            <input type="number" id="simulation-capital" value="10000" min="100">
                        </div>
                        <div class="form-group">
                            <label for="simulation-duration">Czas Trwania (ticki)</label>
                            <input type="number" id="simulation-duration" value="1000" min="100">
                        </div>
                        <button id="run-simulation-btn" class="btn">Uruchom Symulację</button>
                    </div>
                </div>
                
                <div id="simulation-results">
                    <!-- Wyniki symulacji będą tutaj -->
                </div>
            </div>
        </div>
        
        <!-- Settings Tab -->
        <div id="settings-tab" class="tab-content">
            <div class="card">
                <h2>Ustawienia Systemu</h2>
                <div class="grid-container">
                    <div class="card">
                        <h3>Zarządzanie Ryzykiem</h3>
                        <div class="form-group">
                            <label for="risk-level">Poziom Ryzyka</label>
                            <select id="risk-level">
                                <option value="low" {% if settings.risk_level == 'low' %}selected{% endif %}>Niski</option>
                                <option value="medium" {% if settings.risk_level == 'medium' %}selected{% endif %}>Średni</option>
                                <option value="high" {% if settings.risk_level == 'high' %}selected{% endif %}>Wysoki</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="max-position-size">Maksymalny Rozmiar Pozycji (%)</label>
                            <input type="number" id="max-position-size" value="{{ settings.max_position_size }}" min="1" max="100">
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>Automatyzacja</h3>
                        <div class="form-group">
                            <label>
                                <input type="checkbox" id="enable-auto-trading" {% if settings.enable_auto_trading %}checked{% endif %}>
                                Włącz Automatyczny Trading
                            </label>
                        </div>
                    </div>
                </div>
                
                <button class="btn" id="save-settings-btn">Zapisz Ustawienia</button>
            </div>
            
            <div class="card">
                <h2>Status Komponentów</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Komponent</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>API</td>
                            <td><span id="api-status-settings" class="online">Online</span></td>
                        </tr>
                        <tr>
                            <td>Silnik Handlowy</td>
                            <td><span id="trading-status-settings" class="online">Online</span></td>
                        </tr>
                        <tr>
                            <td>Sentymenty AI</td>
                            <td><span id="sentiment-status-settings" class="online">Online</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script src="/static/js/dashboard.js"></script>
    <script src="/static/js/ai_monitor.js"></script>
</body>
</html>
"""
        with open(dashboard_html, 'w', encoding='utf-8') as f:
            f.write(default_dashboard)
        logger.info(f"Utworzono domyślny szablon dashboardu: {dashboard_html}")
        
    # Utwórz plik __init__.py w katalogu templates
    init_file = 'templates/__init__.py'
    if not os.path.exists(init_file):
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write('"""\nPakiet szablonów Flask.\n"""\n')
        logger.info(f"Utworzono plik {init_file}")

def main():
    """Główna funkcja skryptu."""
    logger.info("Rozpoczynanie naprawy dashboardu...")
    
    # Sprawdź pliki statyczne
    check_static_files()
    
    # Sprawdź szablony
    check_templates()
    
    logger.info("Naprawa dashboardu zakończona pomyślnie.")
    logger.info("Możesz teraz uruchomić aplikację poleceniem: python main.py")

if __name__ == "__main__":
    main()

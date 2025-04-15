
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

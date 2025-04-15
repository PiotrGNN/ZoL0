
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

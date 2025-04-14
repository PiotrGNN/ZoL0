
#!/usr/bin/env python3
"""
install_replit.py - Skrypt do bezpiecznej instalacji pakietów w środowisku Replit.

Ten skrypt instaluje tylko niezbędne pakiety potrzebne do działania aplikacji
w środowisku Replit, pomijając ciężkie zależności jak TensorFlow i PyTorch,
które mogą powodować problemy z uprawnieniami.
"""

import os
import sys
import subprocess
import logging

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def install_packages():
    """Instaluje niezbędne pakiety z użyciem flagi --user."""
    
    logging.info("Rozpoczynam instalację pakietów dla środowiska Replit...")
    
    # Użyj flagi --user, aby zainstalować pakiety w katalogu użytkownika
    cmd = [sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info("Instalacja pakietów zakończona pomyślnie.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Błąd podczas instalacji pakietów: {e}")
        sys.exit(1)

def setup_local_directories():
    """Tworzy niezbędne katalogi dla aplikacji."""
    directories = [
        "logs",
        "data/cache",
        "reports",
        "static/img",
        "saved_models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Utworzono katalog: {directory}")

def main():
    """Funkcja główna."""
    logging.info("Rozpoczynam konfigurację projektu dla środowiska Replit...")
    
    # Instalacja pakietów
    install_packages()
    
    # Tworzenie katalogów
    setup_local_directories()
    
    logging.info("Konfiguracja projektu zakończona.")
    logging.info("Możesz teraz uruchomić aplikację za pomocą komendy: python main.py")

if __name__ == "__main__":
    main()

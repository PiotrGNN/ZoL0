#!/usr/bin/env python
"""
Skrypt pomocniczy do uruchamiania serwera Flask.
Rozwiązuje problemy z zajętym portem 5000.
"""
import os
import logging
import sys
from main import app

def kill_processes_on_port(port):
    """Zatrzymuje procesy na wskazanym porcie."""
    try:
        import subprocess
        import signal
        import time
        
        # Znajduje PID procesu używającego danego portu
        output = subprocess.check_output(
            f"lsof -i :{port} -t", shell=True, text=True
        ).strip()
        
        if output:
            pids = output.split('\n')
            for pid in pids:
                print(f"Zatrzymywanie procesu PID {pid} na porcie {port}")
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(0.5)  # Poczekaj na zatrzymanie procesu
                except Exception as e:
                    print(f"Błąd przy zatrzymywaniu procesu {pid}: {e}")
            return True
        return False
    except Exception as e:
        print(f"Nie znaleziono procesów na porcie {port}: {e}")
        return False

if __name__ == "__main__":
    # Konfiguracja logowania
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/server.log"),
            logging.StreamHandler()
        ]
    )

    # Utworzenie katalogu logs jeśli nie istnieje
    os.makedirs("logs", exist_ok=True)

    # Dodanie katalogu głównego do ścieżki Pythona
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # Uruchomienie aplikacji
    port = int(os.environ.get("PORT", 0))  # Port 0 oznacza, że system znajdzie wolny port
    logging.info(f"Uruchamianie aplikacji Flask - system wybierze wolny port")
    app.run(host='0.0.0.0', port=port, debug=True)
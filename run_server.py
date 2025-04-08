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
    port = int(os.environ.get("PORT", 8080))  # Używamy portu 8080 jako domyślnego
    
    # Sprawdzamy, czy port jest zajęty i próbujemy kilka alternatyw
    import socket
    
    # W Replit preferowane porty to 5000, 8080, 3000
    preferred_ports = [8080, 5000, 3000, 8000, 0]  # 0 oznacza automatyczny wybór
    
    for test_port in preferred_ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', test_port))
                port = test_port
                logging.info(f"Znaleziono wolny port: {port}")
                break
        except socket.error:
            logging.warning(f"Port {test_port} zajęty, próbujemy inny...")
    else:
        # Wykonuje się tylko gdy nie znaleziono portu w pętli
        port = 0
        logging.warning("Wszystkie testowane porty zajęte, używam automatycznego wyboru systemu")
    
    # Ustaw zmienną środowiskową, aby inne części aplikacji wiedziały, który port jest używany
    os.environ["PORT"] = str(port)
    
    logging.info(f"Uruchamianie aplikacji Flask na porcie {port if port != 0 else 'automatycznie wybranym'}")
    # debug=False w środowisku produkcyjnym pozwala uniknąć przeładowań w pętli
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
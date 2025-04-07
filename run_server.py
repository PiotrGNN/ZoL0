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
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port_available = False
    
    for test_port in [8080, 3000, 8000, 5001, 0]:  # 0 oznacza automatyczny wybór
        try:
            s.bind(('0.0.0.0', test_port))
            port_available = True
            port = test_port
            break
        except socket.error:
            logging.warning(f"Port {test_port} zajęty, próbujemy inny...")
    s.close()
    
    if not port_available and port != 0:
        logging.warning("Wszystkie testowane porty zajęte, używam automatycznego wyboru systemu")
        port = 0
    
    logging.info(f"Uruchamianie aplikacji Flask na porcie {port if port != 0 else 'automatycznie wybranym'}")
    app.run(host='0.0.0.0', port=port, debug=True)
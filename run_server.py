
#!/usr/bin/env python3
"""
Skrypt pomocniczy do uruchamiania serwera Flask.
Rozwiązuje problemy z zajętym portem 5000.
"""
import os
import signal
import subprocess
import time

def kill_processes_on_port(port):
    """Zatrzymuje procesy na wskazanym porcie."""
    try:
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
    except Exception as e:
        print(f"Nie znaleziono procesów na porcie {port}: {e}")

if __name__ == "__main__":
    PORT = 5000
    print(f"Sprawdzanie portu {PORT}...")
    kill_processes_on_port(PORT)
    print(f"Uruchamianie aplikacji Flask na porcie {PORT}...")
    os.system("python3 main.py")

"""
setup_ssh_tunnel.py
------------------
Skrypt do tworzenia tunelu SSH dla połączenia proxy SOCKS5.
Przydatne, gdy API wymaga stałego adresu IP lub gdy konieczne jest obejście ograniczeń geograficznych.

Użycie:
1. Ustaw zmienne środowiskowe lub edytuj wartości w skrypcie
2. Uruchom: python setup_ssh_tunnel.py
3. Zostaw terminal otwarty (tunel działa w tle)
4. Używaj proxy w swoich aplikacjach: socks5://127.0.0.1:1080
"""

import os
import sys
import time
import subprocess
import signal
import argparse

# Domyślne parametry połączenia
DEFAULT_SSH_HOST = "your_server_ip" # Zmień na adres swojego serwera
DEFAULT_SSH_PORT = 22
DEFAULT_SSH_USER = "root" # Zmień na nazwę użytkownika SSH 
DEFAULT_LOCAL_PORT = 1080
DEFAULT_KEY_PATH = "~/.ssh/id_rsa" # Ścieżka do klucza SSH

def parse_arguments():
    """Parsuje argumenty wiersza poleceń"""
    parser = argparse.ArgumentParser(description="Ustaw tunel SSH dla proxy SOCKS5")

    parser.add_argument("--host", 
                        default=os.getenv("SSH_HOST", DEFAULT_SSH_HOST),
                        help="Adres serwera SSH (domyślnie: z env SSH_HOST lub wartość domyślna)")

    parser.add_argument("--port", 
                        type=int,
                        default=int(os.getenv("SSH_PORT", DEFAULT_SSH_PORT)),
                        help="Port serwera SSH (domyślnie: z env SSH_PORT lub 22)")

    parser.add_argument("--user", 
                        default=os.getenv("SSH_USER", DEFAULT_SSH_USER),
                        help="Nazwa użytkownika SSH (domyślnie: z env SSH_USER lub root)")

    parser.add_argument("--local-port", 
                        type=int,
                        default=int(os.getenv("LOCAL_PORT", DEFAULT_LOCAL_PORT)),
                        help="Lokalny port dla proxy SOCKS5 (domyślnie: z env LOCAL_PORT lub 1080)")

    parser.add_argument("--key", 
                        default=os.getenv("SSH_KEY_PATH", DEFAULT_KEY_PATH),
                        help="Ścieżka do klucza SSH (domyślnie: z env SSH_KEY_PATH lub ~/.ssh/id_rsa)")

    parser.add_argument("--timeout", 
                        type=int,
                        default=int(os.getenv("TUNNEL_TIMEOUT", 0)),
                        help="Timeout tunelu w sekundach, 0 = bez limitu (domyślnie: z env TUNNEL_TIMEOUT lub 0)")

    return parser.parse_args()

def setup_ssh_tunnel(args):
    """Ustanawia tunel SSH dla proxy SOCKS5"""

    print(f"Ustanawianie tunelu SSH SOCKS5 na localhost:{args.local_port}...")
    print(f"Łączenie z serwerem: {args.user}@{args.host}:{args.port}")

    # Przygotowanie polecenia SSH
    cmd = [
        "ssh",
        "-N",  # Nie wykonuj zdalnego polecenia
        "-D", f"{args.local_port}",  # Ustanów tunel SOCKS
        "-o", "ExitOnForwardFailure=yes",  # Zakończ, jeśli przekierowanie się nie powiedzie
        "-o", "ServerAliveInterval=60",  # Wysyłaj pakiety co 60s, aby utrzymać połączenie
        "-o", "ServerAliveCountMax=3",  # Maksymalna liczba nieudanych keepalive przed rozłączeniem
    ]

    if args.key != "":
        cmd.extend(["-i", os.path.expanduser(args.key)])

    if args.port != 22:
        cmd.extend(["-p", str(args.port)])

    cmd.append(f"{args.user}@{args.host}")

    # Obsługa sygnałów do czystego zakończenia
    def signal_handler(sig, frame):
        print("\nPrzerywanie tunelu SSH...")
        if tunnel_process:
            tunnel_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Uruchomienie tunelu
    try:
        tunnel_process = subprocess.Popen(cmd)
        print(f"Tunel SSH uruchomiony (PID: {tunnel_process.pid})")
        print(f"Proxy SOCKS5 dostępne pod adresem: socks5://127.0.0.1:{args.local_port}")
        print("Naciśnij Ctrl+C, aby zatrzymać tunel")

        # Czekanie z timeoutem lub bez limitu
        if args.timeout > 0:
            print(f"Tunel zostanie zamknięty po {args.timeout} sekundach...")
            time.sleep(args.timeout)
            tunnel_process.terminate()
            print("Tunel SSH zakończony (timeout)")
        else:
            # Czekaj na zakończenie procesu
            tunnel_process.wait()

    except Exception as e:
        print(f"Błąd podczas tworzenia tunelu SSH: {e}")
        if tunnel_process:
            tunnel_process.terminate()
        return False

    return True

if __name__ == "__main__":
    args = parse_arguments()

    # Sprawdź, czy podano wymagane parametry
    if args.host == DEFAULT_SSH_HOST:
        print("UWAGA: Używasz domyślnego adresu serwera. Ustaw rzeczywisty adres poprzez:")
        print("- Zmienną środowiskową SSH_HOST")
        print("- Argument --host")
        print("- Edycję wartości DEFAULT_SSH_HOST w kodzie")

    setup_ssh_tunnel(args)
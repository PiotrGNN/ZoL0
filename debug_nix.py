
#!/usr/bin/env python3
"""
Skrypt do diagnozowania problemów z Nix w środowisku Replit.
Sprawdza dostępność wymaganych pakietów i zależności.
"""

import importlib.util
import subprocess
import os
import sys
import json
from datetime import datetime

def check_python_package(package_name):
    """Sprawdza czy pakiet Python jest dostępny."""
    package_name = package_name.replace('-', '_')
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except (ImportError, ValueError):
        return False

def check_system_command(command):
    """Sprawdza czy polecenie systemowe jest dostępne."""
    try:
        subprocess.run([command, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

def get_nix_package_info(package_path):
    """Pobiera informacje o pakiecie Nix."""
    try:
        result = subprocess.run(
            ["nix-store", "--query", "--requisites", package_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.stdout.strip().split('\n')
    except subprocess.SubprocessError:
        return []

def check_nix_env():
    """Sprawdza środowisko Nix."""
    nix_info = {}
    try:
        # Sprawdź wersję Nix
        result = subprocess.run(
            ["nix", "--version"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        nix_info["version"] = result.stdout.strip()
        
        # Sprawdź kanały
        result = subprocess.run(
            ["nix-channel", "--list"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        nix_info["channels"] = result.stdout.strip().split('\n')
        
        return nix_info
    except subprocess.SubprocessError:
        return {"error": "Nie można pobrać informacji o środowisku Nix"}

def main():
    """Główna funkcja diagnostyczna."""
    print(f"=== Diagnoza środowiska Nix w Replit ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    
    # Sprawdź podstawowe zmienne środowiskowe
    print("\n## Zmienne środowiskowe:")
    for var in ["PYTHONPATH", "PYTHONHOME", "NIXPKGS_ALLOW_UNFREE", "NIXPKGS_ALLOW_BROKEN"]:
        print(f"{var}: {os.environ.get(var, 'Nie ustawiono')}")
    
    # Sprawdź informacje o Nix
    print("\n## Informacje o Nix:")
    nix_info = check_nix_env()
    for key, value in nix_info.items():
        print(f"{key}: {value}")
    
    # Sprawdź pakiety Python
    print("\n## Pakiety Python:")
    python_packages = [
        "flask", "dotenv", "requests", "pybit", "numpy", "pandas", 
        "matplotlib", "scikit_learn", "websockets", "uvloop", "orjson"
    ]
    
    for package in python_packages:
        status = "✓" if check_python_package(package) else "✗"
        print(f"{status} {package}")
    
    # Sprawdź polecenia systemowe
    print("\n## Narzędzia systemowe:")
    system_commands = ["jq", "curl", "vim", "python", "python3"]
    
    for command in system_commands:
        status = "✓" if check_system_command(command) else "✗"
        print(f"{status} {command}")
    
    # Sprawdź ścieżki do interpretera Python
    print("\n## Ścieżki do interpretera Python:")
    try:
        python_path = subprocess.run(
            ["which", "python3"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ).stdout.strip()
        print(f"Ścieżka python3: {python_path}")
        
        if python_path:
            # Sprawdź czy to link symboliczny
            real_path = subprocess.run(
                ["readlink", "-f", python_path], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            ).stdout.strip()
            print(f"Rzeczywista ścieżka: {real_path}")
    except subprocess.SubprocessError:
        print("Nie można określić ścieżki do interpretera Python")
    
    print("\n=== Zakończono diagnozę ===")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
setup_local_packages.py - Skrypt do instalacji lokalnych pakietów niezbędnych do działania systemu.

Ten skrypt instaluje wszystkie brakujące pakiety z requirements.txt,
ignorując błędy i próbując zainstalować pakiety z flagą --user, jeśli instalacja
bez tej flagi nie powiedzie się.
"""

import os
import sys
import time
import subprocess
import importlib.util
import logging
from typing import List, Dict, Tuple, Set

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("setup_packages.log"),
        logging.StreamHandler()
    ]
)

def check_package(package_name: str) -> bool:
    """
    Sprawdza, czy pakiet jest zainstalowany.
    
    Args:
        package_name: Nazwa pakietu do sprawdzenia
        
    Returns:
        bool: True jeśli pakiet jest zainstalowany, False w przeciwnym razie
    """
    # Usuń wersję z nazwy pakietu (np. 'flask>=2.0.0' -> 'flask')
    base_package = package_name.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
    
    try:
        return importlib.util.find_spec(base_package) is not None
    except (ModuleNotFoundError, ValueError):
        return False

def install_package(package_name: str, use_user: bool = False) -> bool:
    """
    Instaluje pakiet za pomocą pip.
    
    Args:
        package_name: Nazwa pakietu do zainstalowania
        use_user: Czy użyć flagi --user
        
    Returns:
        bool: True jeśli instalacja się powiodła, False w przeciwnym razie
    """
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if use_user:
        cmd.append("--user")
        
    cmd.append(package_name)
    
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False

def parse_requirements(requirements_file: str) -> List[str]:
    """
    Parsuje plik requirements.txt.
    
    Args:
        requirements_file: Ścieżka do pliku requirements.txt
        
    Returns:
        List[str]: Lista nazw pakietów
    """
    if not os.path.exists(requirements_file):
        logging.error(f"Plik {requirements_file} nie istnieje")
        return []
        
    packages = []
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignoruj komentarze i puste linie
            if line and not line.startswith('#'):
                packages.append(line)
                
    return packages

def get_missing_packages(packages: List[str]) -> List[str]:
    """
    Zwraca listę brakujących pakietów.
    
    Args:
        packages: Lista nazw pakietów do sprawdzenia
        
    Returns:
        List[str]: Lista brakujących pakietów
    """
    missing = []
    for package in packages:
        if not check_package(package):
            missing.append(package)
            
    return missing

def main() -> None:
    """
    Funkcja główna.
    """
    print("🔧 Rozpoczęcie instalacji pakietów...")
    
    # Sprawdź, czy requirements.txt istnieje
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        logging.error(f"Plik {requirements_file} nie istnieje")
        sys.exit(1)
        
    # Parsuj requirements.txt
    packages = parse_requirements(requirements_file)
    logging.info(f"Znaleziono {len(packages)} pakietów w {requirements_file}")
    
    # Znajdź brakujące pakiety
    missing = get_missing_packages(packages)
    
    if not missing:
        logging.info("Wszystkie pakiety są już zainstalowane ✅")
        return
        
    logging.info(f"Znaleziono {len(missing)} brakujących pakietów")
    
    # Zainstaluj brakujące pakiety
    installation_results = {
        "success": [],
        "failed": []
    }
    
    for i, package in enumerate(missing):
        print(f"📦 Instalacja pakietu {i+1}/{len(missing)}: {package}")
        
        # Próbuj zainstalować bez flagi --user
        if install_package(package):
            installation_results["success"].append(package)
            logging.info(f"Zainstalowano pakiet {package} ✅")
        else:
            # Jeśli nie powiodło się, spróbuj z flagą --user
            logging.warning(f"Instalacja pakietu {package} nie powiodła się, próbuję z flagą --user")
            
            if install_package(package, use_user=True):
                installation_results["success"].append(package)
                logging.info(f"Zainstalowano pakiet {package} z flagą --user ✅")
            else:
                installation_results["failed"].append(package)
                logging.error(f"Nie udało się zainstalować pakietu {package} ❌")
                
    # Podsumowanie
    print("\n📋 Podsumowanie instalacji:")
    print(f"- Zainstalowano {len(installation_results['success'])} pakietów")
    print(f"- Nie udało się zainstalować {len(installation_results['failed'])} pakietów")
    
    if installation_results["failed"]:
        print("\n❌ Pakiety, których nie udało się zainstalować:")
        for package in installation_results["failed"]:
            print(f"- {package}")
            
    print("\n✅ Zakończono instalację pakietów")

if __name__ == "__main__":
    main()

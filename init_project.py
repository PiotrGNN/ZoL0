
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def install_dependencies():
    """Install specific versions of dependencies to ensure compatibility"""
    logging.info("Instalowanie zależności z określonymi wersjami...")
    
    # Install core dependencies with specific versions
    os.system("pip install --upgrade requests==2.31.0")
    os.system("pip install --upgrade websocket-client==1.7.0")
    os.system("pip install --upgrade typing-extensions==4.5.0")
    
    # Install from requirements.txt after adding version constraints
    os.system("pip install -r requirements.txt")
    
    logging.info("Instalacja zależności zakończona")

def check_installation():
    """Verify that dependencies were installed correctly"""
    try:
        import requests
        import websocket
        import typing_extensions
        
        logging.info(f"requests: {requests.__version__}")
        logging.info(f"websocket-client: {websocket.__version__}")
        logging.info(f"typing-extensions: {typing_extensions.__version__}")
        
        return True
    except ImportError as e:
        logging.error(f"Nie udało się zweryfikować instalacji: {e}")
        return False

if __name__ == "__main__":
    logging.info("Rozpoczynanie inicjalizacji projektu...")
    install_dependencies()
    if check_installation():
        logging.info("Inicjalizacja projektu zakończona sukcesem")
    else:
        logging.error("Inicjalizacja projektu nie powiodła się")

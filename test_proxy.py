import requests
import json
import time
import os
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Konfiguracja proxy SOCKS5
proxies = {
    'http': 'socks5h://127.0.0.1:1080',
    'https': 'socks5h://127.0.0.1:1080'
}

def test_connection():
    # Test połączenia z API Bybit z użyciem proxy
    try:
        url = 'https://api.bybit.com/v5/market/time'

        # Zawsze przygotuj domyślny czas serwera jako fallback
        server_time = {"timeNow": int(time.time() * 1000)}

        # Wersja bez proxy
        start_time_no_proxy = time.time()
        try:
            response_no_proxy = requests.get(url, timeout=10)
            elapsed_no_proxy = time.time() - start_time_no_proxy
            print(f"Status bez proxy: {response_no_proxy.status_code}, czas: {elapsed_no_proxy:.2f}s")
        except Exception as e:
            print(f"Błąd podczas testowania połączenia bez proxy: {e}")
            elapsed_no_proxy = time.time() - start_time_no_proxy
            print(f"Błąd bez proxy, czas: {elapsed_no_proxy:.2f}s")

        # Wersja z proxy
        start_time_proxy = time.time()
        try:
            response_proxy = requests.get(url, proxies=proxies, timeout=10)
            elapsed_proxy = time.time() - start_time_proxy
            print(f"Status z proxy: {response_proxy.status_code}, czas: {elapsed_proxy:.2f}s")

            if response_proxy.status_code == 200:
                data = response_proxy.json()
                if data.get("retCode") == 0 and "result" in data:
                    server_time = {"timeNow": data["result"]["timeNano"] // 1000000}
                print(f"Dane z proxy: {json.dumps(data, indent=2)}")
                return True
            else:
                print(f"Błąd HTTP z proxy: {response_proxy.status_code}")
                print(f"Używam lokalnego czasu: {server_time}")
                return False
        except Exception as e:
            elapsed_proxy = time.time() - start_time_proxy
            print(f"Błąd podczas testowania połączenia z proxy: {e}")
            print(f"Błąd z proxy, czas: {elapsed_proxy:.2f}s")
            print(f"Używam lokalnego czasu: {server_time}")
            return False
    except Exception as e:
        print(f"Krytyczny błąd podczas testowania połączenia: {e}")
        server_time = {"timeNow": int(time.time() * 1000)}
        print(f"Używam lokalnego czasu: {server_time}")
        return False

if __name__ == "__main__":
    print("Testowanie połączenia przez proxy SOCKS5...")
    result = test_connection()
    print(f"Test zakończony {'pomyślnie' if result else 'niepomyślnie'}")
"""
Test połączenia proxy SOCKS5 przez tunel SSH.

Ten skrypt testuje działanie proxy SOCKS5, które jest utworzone przez tunel SSH.
Wykonuje proste zapytanie HTTP do API Bybit przez proxsy i wyświetla odpowiedź.
"""

import requests
import time
import json

# Konfiguracja proxy SOCKS5
PROXY_URL = "socks5h://127.0.0.1:1080"
proxies = {
    'http': PROXY_URL,
    'https': PROXY_URL
}

def test_proxy():
    """Testuje połączenie przez proxy"""
    print("Testowanie połączenia proxy SOCKS5...")

    try:
        # Test połączenia z publicznym endpointem API Bybit bez uwierzytelniania
        start_time = time.time()
        response = requests.get('https://api.bybit.com/v5/market/time', 
                             proxies=proxies, 
                             timeout=10)

        end_time = time.time()
        latency = (end_time - start_time) * 1000  # w milisekundach

        print(f"Czas odpowiedzi: {latency:.2f} ms")
        print(f"Status kod: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("Odpowiedź API:")
            print(json.dumps(data, indent=2))

            if "retCode" in data and data["retCode"] == 0:
                print("\nTEST UDANY ✅ - Połączenie proxy działa poprawnie")
                return True
            else:
                print("\nTEST NIEUDANY ❌ - Błędna odpowiedź API")
                return False
    except Exception as e:
        print(f"\nTEST NIEUDANY ❌ - Błąd: {str(e)}")
        print("\nWskazówki do rozwiązania problemu:")
        print("1. Upewnij się, że tunel SSH jest uruchomiony (ssh -N -D 1080 user@host)")
        print("2. Sprawdź, czy port 1080 jest używany przez tunel")
        print("3. Zainstaluj wymagane pakiety: pip install requests[socks]")
        return False

if __name__ == "__main__":
    test_proxy()
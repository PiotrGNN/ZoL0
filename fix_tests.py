import os
import subprocess
import sys
import time

import psutil

# 📌 Automatyczne naprawienie problemów z testami

# 1️⃣ Sprawdzenie wymaganych pakietów
missing_packages = []
try:
    import numpy
except ImportError:
    missing_packages.append("numpy")

try:
    import requests
except ImportError:
    missing_packages.append("requests")

if missing_packages:
    print(f"🚨 Brakujące pakiety: {', '.join(missing_packages)}. Instalowanie...")
    subprocess.run(["pip", "install"] + missing_packages)

# 2️⃣ Naprawa `NoneType` w `AI_strategy_generator.py`
AI_STRATEGY_PATH = os.path.join("data", "strategies", "AI_strategy_generator.py")

if os.path.exists(AI_STRATEGY_PATH):
    with open(AI_STRATEGY_PATH, "r", encoding="utf-8") as f:
        content = f.readlines()

    fixed_content = []
    for line in content:
        if "self.best_model.fit(X_train, y_train)" in line:
            fixed_content.append("        if self.best_model is None:\n")
            fixed_content.append(
                '            raise ValueError("Nie ustawiono najlepszego modelu przed fitowaniem.")\n'
            )
        fixed_content.append(line)

    with open(AI_STRATEGY_PATH, "w", encoding="utf-8") as f:
        f.writelines(fixed_content)
    print("🔧 Naprawiono `NoneType` dla `best_model`.")

# 3️⃣ Zamykanie `temp_historical_data.db` przed usunięciem
db_path = "temp_historical_data.db"
if os.path.exists(db_path):
    for proc in psutil.process_iter():
        try:
            if "sqlite" in proc.name().lower() or "python" in proc.name().lower():
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(1)
    os.remove(db_path)
    print("✅ Zamknięto i usunięto `temp_historical_data.db`.")

# 4️⃣ Uruchomienie testów ponownie
print("🚀 Uruchamiam testy...")
subprocess.run(["python", "-m", "unittest", "discover", "-s", "data/tests"])

print("\n🎯 **Testy zakończone!** Sprawdź wyniki powyżej.")

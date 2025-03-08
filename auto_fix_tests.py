import logging
import os
import re
import subprocess
import sys
import time

# 📄 Konfiguracja logowania
LOG_FILE = "auto_fix_log.txt"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

print(f"📄 Logi będą zapisywane w: {LOG_FILE}")
logging.info("🔍 Rozpoczynam automatyczną naprawę testów...")

# 1️⃣ **Dodanie poprawnej ścieżki do sys.path**
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "data")))

# 2️⃣ **Sprawdzenie, czy wymagane pliki istnieją**
required_files = {
    "execution/exchange_connector.py": "data/execution/exchange_connector.py",
    "execution/order_execution.py": "data/execution/order_execution.py",
    "risk_management/stop_loss_manager.py": "data/risk_management/stop_loss_manager.py",
}

missing_files = [
    path for key, path in required_files.items() if not os.path.exists(path)
]

if missing_files:
    print(f"🚨 Brakujące pliki: {missing_files}")
    logging.error(f"🚨 Brakujące pliki: {missing_files}")

    # Jeśli `stop_loss_manager.py` jest w `risk_management/`, ale testy szukają w `execution/`, popraw importy
    if "data/execution/stop_loss_manager.py" in missing_files and os.path.exists(
        required_files["risk_management/stop_loss_manager.py"]
    ):
        print("🔧 Wykryto błędny import `stop_loss_manager.py`. Poprawiam...")
        logging.info("🔧 Wykryto błędny import `stop_loss_manager.py`. Poprawiam...")

        # Przeszukaj wszystkie pliki testowe i popraw importy
        TESTS_DIR = os.path.join(os.path.dirname(__file__), "data", "tests")
        for root, _, files in os.walk(TESTS_DIR):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Zamiana `data.execution.stop_loss_manager` na `data.risk_management.stop_loss_manager`
                    new_content = re.sub(
                        r"from data\.execution\.stop_loss_manager",
                        "from data.risk_management.stop_loss_manager",
                        content,
                    )

                    if new_content != content:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"✅ Poprawiono import w: {file_path}")
                        logging.info(f"✅ Poprawiono import w: {file_path}")

else:
    print("✅ Wszystkie wymagane pliki istnieją.")
    logging.info("✅ Wszystkie wymagane pliki istnieją.")

# 3️⃣ **Naprawa `NoneType` dla `best_model`**
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
    logging.info("🔧 Naprawiono `NoneType` dla `best_model`.")

# 4️⃣ **Naprawa `H` na `h` w testach strategii**
TEST_STRATEGIES_PATH = os.path.join("data", "tests", "test_strategies.py")

if os.path.exists(TEST_STRATEGIES_PATH):
    with open(TEST_STRATEGIES_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content.replace('freq="H"', 'freq="h"')

    if new_content != content:
        with open(TEST_STRATEGIES_PATH, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"🔄 Poprawiono `H` na `h` w: {TEST_STRATEGIES_PATH}")
        logging.info(f"🔄 Poprawiono `H` na `h` w: {TEST_STRATEGIES_PATH}")

# 5️⃣ **Uruchomienie testów ponownie**
print("🚀 Uruchamiam testy...")
logging.info("🚀 Uruchamiam testy ponownie...")
time.sleep(1)

subprocess.run(["python", "-m", "unittest", "discover", "-s", "data/tests"])

print("\n🎯 **Testy zakończone!** Sprawdź wyniki powyżej.")
logging.info("🎯 **Testy zakończone!**")

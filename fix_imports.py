
import logging
import os
import re

# Konfiguracja logowania
log_file = "fix_imports_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Foldery do przeszukania
SEARCH_DIRS = ["data", "ai_models", "config"]

# Lista błędnych importów i ich poprawnych wersji
IMPORT_FIXES = {
    "from data.risk_management.advanced_order_execution import": "from data.execution.advanced_order_execution import",
    "import data.risk_management.advanced_order_execution": "import data.execution.advanced_order_execution",
    "from data.execution.stop_loss_manager import": "from data.risk_management.stop_loss_manager import",
    "import data.execution.stop_loss_manager": "import data.risk_management.stop_loss_manager",
}


def fix_imports_in_file(file_path):
    """Poprawia błędne importy w podanym pliku."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        modified = False
        for old_import, correct_import in IMPORT_FIXES.items():
            if old_import in content:
                content = content.replace(old_import, correct_import)
                modified = True

        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"🔧 Poprawiono import w {file_path}")

    except Exception as e:
        logging.error(f"❌ Błąd podczas edycji {file_path}: {e}")


def scan_and_fix_imports():
    """Przeszukuje projekt i poprawia błędne importy."""
    logging.info("🚀 Rozpoczynam poprawę importów...")
    for search_dir in SEARCH_DIRS:
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    fix_imports_in_file(file_path)

    logging.info("✅ Naprawa importów zakończona.")
    print(f"📜 Logi zapisane w: {log_file}")


if __name__ == "__main__":
    scan_and_fix_imports()

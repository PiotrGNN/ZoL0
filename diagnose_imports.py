#!/usr/bin/env python3
"""
Script to diagnose import errors in the AI models
"""

import os
import sys
import traceback
import importlib


def test_import(module_name):
    print(f"Testing import: {module_name}")
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True, module
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        traceback.print_exc()
        return False, None


# Add the root directory to path to allow imports
sys.path.insert(0, os.path.abspath("."))

# Test key imports
print("\n===== Testing python_libs imports =====")
test_import("python_libs")
test_import("python_libs.model_tester")
test_import("python_libs.simplified_trading_engine")

print("\n===== Testing ZoL0 imports =====")
test_import("ZoL0")

# Try importing AI models
print("\n===== Testing AI model imports =====")

# First test direct imports of key models
try:
    from ZoL0.ai_models.anomaly_detection import AnomalyDetector

    print("✅ Successfully imported AnomalyDetector")
except Exception as e:
    print(f"❌ Failed to import AnomalyDetector: {e}")

try:
    from ZoL0.ai_models.model_recognition import ModelRecognizer

    print("✅ Successfully imported ModelRecognizer")
except Exception as e:
    print(f"❌ Failed to import ModelRecognizer: {e}")

# Test general ai_models import
test_import("ZoL0.ai_models")

# Try to find and import all modules in ai_models
print("\n===== Testing all AI models =====")
ai_models_dir = "ZoL0/ai_models"
if os.path.exists(ai_models_dir):
    for file in sorted(os.listdir(ai_models_dir)):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = f"ZoL0.ai_models.{file[:-3]}"
            success, module = test_import(module_name)
            if success:
                # Try to identify model classes
                for name in dir(module):
                    if (
                        name.endswith("Model")
                        or "Detector" in name
                        or "Recognizer" in name
                        or "Analyzer" in name
                    ) and not name.startswith("__"):
                        try:
                            cls = getattr(module, name)
                            if isinstance(cls, type):  # Check if it's a class
                                print(f"  Found potential model class: {name}")
                                # Check if instantiable
                                try:
                                    instance = cls()
                                    print(f"    ✅ Successfully instantiated {name}")
                                except Exception as e:
                                    print(f"    ❌ Failed to instantiate {name}: {e}")
                        except Exception as e:
                            print(f"  ❌ Error inspecting {name}: {e}")
else:
    print(f"❌ AI models directory not found: {ai_models_dir}")

print("\nDiagnosis complete.")

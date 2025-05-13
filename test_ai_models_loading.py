#!/usr/bin/env python3
"""
test_ai_models_loading.py - Script to test loading AI models and verify their functionality.
This script checks if all AI models can be properly imported and instantiated.
"""

import os
import sys
import logging
import importlib
import importlib.util
import traceback
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/model_loading_test.log", mode="w"),
    ],
)
logger = logging.getLogger("test_ai_models_loading")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)


def import_module_safely(module_name: str) -> Tuple[bool, Any]:
    """
    Safely import a module and return success status and the module if successful.

    Args:
        module_name: Name of the module to import

    Returns:
        Tuple of (success_status, module_or_error_message)
    """
    try:
        module = importlib.import_module(module_name)
        return True, module
    except ImportError as e:
        logger.error(f"Import error for {module_name}: {e}")
        return False, str(e)
    except Exception as e:
        logger.error(f"Unexpected error importing {module_name}: {e}")
        traceback.print_exc()
        return False, str(e)


def get_ai_model_files(ai_models_dir: str = "ZoL0/ai_models") -> List[str]:
    """
    Get list of Python files in the AI models directory.

    Args:
        ai_models_dir: Path to the AI models directory

    Returns:
        List of Python file names
    """
    try:
        if not os.path.exists(ai_models_dir):
            logger.error(f"AI models directory not found: {ai_models_dir}")
            return []

        files = []
        for file in os.listdir(ai_models_dir):
            if file.endswith(".py") and not file.startswith("__"):
                files.append(file)

        return sorted(files)
    except Exception as e:
        logger.error(f"Error scanning AI models directory: {e}")
        return []


def test_model_class(cls_name: str, cls: Any) -> Dict[str, Any]:
    """
    Test if a class is a valid AI model by checking for expected methods.

    Args:
        cls_name: Name of the class
        cls: The class object

    Returns:
        Dictionary with test results
    """
    result = {
        "name": cls_name,
        "is_class": isinstance(cls, type),
        "has_fit": False,
        "has_predict": False,
        "has_detect": False,
        "instantiable": False,
        "instance": None,
        "error": None,
    }

    if not result["is_class"]:
        result["error"] = "Not a class"
        return result

    # Check for expected methods
    result["has_fit"] = hasattr(cls, "fit") or any(
        hasattr(b, "fit") for b in cls.__mro__
    )
    result["has_predict"] = hasattr(cls, "predict") or any(
        hasattr(b, "predict") for b in cls.__mro__
    )
    result["has_detect"] = hasattr(cls, "detect") or any(
        hasattr(b, "detect") for b in cls.__mro__
    )

    # Try instantiating the class
    try:
        instance = cls()
        result["instantiable"] = True
        result["instance"] = instance
    except Exception as e:
        result["error"] = str(e)

    return result


def scan_module_for_models(module: Any) -> List[Dict[str, Any]]:
    """
    Scan a module for classes that might be AI models.

    Args:
        module: Imported module

    Returns:
        List of dictionaries with info about potential AI model classes
    """
    potential_models = []

    for name in dir(module):
        if name.startswith("__"):
            continue

        try:
            attr = getattr(module, name)

            # Only consider classes defined in this module
            if isinstance(attr, type) and attr.__module__ == module.__name__:
                # Check if it looks like an AI model
                if (
                    any(
                        method in dir(attr)
                        for method in ["fit", "predict", "detect", "train", "evaluate"]
                    )
                    or "Model" in name
                    or "Detector" in name
                    or "Recognizer" in name
                    or "Analyzer" in name
                ):
                    model_info = test_model_class(name, attr)
                    potential_models.append(model_info)
        except Exception as e:
            logger.warning(f"Error inspecting {name} in {module.__name__}: {e}")

    return potential_models


def test_model_instance(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test a model instance with some synthetic data.

    Args:
        model_info: Dictionary with model info including the instance

    Returns:
        Dictionary with test results
    """
    instance = model_info["instance"]
    if instance is None:
        return {"test_success": False, "error": "No instance available"}

    results = {"test_success": False}

    # Create some synthetic test data
    try:
        X = np.random.random((10, 5))
        y = np.random.random(10)

        # Test fit if available
        if model_info["has_fit"]:
            try:
                instance.fit(X, y)
                results["fit_success"] = True
            except Exception as e:
                results["fit_success"] = False
                results["fit_error"] = str(e)

        # Test predict if available
        if model_info["has_predict"]:
            try:
                prediction = instance.predict(X)
                results["predict_success"] = True
                results["prediction_shape"] = str(np.array(prediction).shape)
            except Exception as e:
                results["predict_success"] = False
                results["predict_error"] = str(e)

        # Test detect if available
        if model_info["has_detect"]:
            try:
                detection = instance.detect(X)
                results["detect_success"] = True
            except Exception as e:
                results["detect_success"] = False
                results["detect_error"] = str(e)

        # If at least one test was successful, consider the overall test successful
        results["test_success"] = any(
            [
                results.get("fit_success", False),
                results.get("predict_success", False),
                results.get("detect_success", False),
            ]
        )

    except Exception as e:
        results["error"] = str(e)

    return results


def test_python_libs_tester():
    """Test the ModelTester class from python_libs."""
    logger.info("Testing python_libs.model_tester...")

    success, result = import_module_safely("python_libs.model_tester")
    if not success:
        logger.error(f"Could not import python_libs.model_tester: {result}")
        return False

    try:
        ModelTester = result.ModelTester
        tester = ModelTester(models_path="ZoL0/ai_models")
        logger.info("Successfully instantiated ModelTester")
        return True
    except Exception as e:
        logger.error(f"Error instantiating ModelTester: {e}")
        return False


def main():
    """Main function to test loading AI models."""
    logger.info("Starting AI model loading test")

    # Test importing python_libs
    test_python_libs_tester()

    # Get AI model files
    ai_models_dir = "ZoL0/ai_models"
    model_files = get_ai_model_files(ai_models_dir)
    logger.info(f"Found {len(model_files)} AI model files")

    # Add ai_models to module search path if needed
    if ai_models_dir not in sys.path:
        sys.path.append(os.path.dirname(ai_models_dir))

    # Test importing each module
    all_models = []
    import_results = {}

    for file in model_files:
        module_name = f"ZoL0.ai_models.{file[:-3]}"
        logger.info(f"Testing import for {module_name}")

        success, module = import_module_safely(module_name)
        import_results[module_name] = {"success": success}

        if success:
            # Scan module for model classes
            models = scan_module_for_models(module)
            logger.info(f"Found {len(models)} potential models in {module_name}")

            # Add module name to model info
            for model in models:
                model["module"] = module_name
                all_models.append(model)
        else:
            import_results[module_name][
                "error"
            ] = module  # module contains error message

    # Test model instances
    for model in all_models:
        if model["instantiable"] and model["instance"]:
            logger.info(f"Testing model: {model['name']} from {model['module']}")
            test_results = test_model_instance(model)
            model["test_results"] = test_results

            if test_results["test_success"]:
                logger.info(f"✅ {model['name']} tests passed")
            else:
                if "error" in test_results:
                    logger.warning(
                        f"❌ {model['name']} tests failed: {test_results['error']}"
                    )
                else:
                    logger.warning(f"❌ {model['name']} tests failed")

    # Summarize results
    logger.info("\nModule Import Results:")
    successful_imports = sum(
        1 for result in import_results.values() if result["success"]
    )
    logger.info(
        f"✅ Successfully imported {successful_imports} out of {len(import_results)} modules"
    )

    failed_imports = [
        (name, result["error"])
        for name, result in import_results.items()
        if not result["success"]
    ]
    if failed_imports:
        logger.info("\nFailed imports:")
        for name, error in failed_imports:
            logger.info(f"❌ {name}: {error}")

    logger.info("\nModel Test Results:")
    instantiable_models = sum(1 for model in all_models if model["instantiable"])
    logger.info(
        f"Found {len(all_models)} model classes, {instantiable_models} could be instantiated"
    )

    successful_tests = sum(
        1
        for model in all_models
        if model.get("test_results", {}).get("test_success", False)
    )
    logger.info(f"✅ {successful_tests} models passed tests")

    with_fit = sum(1 for model in all_models if model["has_fit"])
    with_predict = sum(1 for model in all_models if model["has_predict"])
    with_detect = sum(1 for model in all_models if model["has_detect"])

    logger.info(f"Models with fit(): {with_fit}")
    logger.info(f"Models with predict(): {with_predict}")
    logger.info(f"Models with detect(): {with_detect}")

    # Print detailed results for each model
    logger.info("\nDetailed Model Results:")
    for model in all_models:
        status = "✅" if model["instantiable"] else "❌"
        methods = []
        if model["has_fit"]:
            methods.append("fit")
        if model["has_predict"]:
            methods.append("predict")
        if model["has_detect"]:
            methods.append("detect")

        method_str = ", ".join(methods) if methods else "no methods"
        logger.info(f"{status} {model['module']}.{model['name']} - {method_str}")

        if not model["instantiable"] and model["error"]:
            logger.info(f"   Error: {model['error']}")

    return successful_imports > 0 and successful_tests > 0


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("AI model loading test completed successfully")
        sys.exit(0)
    else:
        logger.error("AI model loading test failed")
        sys.exit(1)

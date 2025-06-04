class ModelTester:
    """Minimal stub for model testing."""
    def __init__(self, models_path: str = "ai_models", log_path: str = None):
        self.models_path = models_path
        self.log_path = log_path
        self._loaded_models = []

    def run_tests(self):
        """Pretend to load models by instantiating simple objects."""
        import importlib, pkgutil
        for loader, name, ispkg in pkgutil.iter_modules([self.models_path]):
            try:
                module = importlib.import_module(f"{self.models_path}.{name}")
                for attr in dir(module):
                    obj = getattr(module, attr)
                    if isinstance(obj, type) and hasattr(obj, "predict"):
                        self._loaded_models.append({"name": name, "instance": obj()})
            except Exception:
                continue

    def get_loaded_models(self):
        return self._loaded_models

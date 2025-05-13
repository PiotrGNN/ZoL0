# core/market_data/cache.py

class DataCache:
    def __init__(self, path):
        self.path = path
    def get(self, *args, **kwargs):
        return None
    def set(self, *args, **kwargs):
        pass

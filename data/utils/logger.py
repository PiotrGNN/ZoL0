import logging

class LogHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def setup_logging(level=logging.INFO, log_file=None):
    logger = logging.getLogger('trading_system')
    logger.setLevel(level)
    handler = LogHandler()
    logger.handlers = [handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)
    return logger

"""
Inicjalizacja pakietu execution.
"""

# Lista modułów eksportowanych przez pakiet
__all__ = [
    "bybit_connector",
    "exchange_connector",
    "order_execution",
    "trade_executor",
    "advanced_order_execution",
    "latency_optimizer"
]

# Proste importy bezpośrednie
try:
    from .bybit_connector import BybitConnector
    from .exchange_connector import ExchangeConnector
    from .order_execution import OrderExecution
    from .trade_executor import TradeExecutor
    from .advanced_order_execution import AdvancedOrderExecution
    from .latency_optimizer import LatencyOptimizer
except ImportError as e:
    import logging
    logging.warning(f"Nie udało się zaimportować niektórych modułów z pakietu execution: {e}")
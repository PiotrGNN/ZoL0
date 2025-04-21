"""
Moduł autonomicznego systemu AI dla ZoL0.

Ten moduł implementuje autonomiczny system handlu oparty na meta-agencie koordynującym
decyzje różnych modeli AI, z adaptacyjnym zarządzaniem ryzykiem i wyjaśnialnością.

Główne komponenty:
- Meta-Agent: koordynuje decyzje wszystkich modeli AI
- Vector Memory: przechowuje konteksty decyzyjne i umożliwia uczenie się na podstawie doświadczeń
- Autonomous Risk Manager: adaptacyjnie zarządza ryzykiem i wielkością pozycji
- AI Explainer: generuje zrozumiałe wyjaśnienia decyzji AI
- Autonomous Trading: główny moduł integrujący wszystkie komponenty

Sposób użycia:
```python
from ai_models.autonomous.autonomous_trading import AutonomousTrading

# Inicjalizacja systemu
auto_system = AutonomousTrading()
auto_system.initialize()

# Przetwarzanie danych rynkowych
result = auto_system.process_market_data(market_data, context)
decision = result['decision']
explanation = result['explanation']

# Aktualizacja systemu o wyniki transakcji
auto_system.update_trade_result({
    'profit_pct': 2.5,
    'max_drawdown': 0.8,
    'position_risk': 0.02
})

# Generowanie raportów i wizualizacji
report = auto_system.generate_performance_report()
visuals = auto_system.generate_visualizations()
```

Autor: ZoL0 Team
Data: kwiecień 2025
"""

from ai_models.autonomous.meta_agent import MetaAgent
from ai_models.autonomous.vector_memory import VectorMemory
from ai_models.autonomous.risk_manager import AutonomousRiskManager
from ai_models.autonomous.explainable_ai import AIExplainer
from ai_models.autonomous.autonomous_trading import AutonomousTrading, ModelWrapper

__all__ = [
    'MetaAgent',
    'VectorMemory',
    'AutonomousRiskManager', 
    'AIExplainer',
    'AutonomousTrading',
    'ModelWrapper'
]

__version__ = "1.0.0"
"""
dashboard_api_auto.py - Blueprint Flask dla autonomicznego systemu AI

Ten moduł zawiera endpointy API dla autonomicznego systemu tradingowego
opartego na AI, które są używane przez dashboard ZoL0.
"""

import os
import json
import logging
import traceback
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/autonomous_api.log")
    ]
)
logger = logging.getLogger(__name__)

# Importy dla JWT
try:
    from flask import current_app
    from functools import wraps
except ImportError:
    logger.error("Brak wymaganych importów dla JWT")

# Funkcje pomocnicze dla autoryzacji
def jwt_required(f):
    """Dekorator wymagający ważnego tokenu JWT"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return jsonify({"message": "Token is missing"}), 401
                
            token = auth_header.split(" ")[1]
            # W rzeczywistej implementacji tutaj sprawdzalibyśmy token
            # Na potrzeby przykładu zakładamy, że token jest prawidłowy
            request.username = "admin"  # Zazwyczaj wyciągane z tokenu
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Błąd autoryzacji: {e}")
            return jsonify({"message": "Invalid token"}), 401
    return decorated_function

# Inicjalizacja autonomicznego systemu AI
try:
    from ai_models.autonomous.autonomous_trading import AutonomousTrading
    from ai_models.autonomous.meta_agent import MetaAgent
    from ai_models.autonomous.risk_manager import AutonomousRiskManager
    
    # Utworzenie instancji systemu
    autonomous_system = AutonomousTrading()
    autonomous_system.initialize()
    logger.info("Zainicjalizowano autonomiczny system AI w blueprint")
    
    autonomous_system_available = True
except ImportError as e:
    logger.warning(f"Nie można zaimportować modułów autonomicznego systemu: {e}")
    autonomous_system = None
    autonomous_system_available = False
except Exception as e:
    logger.error(f"Błąd podczas inicjalizacji autonomicznego systemu: {e}")
    logger.error(traceback.format_exc())
    autonomous_system = None
    autonomous_system_available = False

# Utworzenie Blueprint
auto_api = Blueprint('autonomous_api', __name__)

@auto_api.route('/api/autonomous/system-info', methods=['GET'])
@jwt_required
def get_autonomous_system_info():
    """Zwraca podstawowe informacje o autonomicznym systemie."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "available": False,
                "error": "System autonomiczny nie jest dostępny"
            })
        
        return jsonify({
            "available": True,
            "version": "1.0.0",
            "status": "active" if autonomous_system.is_initialized else "inactive",
            "components": ["meta_agent", "risk_manager", "explainer", "vector_memory"],
            "meta_agent_status": {
                "models_count": len(autonomous_system.meta_agent.models_state),
                "memory_size": autonomous_system.meta_agent.memory.get_statistics().get("memory_size", 0)
            }
        })
    except Exception as e:
        logger.error(f"Błąd podczas pobierania informacji o systemie autonomicznym: {e}")
        return jsonify({
            "available": False,
            "error": str(e)
        })

@auto_api.route('/api/autonomous/register-model', methods=['POST'])
@jwt_required
def register_model():
    """Rejestruje nowy model w Meta-Agencie."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "success": False,
                "error": "System autonomiczny nie jest dostępny"
            }), 503
        
        data = request.json
        if not data or "model_name" not in data or "model_path" not in data:
            return jsonify({
                "success": False,
                "error": "Nieprawidłowe dane. Wymagane pola: 'model_name' i 'model_path'"
            }), 400
        
        model_name = data["model_name"]
        model_path = data["model_path"]
        
        # Zarejestruj model w systemie
        # (W rzeczywistej implementacji ładowalibyśmy model z pliku i rejestrowali go)
        autonomous_system.meta_agent.models_state[model_name] = {
            "weight": data.get("weight", 1.0),
            "enabled": True,
            "performance": {
                "decisions_count": 0,
                "correct_decisions": 0,
                "incorrect_decisions": 0,
                "profit_sum": 0.0,
                "loss_sum": 0.0,
                "last_decisions": []
            }
        }
        
        logger.info(f"Zarejestrowano nowy model: {model_name}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Błąd podczas rejestracji modelu: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@auto_api.route('/api/autonomous/models', methods=['GET'])
@jwt_required
def get_autonomous_models():
    """Zwraca listę dostępnych modeli w systemie autonomicznym."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "models": [],
                "error": "System autonomiczny nie jest dostępny"
            })
        
        models = []
        for model_name, model_state in autonomous_system.meta_agent.models_state.items():
            perf = model_state["performance"]
            
            # Oblicz statystyki
            total_decisions = perf["decisions_count"]
            accuracy = perf["correct_decisions"] / max(total_decisions, 1)
            avg_profit = perf["profit_sum"] / max(total_decisions, 1)
            
            models.append({
                "name": model_name,
                "weight": model_state["weight"],
                "enabled": model_state["enabled"],
                "accuracy": accuracy * 100,
                "avg_profit": avg_profit,
                "total_decisions": total_decisions,
                "last_update": datetime.now().isoformat()
            })
        
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Błąd podczas pobierania listy modeli: {e}")
        return jsonify({
            "models": [],
            "error": str(e)
        })

@auto_api.route('/api/autonomous/models/<model_name>', methods=['PUT'])
@jwt_required
def update_model_status(model_name):
    """Aktualizuje status modelu (włączony/wyłączony)."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "success": False,
                "error": "System autonomiczny nie jest dostępny"
            }), 503
        
        if model_name not in autonomous_system.meta_agent.models_state:
            return jsonify({
                "success": False,
                "error": f"Model '{model_name}' nie istnieje"
            }), 404
        
        data = request.json
        if not data or "enabled" not in data:
            return jsonify({
                "success": False,
                "error": "Nieprawidłowe dane. Wymagane pole: 'enabled'"
            }), 400
        
        # Aktualizuj status modelu
        autonomous_system.meta_agent.models_state[model_name]["enabled"] = bool(data["enabled"])
        
        logger.info(f"Zaktualizowano status modelu {model_name}: enabled={data['enabled']}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji statusu modelu: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@auto_api.route('/api/autonomous/process-market-data', methods=['POST'])
@jwt_required
def process_market_data():
    """Przetwarza dane rynkowe przez system autonomiczny."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "success": False,
                "error": "System autonomiczny nie jest dostępny"
            }), 503
        
        data = request.json
        if not data or "market_data" not in data:
            return jsonify({
                "success": False,
                "error": "Nieprawidłowe dane. Wymagane pole: 'market_data'"
            }), 400
        
        # Przetwórz dane rynkowe
        result = autonomous_system.process_market_data(data["market_data"], data.get("context", {}))
        
        return jsonify({
            "success": True,
            "result": result
        })
    except Exception as e:
        logger.error(f"Błąd podczas przetwarzania danych rynkowych: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@auto_api.route('/api/autonomous/update-trade-result', methods=['POST'])
@jwt_required
def update_trade_result():
    """Aktualizuje wynik transakcji w systemie autonomicznym."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "success": False,
                "error": "System autonomiczny nie jest dostępny"
            }), 503
        
        data = request.json
        if not data:
            return jsonify({
                "success": False,
                "error": "Nieprawidłowe dane"
            }), 400
        
        # Aktualizuj wynik transakcji
        autonomous_system.update_trade_result(data)
        
        logger.info(f"Zaktualizowano wynik transakcji: profit_pct={data.get('profit_pct', 'N/A')}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji wyniku transakcji: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@auto_api.route('/api/autonomous/memory-stats', methods=['GET'])
@jwt_required
def get_memory_stats():
    """Zwraca statystyki pamięci wektorowej."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "stats": {},
                "error": "System autonomiczny nie jest dostępny"
            })
        
        # Pobierz statystyki pamięci
        memory_stats = autonomous_system.meta_agent.memory.get_statistics() if hasattr(autonomous_system.meta_agent.memory, 'get_statistics') else {}
        
        return jsonify({"stats": memory_stats})
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statystyk pamięci: {e}")
        return jsonify({
            "stats": {},
            "error": str(e)
        })

@auto_api.route('/api/autonomous/performance-report', methods=['GET'])
@jwt_required
def get_performance_report():
    """Generuje raport wydajności autonomicznego systemu."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "report": {},
                "error": "System autonomiczny nie jest dostępny"
            })
        
        # Wygeneruj raport wydajności
        report = autonomous_system.generate_performance_report()
        
        return jsonify({"report": report})
    except Exception as e:
        logger.error(f"Błąd podczas generowania raportu wydajności: {e}")
        return jsonify({
            "report": {},
            "error": str(e)
        })

@auto_api.route('/api/autonomous/reset', methods=['POST'])
@jwt_required
def reset_autonomous_system():
    """Resetuje stan systemu autonomicznego."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "success": False,
                "error": "System autonomiczny nie jest dostępny"
            }), 503
        
        # Zainicjalizuj system ponownie
        autonomous_system.initialize()
        
        logger.info("Zresetowano stan systemu autonomicznego")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Błąd podczas resetowania systemu autonomicznego: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@auto_api.route('/api/autonomous/save-state', methods=['POST'])
@jwt_required
def save_autonomous_state():
    """Zapisuje stan systemu autonomicznego."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "success": False,
                "error": "System autonomiczny nie jest dostępny"
            }), 503
        
        # Zapisz stan systemu
        autonomous_system.save_state()
        
        logger.info("Zapisano stan systemu autonomicznego")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania stanu systemu autonomicznego: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@auto_api.route('/api/autonomous/quotes', methods=['GET'])
@jwt_required
def get_ai_quotes():
    """Zwraca inspirujące cytaty od autonomicznego systemu AI."""
    try:
        quotes = [
            "Przyszłość tradingu należy do autonomicznych systemów AI.",
            "Adaptatywne strategie przetrwają w burzliwych warunkach rynkowych.",
            "Uczenie przez doświadczenie to klucz do sukcesu w tradingu algorytmicznym.",
            "Większa autonomia oznacza mniejszy wpływ emocji na decyzje handlowe.",
            "System, który się nie rozwija, staje się przestarzały.",
            "Prawdziwa inteligencja to zdolność adaptacji do zmiennych warunków.",
            "Analiza danych bez autonomicznego działania to tylko teoria.",
            "W tradingu liczy się nie ilość decyzji, ale ich jakość.",
            "Autonomia to balans między odważnymi decyzjami a zarządzaniem ryzykiem.",
            "Przyszłość należy do systemów, które potrafią się uczyć na własnych błędach."
        ]
        
        # Wybierz losowy cytat
        selected_quote = random.choice(quotes)
        
        return jsonify({
            "quote": selected_quote,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Błąd podczas pobierania cytatów AI: {e}")
        return jsonify({
            "quote": "Nawet AI ma czasem problemy techniczne...",
            "error": str(e)
        })

@auto_api.route('/api/autonomous/recommendation', methods=['GET'])
@jwt_required
def get_autonomous_recommendation():
    """Zwraca rekomendację dotyczącą konfiguracji autonomicznego systemu."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "recommendation": {},
                "error": "System autonomiczny nie jest dostępny"
            })
        
        # Symuluj analizę i rekomendację
        if autonomous_system.meta_agent and hasattr(autonomous_system.meta_agent, 'models_state'):
            # Znajdź model z najniższą wagą
            worst_model = min(
                autonomous_system.meta_agent.models_state.items(),
                key=lambda x: x[1]['weight']
            )[0] if autonomous_system.meta_agent.models_state else "N/A"
            
            # Znajdź model z najwyższą wagą
            best_model = max(
                autonomous_system.meta_agent.models_state.items(),
                key=lambda x: x[1]['weight']
            )[0] if autonomous_system.meta_agent.models_state else "N/A"
            
            recommendation = {
                "suggestions": [
                    f"Rozważ zmniejszenie wagi modelu '{worst_model}' lub jego ponowne przeszkolenie.",
                    f"Model '{best_model}' osiąga najlepsze wyniki - rozważ zwiększenie jego wagi.",
                    "Dodaj więcej transakcji do pamięci wektorowej, aby poprawić podobne decyzje.",
                    "Dostosuj poziom ryzyka w zależności od aktualnej zmienności rynku."
                ],
                "risk_profile": "moderate",
                "timestamp": datetime.now().isoformat()
            }
        else:
            recommendation = {
                "suggestions": [
                    "Zainicjalizuj system i dodaj więcej modeli, aby uzyskać pełne rekomendacje.",
                    "Rozpocznij od małych wielkości pozycji, aby przetestować system w praktyce.",
                    "Ustaw poziom ryzyka adekwatnie do swojej tolerancji."
                ],
                "risk_profile": "conservative",
                "timestamp": datetime.now().isoformat()
            }
        
        return jsonify({"recommendation": recommendation})
    except Exception as e:
        logger.error(f"Błąd podczas generowania rekomendacji: {e}")
        return jsonify({
            "recommendation": {},
            "error": str(e)
        })

@auto_api.route('/api/autonomous/monitor/start', methods=['POST'])
@jwt_required
def start_autonomous_monitoring():
    """Uruchamia monitoring autonomicznego systemu."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "success": False,
                "error": "System autonomiczny nie jest dostępny"
            }), 503
        
        data = request.json
        interval = data.get('interval', 3600) if data else 3600  # Domyślnie co godzinę
        
        # Uruchom monitoring
        if hasattr(autonomous_system.meta_agent, 'start_monitoring'):
            autonomous_system.meta_agent.start_monitoring(interval)
            logger.info(f"Uruchomiono monitoring z interwałem {interval} sekund")
            return jsonify({
                "success": True,
                "message": f"Uruchomiono monitoring z interwałem {interval} sekund"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Meta-Agent nie obsługuje monitoringu"
            }), 501
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania monitoringu: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@auto_api.route('/api/autonomous/monitor/stop', methods=['POST'])
@jwt_required
def stop_autonomous_monitoring():
    """Zatrzymuje monitoring autonomicznego systemu."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "success": False,
                "error": "System autonomiczny nie jest dostępny"
            }), 503
        
        # Zatrzymaj monitoring
        if hasattr(autonomous_system.meta_agent, 'stop_monitoring'):
            autonomous_system.meta_agent.stop_monitoring()
            logger.info("Zatrzymano monitoring")
            return jsonify({
                "success": True,
                "message": "Zatrzymano monitoring"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Meta-Agent nie obsługuje monitoringu"
            }), 501
    except Exception as e:
        logger.error(f"Błąd podczas zatrzymywania monitoringu: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@auto_api.route('/api/autonomous/training/start', methods=['POST'])
@jwt_required
def start_autonomous_training():
    """Uruchamia trening autonomicznego systemu."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "success": False,
                "error": "System autonomiczny nie jest dostępny"
            }), 503
        
        data = request.json
        
        # Tutaj dodamy rzeczywistą implementację treningu, ale na potrzeby
        # przykładu zakładamy, że trening został uruchomiony
        logger.info("Uruchomiono trening autonomicznego systemu")
        
        return jsonify({
            "success": True,
            "message": "Uruchomiono trening",
            "training_id": f"training_{int(datetime.now().timestamp())}",
            "estimated_completion": (datetime.now() + timedelta(hours=2)).isoformat()
        })
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania treningu: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@auto_api.route('/api/autonomous/visualizations', methods=['GET'])
@jwt_required
def get_autonomous_visualizations():
    """Zwraca wizualizacje wydajności autonomicznego systemu."""
    try:
        if not autonomous_system_available:
            return jsonify({
                "visualizations": {},
                "error": "System autonomiczny nie jest dostępny"
            })
        
        # Generuj wizualizacje
        if hasattr(autonomous_system, 'generate_visualizations'):
            visualizations = autonomous_system.generate_visualizations()
        else:
            # Na potrzeby przykładu tworzymy dummy data
            visualizations = {
                "model_weights": {
                    "type": "image",
                    "format": "base64",
                    "data": "dummy_base64_data",
                    "alt": "Wagi modeli w Meta-Agencie"
                },
                "decision_history": {
                    "type": "image",
                    "format": "base64",
                    "data": "dummy_base64_data",
                    "alt": "Historia decyzji Meta-Agenta"
                }
            }
        
        return jsonify({"visualizations": visualizations})
    except Exception as e:
        logger.error(f"Błąd podczas generowania wizualizacji: {e}")
        return jsonify({
            "visualizations": {},
            "error": str(e)
        })

# Dodatkowe endpointy można dodawać poniżej
"""
Moduł Meta-Agenta - centralnego koordynatora systemu autonomicznego AI.

Meta-Agent odpowiada za:
1. Integrację i koordynację decyzji różnych modeli AI
2. Analizę historycznych decyzji i kontekstów poprzez pamięć wektorową
3. Dynamiczną adaptację strategii handlowej na podstawie aktualnych warunków rynkowych
4. Ciągłe uczenie się i optymalizację strategii handlowych

Ten moduł stanowi "mózg" autonomicznego systemu AI w ZoL0.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import traceback
import threading
import time
from collections import defaultdict

# Import modułów autonomicznego AI
from ai_models.autonomous.vector_memory import VectorMemory

# Domyślna konfiguracja dla modułów AI
DEFAULT_MODELS_CONFIG = {
    "RandomForest": {"weight": 1.0, "enabled": True},
    "SentimentAnalyzer": {"weight": 0.8, "enabled": True},
    "AnomalyDetector": {"weight": 0.7, "enabled": True},
    "ModelRecognizer": {"weight": 1.2, "enabled": True},
    "RLAgent": {"weight": 1.0, "enabled": False},  # Domyślnie wyłączony
}

# Konfiguracja logowania
logger = logging.getLogger(__name__)

class MetaAgent:
    """
    Meta-Agent - centralny koordynator autonomicznego systemu AI.
    
    Odpowiada za integrację decyzji różnych modeli AI oraz dynamiczną
    adaptację strategii handlowej na podstawie aktualnych warunków rynkowych.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicjalizuje Meta-Agenta.
        
        Args:
            config: Konfiguracja Meta-Agenta
        """
        self.config = config or {}
        
        # Inicjalizacja konfiguracji z wartościami domyślnymi jeśli brakuje kluczy
        if "models_config" not in self.config:
            self.config["models_config"] = DEFAULT_MODELS_CONFIG.copy()
            logger.warning("Nie znaleziono konfiguracji 'models_config' - używam domyślnej")
        
        # Inicjalizacja pamięci wektorowej
        self.memory = VectorMemory({
            "memory_size": self.config.get("memory_size", 10000),
            "embedding_dim": self.config.get("embedding_dim", 64)
        })
        
        # Stan modeli i ich skuteczność
        self.models_state = {
            model_name: {
                "weight": model_config["weight"],
                "enabled": model_config["enabled"],
                "performance": {
                    "decisions_count": 0,
                    "correct_decisions": 0,
                    "incorrect_decisions": 0,
                    "profit_sum": 0.0,
                    "loss_sum": 0.0,
                    "last_decisions": []  # Lista ostatnich decyzji i ich wyników
                }
            }
            for model_name, model_config in self.config["models_config"].items()
        }
        
        # Aktualny stan rynku i pozycji
        self.market_state = {
            "open_positions": [],
            "market_condition": "NEUTRAL",  # BULLISH, BEARISH, NEUTRAL, VOLATILE
            "last_signals": [],
            "volatility": 0.0,
            "trend": 0.0
        }
        
        # Pamięć ostatnich decyzji
        self.recent_decisions = []
        
        # Wątki monitorujące
        self.monitoring_thread = None
        self.is_monitoring = False
        
        logger.info("Zainicjalizowano Meta-Agenta systemu autonomicznego AI")
    
    def _update_model_weights(self):
        """
        Aktualizuje wagi modeli na podstawie ich skuteczności.
        """
        if not self.config.get("adaptive_learning", True):
            logger.debug("Adaptacyjne uczenie jest wyłączone, wagi modeli pozostają bez zmian")
            return
        
        for model_name, state in self.models_state.items():
            perf = state["performance"]
            
            # Jeśli model podjął wystarczającą liczbę decyzji
            if perf["decisions_count"] >= self.config.get("model_performance_window", 50):
                # Wylicz skuteczność (accuracy)
                accuracy = perf["correct_decisions"] / max(perf["decisions_count"], 1)
                
                # Wylicz średni profit/loss na decyzję
                avg_profit = perf["profit_sum"] / max(perf["decisions_count"], 1)
                
                # Aktualizuj wagę na podstawie skuteczności i średniego zysku
                old_weight = state["weight"]
                
                # Nowa waga zależy od accuracy i avg_profit
                new_weight = old_weight * (1.0 + self.config["learning_rate"] * (accuracy - 0.5) * 2)
                new_weight = new_weight * (1.0 + self.config["learning_rate"] * avg_profit * 5)
                
                # Ogranicz wagę do sensownego zakresu (0.1 - 3.0)
                new_weight = max(0.1, min(3.0, new_weight))
                
                # Aktualizuj wagę
                state["weight"] = new_weight
                
                logger.info(f"Zaktualizowano wagę modelu {model_name}: {old_weight:.2f} -> {new_weight:.2f} "
                           f"(accuracy: {accuracy:.2f}, avg_profit: {avg_profit:.4f})")
                
                # Zresetuj część statystyk, zachowując ostatnie decyzje
                perf["decisions_count"] = len(perf["last_decisions"])
                perf["correct_decisions"] = sum(1 for d in perf["last_decisions"] if d.get("correct", False))
                perf["incorrect_decisions"] = sum(1 for d in perf["last_decisions"] if not d.get("correct", True))
                perf["profit_sum"] = sum(d.get("profit", 0) for d in perf["last_decisions"])
                perf["loss_sum"] = sum(d.get("loss", 0) for d in perf["last_decisions"])
    
    def _update_market_condition(self, market_data: Dict[str, Any]):
        """
        Aktualizuje ocenę stanu rynku na podstawie danych rynkowych.
        
        Args:
            market_data: Dane rynkowe
        """
        try:
            # Pobierz dane cenowe
            closes = market_data.get("close", [])
            if len(closes) < 20:
                logger.warning("Za mało danych cenowych do analizy stanu rynku")
                return
                
            # Wylicz proste metryki
            closes = np.array(closes[-100:])
            returns = np.diff(closes) / closes[:-1]
            
            # Wylicz zmienność (volatility)
            volatility = np.std(returns) * np.sqrt(252)  # Annualizowana zmienność
            
            # Wylicz kierunek trendu
            sma20 = np.mean(closes[-20:])
            sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma20
            trend = (sma20 / sma50) - 1.0
            
            # Określ stan rynku
            if volatility > 0.03:  # Ponad 3% dziennej zmienności
                market_condition = "VOLATILE"
            elif trend > 0.01:  # Trend wzrostowy (ponad 1%)
                market_condition = "BULLISH"
            elif trend < -0.01:  # Trend spadkowy (poniżej -1%)
                market_condition = "BEARISH"
            else:
                market_condition = "NEUTRAL"
            
            # Aktualizuj stan rynku
            self.market_state["market_condition"] = market_condition
            self.market_state["volatility"] = float(volatility)
            self.market_state["trend"] = float(trend)
            
            logger.debug(f"Zaktualizowano stan rynku: {market_condition} (volatility: {volatility:.4f}, trend: {trend:.4f})")
                
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji stanu rynku: {str(e)}")
            logger.error(traceback.format_exc())
    
    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Przetwarza dane rynkowe i aktualizuje stan rynku.
        
        Args:
            market_data: Dane rynkowe w formacie słownika
            
        Returns:
            Zaktualizowany stan rynku
        """
        try:
            # Aktualizuj stan rynku
            self._update_market_condition(market_data)
            
            return self.market_state
            
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania danych rynkowych: {str(e)}")
            logger.error(traceback.format_exc())
            return self.market_state
    
    def evaluate_models_decision(self, models_decisions: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ocenia i integruje decyzje różnych modeli AI.
        
        Args:
            models_decisions: Decyzje poszczególnych modeli
            market_data: Dane rynkowe
            
        Returns:
            Zintegrowana decyzja zawierająca:
            - final_decision: Ostateczna decyzja (BUY/SELL/HOLD)
            - confidence: Poziom pewności (0-1)
            - models_contribution: Wkład poszczególnych modeli
            - context_similarity: Podobieństwo do historycznych kontekstów
        """
        try:
            # Przygotuj aktualny kontekst decyzyjny
            context = {
                "model_decisions": models_decisions,
                "market_data": market_data,
                "market_state": self.market_state,
                "timestamp": datetime.now().isoformat()
            }
            
            # Wyszukaj podobne konteksty w pamięci
            similar_contexts = []
            if self.config.get("use_historical_context", True):
                similar_contexts = self.memory.search_similar(context, k=5)
            
            # Inicjalizuj wagi głosów
            vote_weights = {
                "BUY": 0.0,
                "SELL": 0.0,
                "HOLD": 0.0
            }
            
            # Zbierz głosy modeli, uwzględniając ich wagi
            models_contribution = {}
            total_weight = 0.0
            
            for model_name, decision in models_decisions.items():
                # Pobierz konfigurację i stan modelu
                model_state = self.models_state.get(model_name, {
                    "weight": 1.0,
                    "enabled": True
                })
                
                if not model_state.get("enabled", True):
                    continue
                
                model_weight = model_state.get("weight", 1.0)
                total_weight += model_weight
                
                # Mapuj różne formaty decyzji na standardowe
                std_decision = "HOLD"  # Domyślna decyzja
                confidence = 0.5  # Domyślny poziom pewności
                
                # Przetwórz decyzję w różnych formatach
                if isinstance(decision, str):
                    decision_upper = decision.upper()
                    if decision_upper in ["BUY", "LONG"]:
                        std_decision = "BUY"
                        confidence = 0.8  # Domyślna pewność jeśli nie podano
                    elif decision_upper in ["SELL", "SHORT"]:
                        std_decision = "SELL"
                        confidence = 0.8
                    else:
                        std_decision = "HOLD"
                        confidence = 0.6
                        
                elif isinstance(decision, dict):
                    # Format: {"decision": "BUY", "confidence": 0.9}
                    decision_val = decision.get("decision", "HOLD").upper()
                    confidence = decision.get("confidence", 0.5)
                    
                    if decision_val in ["BUY", "LONG"]:
                        std_decision = "BUY"
                    elif decision_val in ["SELL", "SHORT"]:
                        std_decision = "SELL"
                    else:
                        std_decision = "HOLD"
                
                # Uwzględnij wagę modelu i pewność w głosowaniu
                vote_weight = model_weight * confidence
                vote_weights[std_decision] += vote_weight
                
                # Zapisz wkład modelu
                models_contribution[model_name] = {
                    "decision": std_decision,
                    "confidence": confidence,
                    "weight": model_weight,
                    "vote_weight": vote_weight
                }
            
            # Uwzględnij podobne konteksty historyczne w głosowaniu
            context_contribution = {}
            for similar in similar_contexts:
                sim_context = similar.get("context", {})
                sim_decision = sim_context.get("final_decision", "HOLD")
                sim_result = similar.get("metadata", {}).get("result", {})
                similarity = similar.get("similarity", 0.0)
                
                # Uwzględnij tylko konteksty z wystarczającym podobieństwem
                if similarity >= self.config.get("context_similarity_threshold", 0.8):
                    # Jeśli jest wynik, sprawdź, czy decyzja była dobra
                    if sim_result:
                        profit = sim_result.get("profit_pct", 0.0)
                        # Jeśli profit jest dodatni, to decyzja była dobra
                        decision_weight = similarity * (1.0 + min(1.0, profit * 5))
                        vote_weights[sim_decision] += decision_weight
                        
                        context_contribution[similar.get("metadata", {}).get("id", "unknown")] = {
                            "decision": sim_decision,
                            "similarity": similarity,
                            "profit": profit,
                            "weight": decision_weight
                        }
            
            # Znormalizuj głosy
            total_votes = sum(vote_weights.values())
            if total_votes > 0:
                for decision in vote_weights:
                    vote_weights[decision] /= total_votes
            
            # Wybierz decyzję z największą wagą głosów
            final_decision = max(vote_weights, key=vote_weights.get)
            confidence = vote_weights[final_decision]
            
            # Jeśli pewność jest poniżej progu, zmień na HOLD
            if confidence < self.config.get("decision_threshold", 0.6):
                final_decision = "HOLD"
                confidence = vote_weights["HOLD"]
            
            # Dodaj kontekst decyzyjny do pamięci
            context["final_decision"] = final_decision
            self.memory.store(context)
            
            # Dodaj decyzję do listy ostatnich decyzji
            self.recent_decisions.append({
                "timestamp": datetime.now().isoformat(),
                "decision": final_decision,
                "confidence": confidence,
                "models_decisions": models_decisions
            })
            
            # Ogranicz listę do ostatnich 100 decyzji
            if len(self.recent_decisions) > 100:
                self.recent_decisions = self.recent_decisions[-100:]
            
            # Przygotuj wynik
            result = {
                "final_decision": final_decision,
                "confidence": confidence,
                "vote_weights": vote_weights,
                "models_contribution": models_contribution,
                "context_similarity": {
                    "similar_contexts_count": len(similar_contexts),
                    "context_contribution": context_contribution
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Meta-Agent: Decyzja finalna: {final_decision} (pewność: {confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Błąd podczas ewaluacji decyzji modeli: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "final_decision": "HOLD",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def update_decision_outcome(self, decision_id: str, outcome: Dict[str, Any]) -> bool:
        """
        Aktualizuje wynik decyzji w pamięci i ocenianie modeli.
        
        Args:
            decision_id: ID decyzji
            outcome: Wynik decyzji (profit, czas trwania itp.)
            
        Returns:
            True jeśli zaktualizowano pomyślnie, False w przeciwnym wypadku
        """
        try:
            # Pobierz kontekst z pamięci
            context_data = self.memory.get_context_by_id(decision_id)
            if not context_data:
                logger.warning(f"Nie znaleziono kontekstu o ID: {decision_id}")
                return False
            
            context = context_data.get("context", {})
            
            # Aktualizuj wynik w pamięci
            self.memory.update_result(decision_id, outcome)
            
            # Oceń poprawność decyzji modeli
            final_decision = context.get("final_decision")
            profit = outcome.get("profit_pct", 0.0)
            
            # Aktualizuj statystyki modeli
            models_decisions = context.get("model_decisions", {})
            for model_name, decision in models_decisions.items():
                if model_name not in self.models_state:
                    continue
                
                model_perf = self.models_state[model_name]["performance"]
                
                # Zwiększ licznik decyzji
                model_perf["decisions_count"] += 1
                
                # Przetwórz decyzję modelu do standardowego formatu
                std_decision = "HOLD"
                if isinstance(decision, str):
                    decision_upper = decision.upper()
                    if decision_upper in ["BUY", "LONG"]:
                        std_decision = "BUY"
                    elif decision_upper in ["SELL", "SHORT"]:
                        std_decision = "SELL"
                elif isinstance(decision, dict):
                    decision_val = decision.get("decision", "HOLD").upper()
                    if decision_val in ["BUY", "LONG"]:
                        std_decision = "BUY"
                    elif decision_val in ["SELL", "SHORT"]:
                        std_decision = "SELL"
                
                # Sprawdź, czy decyzja modelu była zgodna z finalną
                if std_decision == final_decision:
                    # Jeśli finalna decyzja przyniosła zysk, to decyzja modelu była dobra
                    if (final_decision == "BUY" and profit > 0) or (final_decision == "SELL" and profit < 0):
                        model_perf["correct_decisions"] += 1
                        model_perf["profit_sum"] += abs(profit)
                    else:
                        model_perf["incorrect_decisions"] += 1
                        model_perf["loss_sum"] += abs(profit)
                
                # Zapisz decyzję w historii
                model_perf["last_decisions"].append({
                    "decision": std_decision,
                    "final_decision": final_decision,
                    "profit": profit,
                    "correct": (std_decision == final_decision and 
                            ((final_decision == "BUY" and profit > 0) or 
                             (final_decision == "SELL" and profit < 0)))
                })
                
                # Ogranicz historię do ostatnich 50 decyzji
                if len(model_perf["last_decisions"]) > 50:
                    model_perf["last_decisions"] = model_perf["last_decisions"][-50:]
            
            # Aktualizuj wagi modeli
            self._update_model_weights()
            
            logger.info(f"Zaktualizowano wynik decyzji ID: {decision_id}, profit: {profit:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji wyniku decyzji: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def get_position_sizing(self, decision: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wyznacza rozmiar pozycji na podstawie decyzji, ryzyka i stanu portfela.
        
        Args:
            decision: Decyzja handlowa
            portfolio: Stan portfela
            
        Returns:
            Parametry pozycji (rozmiar, wartość, stop loss, take profit)
        """
        try:
            # Pobierz parametry ryzyka z konfiguracji
            risk_config = self.config.get("risk_management", {})
            max_position_size_pct = risk_config.get("max_position_size_pct", 0.1)
            risk_per_trade_pct = risk_config.get("risk_per_trade_pct", 0.02)
            dynamic_position_sizing = risk_config.get("dynamic_position_sizing", True)
            
            # Pobierz wartość portfela
            portfolio_value = portfolio.get("total_value", 10000.0)
            
            # Pobierz pewność decyzji
            confidence = decision.get("confidence", 0.5)
            
            # Bazowy rozmiar pozycji (% portfela)
            base_position_size_pct = risk_per_trade_pct * 10  # 2% ryzyka -> 20% wielkości pozycji
            
            # Dostosuj rozmiar pozycji na podstawie pewności decyzji jeśli włączono dynamiczne wymiarowanie
            if dynamic_position_sizing:
                # Skaluj rozmiar pozycji na podstawie pewności (0.5-1.0) -> (0.5-1.5) mnożnik
                confidence_factor = 0.5 + confidence
                position_size_pct = base_position_size_pct * confidence_factor
            else:
                position_size_pct = base_position_size_pct
            
            # Ogranicz do maksymalnego rozmiaru pozycji
            position_size_pct = min(position_size_pct, max_position_size_pct)
            
            # Oblicz wartość pozycji
            position_value = portfolio_value * position_size_pct
            
            # Oblicz parametry SL/TP
            # Bazowy poziom stop loss jako % wartości pozycji
            stop_loss_pct = risk_per_trade_pct / position_size_pct
            
            # Take profit jako wielokrotność stop loss (np. 2:1)
            take_profit_pct = stop_loss_pct * 2
            
            # Dostosuj SL/TP na podstawie warunków rynkowych
            market_condition = self.market_state.get("market_condition", "NEUTRAL")
            volatility = self.market_state.get("volatility", 0.01)
            
            if market_condition == "VOLATILE":
                # W zmiennym rynku zwiększ SL, aby uniknąć przedwczesnego zamknięcia
                stop_loss_pct *= 1.5
                # Zwiększ również TP
                take_profit_pct *= 1.2
            elif market_condition == "BULLISH" and decision.get("final_decision") == "BUY":
                # W trendzie wzrostowym dla pozycji długich zwiększ TP
                take_profit_pct *= 1.3
            elif market_condition == "BEARISH" and decision.get("final_decision") == "SELL":
                # W trendzie spadkowym dla pozycji krótkich zwiększ TP
                take_profit_pct *= 1.3
            
            # Dostosuj SL/TP do zmienności rynku
            volatility_factor = min(2.0, max(0.5, volatility / 0.01))  # Bazowo zakładamy zmienność 1%
            stop_loss_pct *= volatility_factor
            take_profit_pct *= volatility_factor
            
            # Wynik
            result = {
                "position_size_pct": float(position_size_pct),
                "position_value": float(position_value),
                "stop_loss_pct": float(stop_loss_pct),
                "take_profit_pct": float(take_profit_pct),
                "risk_per_trade_pct": float(risk_per_trade_pct)
            }
            
            logger.debug(f"Wyliczono parametry pozycji: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Błąd podczas wyznaczania rozmiaru pozycji: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "position_size_pct": 0.05,  # Domyślny rozmiar 5%
                "stop_loss_pct": 0.05,      # Domyślny SL 5%
                "take_profit_pct": 0.1      # Domyślny TP 10%
            }
    
    def start_monitoring(self, interval: int = 3600):
        """
        Uruchamia wątek monitorujący stan systemu.
        
        Args:
            interval: Interwał monitorowania w sekundach
        """
        if self.is_monitoring:
            logger.warning("Monitoring już jest uruchomiony")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Uruchomiono monitoring systemu z interwałem {interval} sekund")
    
    def stop_monitoring(self):
        """
        Zatrzymuje wątek monitorujący.
        """
        self.is_monitoring = False
        logger.info("Zatrzymano monitoring systemu")
    
    def _monitoring_loop(self, interval: int):
        """
        Główna pętla monitorująca.
        
        Args:
            interval: Interwał monitorowania w sekundach
        """
        while self.is_monitoring:
            try:
                # Wykonaj okresowe zadania
                self._periodic_tasks()
                
            except Exception as e:
                logger.error(f"Błąd w pętli monitorującej: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Poczekaj do następnego interwału
            time.sleep(interval)
    
    def _periodic_tasks(self):
        """
        Wykonuje okresowe zadania monitorujące i konserwacyjne.
        """
        # 1. Zapisz stan pamięci wektorowej
        try:
            self.memory.save_state()
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania stanu pamięci: {str(e)}")
        
        # 2. Wygeneruj statystyki
        try:
            memory_stats = self.memory.get_statistics()
            logger.info(f"Statystyki pamięci: {memory_stats}")
        except Exception as e:
            logger.error(f"Błąd podczas generowania statystyk pamięci: {str(e)}")
        
        # 3. Logi wydajności modeli
        try:
            for model_name, state in self.models_state.items():
                perf = state["performance"]
                if perf["decisions_count"] > 0:
                    accuracy = perf["correct_decisions"] / perf["decisions_count"]
                    avg_profit = perf["profit_sum"] / perf["decisions_count"]
                    logger.info(f"Model {model_name}: accuracy={accuracy:.2f}, avg_profit={avg_profit:.4f}, "
                              f"weight={state['weight']:.2f}, enabled={state['enabled']}")
        except Exception as e:
            logger.error(f"Błąd podczas logowania wydajności modeli: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Zwraca aktualny status systemu autonomicznego AI.
        
        Returns:
            Słownik ze statusem systemu
        """
        try:
            # Statystyki pamięci
            memory_stats = self.memory.get_statistics()
            
            # Statystyki modeli
            models_stats = {}
            for model_name, state in self.models_state.items():
                perf = state["performance"]
                accuracy = perf["correct_decisions"] / max(perf["decisions_count"], 1)
                avg_profit = perf["profit_sum"] / max(perf["decisions_count"], 1)
                
                models_stats[model_name] = {
                    "weight": state["weight"],
                    "enabled": state["enabled"],
                    "accuracy": accuracy,
                    "avg_profit": avg_profit,
                    "decisions_count": perf["decisions_count"]
                }
            
            # Ostatnie decyzje
            recent_decisions = self.recent_decisions[-10:] if self.recent_decisions else []
            
            # Status systemu
            status = {
                "autonomous_mode": self.config.get("autonomous_mode", True),
                "memory": memory_stats,
                "models": models_stats,
                "market_state": self.market_state,
                "recent_decisions": recent_decisions,
                "monitoring_active": self.is_monitoring,
                "timestamp": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania statusu systemu: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def save_state(self, filepath: str = None) -> bool:
        """
        Zapisuje stan Meta-Agenta.
        
        Args:
            filepath: Ścieżka do zapisu stanu
            
        Returns:
            True jeśli zapisano pomyślnie, False w przeciwnym wypadku
        """
        try:
            # Zapisz stan pamięci wektorowej
            memory_saved = self.memory.save_state()
            
            if filepath is None:
                # Generuj domyślną ścieżkę
                base_dir = os.path.dirname(os.path.dirname(__file__))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(base_dir, "saved_states", f"meta_agent_state_{timestamp}.json")
                
                # Utwórz katalog jeśli nie istnieje
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Przygotuj dane do zapisu
            state = {
                "config": self.config,
                "models_state": {
                    model_name: {
                        "weight": model_state["weight"],
                        "enabled": model_state["enabled"],
                        "performance": {
                            "decisions_count": model_state["performance"]["decisions_count"],
                            "correct_decisions": model_state["performance"]["correct_decisions"],
                            "incorrect_decisions": model_state["performance"]["incorrect_decisions"],
                            "profit_sum": model_state["performance"]["profit_sum"],
                            "loss_sum": model_state["performance"]["loss_sum"]
                        }
                    }
                    for model_name, model_state in self.models_state.items()
                },
                "market_state": self.market_state,
                "timestamp": datetime.now().isoformat()
            }
            
            # Zapisz do pliku
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Zapisano stan Meta-Agenta do pliku: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania stanu Meta-Agenta: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Wczytuje stan Meta-Agenta.
        
        Args:
            filepath: Ścieżka do pliku ze stanem
            
        Returns:
            True jeśli wczytano pomyślnie, False w przeciwnym wypadku
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Nie znaleziono pliku: {filepath}")
                return False
                
            # Wczytaj z pliku
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            # Zaktualizuj konfigurację, zachowując aktualne wartości dla nieistniejących kluczy
            for key, value in state.get("config", {}).items():
                self.config[key] = value
            
            # Zaktualizuj stan modeli
            for model_name, model_state in state.get("models_state", {}).items():
                if model_name in self.models_state:
                    self.models_state[model_name]["weight"] = model_state.get("weight", 1.0)
                    self.models_state[model_name]["enabled"] = model_state.get("enabled", True)
                    
                    # Zaktualizuj statystyki wydajności
                    perf = model_state.get("performance", {})
                    self.models_state[model_name]["performance"]["decisions_count"] = perf.get("decisions_count", 0)
                    self.models_state[model_name]["performance"]["correct_decisions"] = perf.get("correct_decisions", 0)
                    self.models_state[model_name]["performance"]["incorrect_decisions"] = perf.get("incorrect_decisions", 0)
                    self.models_state[model_name]["performance"]["profit_sum"] = perf.get("profit_sum", 0.0)
                    self.models_state[model_name]["performance"]["loss_sum"] = perf.get("loss_sum", 0.0)
            
            # Zaktualizuj stan rynku
            for key, value in state.get("market_state", {}).items():
                self.market_state[key] = value
                    
            logger.info(f"Wczytano stan Meta-Agenta z pliku: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania stanu Meta-Agenta: {str(e)}")
            logger.error(traceback.format_exc())
            return False
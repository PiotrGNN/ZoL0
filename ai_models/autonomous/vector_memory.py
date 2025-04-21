"""
Moduł pamięci wektorowej dla AI - przechowuje i wyszukuje konteksty decyzyjne.

Ten moduł implementuje pamięć wektorową, która pozwala na:
1. Przechowywanie kontekstów decyzyjnych z ich embeddingami
2. Wyszukiwanie podobnych kontekstów historycznych
3. Uczenie się na podstawie doświadczeń poprzez analizę podobnych kontekstów

Pamięć ta stanowi kluczowy element Meta-Agenta, umożliwiając mu podejmowanie
decyzji na podstawie doświadczeń i aktualnego kontekstu rynkowego.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
import random
import uuid
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import traceback

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class VectorMemory:
    """
    Pamięć wektorowa przechowująca i wyszukująca konteksty decyzyjne.
    Wykorzystuje embeddingi do reprezentowania kontekstów i szybkiego wyszukiwania podobnych przypadków.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicjalizuje pamięć wektorową.
        
        Args:
            config: Konfiguracja pamięci (rozmiar, wymiar embeddingów, itp.)
        """
        default_config = {
            "memory_size": 10000,  # Maksymalna liczba przechowywanych kontekstów
            "embedding_dim": 64,    # Wymiar wektorów embeddingów
            "storage_dir": os.path.join(os.path.dirname(__file__), "memory_storage"),
            "use_faiss": FAISS_AVAILABLE
        }
        
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            # Upewnij się, że klucz use_faiss zawsze istnieje w konfiguracji
            if "use_faiss" not in self.config:
                self.config["use_faiss"] = FAISS_AVAILABLE
        
        # Inicjalizacja pamięci
        self.memory = []  # Lista kontekstów
        self.embeddings = []  # Lista embeddingów
        self.metadata = []  # Lista metadanych (timestamp, wynik itp.)
        
        # Inicjalizacja indeksu FAISS jeśli dostępny
        self.index = None
        if self.config.get("use_faiss", False) and FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(self.config["embedding_dim"])
        
        # Utwórz katalog pamięci jeśli nie istnieje
        os.makedirs(self.config["storage_dir"], exist_ok=True)
        
        logger.info("Zainicjalizowano pamięć wektorową")
    
    def _generate_embedding(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Generuje embedding dla kontekstu decyzyjnego.
        
        Args:
            context: Kontekst decyzyjny (dane rynkowe, sygnały modeli itp.)
            
        Returns:
            Wektor embeddingu
        """
        # To jest prosta implementacja generowania embeddingów
        # W rzeczywistej implementacji warto użyć bardziej zaawansowanych technik
        # np. modelu neuronowego do generowania embeddingów
        
        # Inicjalizacja wektora embeddingu zerami
        embedding = np.zeros(self.config["embedding_dim"])
        
        # 1. Uwzględnij decyzje modeli
        model_decisions = context.get("model_decisions", {})
        for i, (model_name, decision) in enumerate(model_decisions.items()):
            # Kodowanie decyzji jako liczby
            decision_value = 0.0
            if isinstance(decision, (int, float)):
                decision_value = decision
            elif isinstance(decision, str):
                if decision in ["BUY", "LONG"]:
                    decision_value = 1.0
                elif decision in ["SELL", "SHORT"]:
                    decision_value = -1.0
                elif decision in ["HOLD", "WAIT"]:
                    decision_value = 0.0
                    
            # Dodaj wpływ decyzji do embeddingu
            idx = hash(model_name) % (self.config["embedding_dim"] // 2)
            embedding[idx] += decision_value
        
        # 2. Uwzględnij dane rynkowe
        market_data = context.get("market_data", {})
        market_features = [
            "close", "volume", "rsi", "macd", "sma_20", "ema_50", "bollinger_upper", "bollinger_lower"
        ]
        
        for feature in market_features:
            if feature in market_data:
                try:
                    value = market_data[feature]
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        # Jeśli to lista/tablica, weź ostatnią wartość
                        value = value[-1]
                        
                    if isinstance(value, (int, float)):
                        # Dodaj wpływ cechy do embeddingu
                        idx = hash(feature) % (self.config["embedding_dim"] // 2)
                        embedding[self.config["embedding_dim"] // 2 + idx % (self.config["embedding_dim"] // 2)] = value
                except:
                    pass
        
        # 3. Uwzględnij finalną decyzję
        final_decision = context.get("final_decision", "UNKNOWN")
        decision_encoding = {
            "BUY": 1.0, "LONG": 1.0,
            "SELL": -1.0, "SHORT": -1.0,
            "HOLD": 0.0, "WAIT": 0.0, 
            "UNKNOWN": 0.0
        }
        
        decision_value = decision_encoding.get(final_decision, 0.0)
        embedding[0] = decision_value  # Pierwsza pozycja to finalna decyzja
        
        # Normalizacja wektora
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def store(self, context: Dict[str, Any], result: Dict[str, Any] = None) -> bool:
        """
        Przechowuje kontekst decyzyjny w pamięci wraz z embeddingiem.
        
        Args:
            context: Kontekst decyzyjny
            result: Wynik decyzji (opcjonalnie, może być dodany później)
            
        Returns:
            True jeśli dodano pomyślnie, False w przeciwnym wypadku
        """
        try:
            # Wygeneruj embedding
            embedding = self._generate_embedding(context)
            
            # Przygotuj metadane
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "id": str(uuid.uuid4()),
                "result": result
            }
            
            # Dodaj do pamięci
            self.memory.append(context)
            self.embeddings.append(embedding)
            self.metadata.append(metadata)
            
            # Aktualizuj indeks FAISS jeśli używany
            if self.config["use_faiss"] and FAISS_AVAILABLE and self.index is not None:
                self.index.add(np.array([embedding], dtype=np.float32))
            
            # Usuń najstarsze konteksty jeśli przekroczono maksymalny rozmiar pamięci
            if len(self.memory) > self.config["memory_size"]:
                excess = len(self.memory) - self.config["memory_size"]
                self.memory = self.memory[excess:]
                self.embeddings = self.embeddings[excess:]
                self.metadata = self.metadata[excess:]
                
                # Przebuduj indeks FAISS jeśli używany
                if self.config["use_faiss"] and FAISS_AVAILABLE and self.index is not None:
                    self.index = faiss.IndexFlatL2(self.config["embedding_dim"])
                    self.index.add(np.array(self.embeddings, dtype=np.float32))
            
            logger.debug(f"Zapisano kontekst decyzyjny w pamięci, ID: {metadata['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania kontekstu w pamięci: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def update_result(self, context_id: str, result: Dict[str, Any]) -> bool:
        """
        Aktualizuje wynik dla istniejącego kontekstu.
        
        Args:
            context_id: ID kontekstu do aktualizacji
            result: Wynik do dodania
            
        Returns:
            True jeśli zaktualizowano pomyślnie, False w przeciwnym wypadku
        """
        try:
            # Znajdź kontekst po ID
            for i, meta in enumerate(self.metadata):
                if meta["id"] == context_id:
                    self.metadata[i]["result"] = result
                    logger.debug(f"Zaktualizowano wynik dla kontekstu ID: {context_id}")
                    return True
                    
            logger.warning(f"Nie znaleziono kontekstu o ID: {context_id}")
            return False
            
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji wyniku: {str(e)}")
            return False
    
    def search_similar(self, query_context: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """
        Wyszukuje k najbardziej podobnych kontekstów do podanego.
        
        Args:
            query_context: Kontekst do porównania
            k: Liczba zwracanych najbardziej podobnych kontekstów
            
        Returns:
            Lista podobnych kontekstów wraz z metadanymi i podobieństwem
        """
        try:
            if not self.embeddings:
                logger.warning("Pamięć wektorowa jest pusta")
                return []
                
            # Wygeneruj embedding dla zapytania
            query_embedding = self._generate_embedding(query_context)
            
            # Znajdź podobne konteksty
            if self.config["use_faiss"] and FAISS_AVAILABLE and self.index is not None and len(self.memory) > 0:
                # Wyszukiwanie z FAISS
                k = min(k, len(self.embeddings))
                distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
                distances = distances[0]
                indices = indices[0]
                
                # Konwertuj odległości na podobieństwo (1 - odległość znormalizowana)
                max_dist = np.max(distances) if distances.size > 0 else 1.0
                similarities = 1.0 - distances / max_dist if max_dist > 0 else np.ones_like(distances)
                
                results = []
                for i, idx in enumerate(indices):
                    if idx < len(self.memory):
                        results.append({
                            "context": self.memory[idx],
                            "metadata": self.metadata[idx],
                            "similarity": float(similarities[i])
                        })
            else:
                # Wyszukiwanie z cosine similarity
                similarities = []
                for emb in self.embeddings:
                    similarity = float(cosine_similarity([query_embedding], [emb])[0][0])
                    similarities.append(similarity)
                    
                # Znajdź k najbardziej podobnych kontekstów
                k = min(k, len(similarities))
                top_indices = np.argsort(similarities)[-k:][::-1]
                
                results = []
                for idx in top_indices:
                    results.append({
                        "context": self.memory[idx],
                        "metadata": self.metadata[idx],
                        "similarity": float(similarities[idx])
                    })
                    
            logger.debug(f"Znaleziono {len(results)} podobnych kontekstów")
            return results
            
        except Exception as e:
            logger.error(f"Błąd podczas wyszukiwania podobnych kontekstów: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def get_context_by_id(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Pobiera kontekst o podanym ID.
        
        Args:
            context_id: ID kontekstu
            
        Returns:
            Kontekst lub None jeśli nie znaleziono
        """
        try:
            for i, meta in enumerate(self.metadata):
                if meta["id"] == context_id:
                    return {
                        "context": self.memory[i],
                        "metadata": self.metadata[i]
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Błąd podczas pobierania kontekstu: {str(e)}")
            return None
    
    def save_state(self, filepath: str = None) -> bool:
        """
        Zapisuje stan pamięci wektorowej do pliku.
        
        Args:
            filepath: Ścieżka do pliku (opcjonalnie)
            
        Returns:
            True jeśli zapisano pomyślnie, False w przeciwnym wypadku
        """
        try:
            if filepath is None:
                # Generuj domyślną ścieżkę
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.config["storage_dir"], f"vector_memory_state_{timestamp}.pkl")
                
            # Przygotuj dane do zapisu
            state = {
                "memory": self.memory,
                "embeddings": self.embeddings,
                "metadata": self.metadata,
                "config": self.config,
                "timestamp": datetime.now().isoformat()
            }
            
            # Zapisz do pliku
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Zapisano stan pamięci wektorowej do pliku: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania stanu pamięci: {str(e)}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Wczytuje stan pamięci wektorowej z pliku.
        
        Args:
            filepath: Ścieżka do pliku
            
        Returns:
            True jeśli wczytano pomyślnie, False w przeciwnym wypadku
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Nie znaleziono pliku: {filepath}")
                return False
                
            # Wczytaj z pliku
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            # Przywróć stan
            self.memory = state["memory"]
            self.embeddings = state["embeddings"]
            self.metadata = state["metadata"]
            
            # Zachowaj aktualną konfigurację, ale zaktualizuj niektóre parametry
            self.config["memory_size"] = max(self.config["memory_size"], len(self.memory))
            
            # Odtwórz indeks FAISS jeśli używany
            if self.config["use_faiss"] and FAISS_AVAILABLE:
                self.index = faiss.IndexFlatL2(self.config["embedding_dim"])
                if self.embeddings:
                    self.index.add(np.array(self.embeddings, dtype=np.float32))
                    
            logger.info(f"Wczytano stan pamięci wektorowej z pliku: {filepath}, {len(self.memory)} kontekstów")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania stanu pamięci: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def clear(self) -> bool:
        """
        Czyści pamięć wektorową.
        
        Returns:
            True jeśli wyczyszczono pomyślnie, False w przeciwnym wypadku
        """
        try:
            self.memory = []
            self.embeddings = []
            self.metadata = []
            
            # Zresetuj indeks FAISS jeśli używany
            if self.config["use_faiss"] and FAISS_AVAILABLE:
                self.index = faiss.IndexFlatL2(self.config["embedding_dim"])
                
            logger.info("Wyczyszczono pamięć wektorową")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas czyszczenia pamięci: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Zwraca statystyki pamięci wektorowej.
        
        Returns:
            Słownik ze statystykami
        """
        try:
            # Zlicz decyzje
            decision_counts = {}
            for ctx in self.memory:
                decision = ctx.get("final_decision", "UNKNOWN")
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
                
            # Zlicz wyniki
            result_stats = {
                "with_result": 0,
                "without_result": 0,
                "positive_profit": 0,
                "negative_profit": 0,
                "avg_profit": 0.0
            }
            
            profits = []
            for meta in self.metadata:
                if meta.get("result"):
                    result_stats["with_result"] += 1
                    profit = meta["result"].get("profit_pct", 0)
                    if profit > 0:
                        result_stats["positive_profit"] += 1
                    elif profit < 0:
                        result_stats["negative_profit"] += 1
                    profits.append(profit)
                else:
                    result_stats["without_result"] += 1
            
            if profits:
                result_stats["avg_profit"] = sum(profits) / len(profits)
                
            # Przygotuj statystyki
            stats = {
                "total_contexts": len(self.memory),
                "memory_usage_pct": len(self.memory) / self.config["memory_size"] * 100 if self.config["memory_size"] > 0 else 0,
                "decision_distribution": decision_counts,
                "results": result_stats
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania statystyk pamięci: {str(e)}")
            return {"error": str(e)}
    
    def export_to_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Eksportuje zawartość pamięci wektorowej do DataFrame.
        
        Returns:
            DataFrame z danymi lub None w przypadku błędu
        """
        try:
            rows = []
            for i, (ctx, meta) in enumerate(zip(self.memory, self.metadata)):
                row = {
                    "id": meta.get("id", f"ctx_{i}"),
                    "timestamp": meta.get("timestamp", ""),
                    "decision": ctx.get("final_decision", "UNKNOWN")
                }
                
                # Dodaj wyniki jeśli są dostępne
                if meta.get("result"):
                    row["profit"] = meta["result"].get("profit_pct", 0)
                    row["max_drawdown"] = meta["result"].get("max_drawdown", 0)
                
                rows.append(row)
                
            df = pd.DataFrame(rows)
            return df
            
        except Exception as e:
            logger.error(f"Błąd podczas eksportu do DataFrame: {str(e)}")
            return None
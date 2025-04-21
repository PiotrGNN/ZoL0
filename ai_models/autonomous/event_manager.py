"""
Moduł zarządzania komunikacją między komponentami autonomicznego systemu AI.

Implementuje wzorzec publikuj-subskrybuj do efektywnej komunikacji między różnymi 
elementami systemu autonomicznego bez ścisłych zależności między nimi.
"""

import logging
import threading
import queue
import time
import traceback
from typing import Dict, List, Callable, Any, Set
from datetime import datetime

# Konfiguracja logowania
logger = logging.getLogger(__name__)

class EventManager:
    """
    Manager zdarzeń implementujący wzorzec publikuj-subskrybuj do komunikacji między
    komponentami autonomicznego systemu AI.
    """
    
    def __init__(self):
        """
        Inicjalizuje managera zdarzeń.
        """
        # Słownik subskrybentów: {nazwa_zdarzenia: [lista_subskrybentów]}
        self._subscribers: Dict[str, List[Callable]] = {}
        
        # Kolejka zdarzeń
        self._event_queue = queue.Queue()
        
        # Flaga aktywności
        self._is_running = False
        
        # Wątek przetwarzający zdarzenia
        self._processing_thread = None
        
        # Blokada dla bezpiecznego dostępu do słownika subskrybentów
        self._subscribers_lock = threading.Lock()
        
        # Zbiór aktywnych typów zdarzeń (do debugowania)
        self._active_event_types: Set[str] = set()
        
        logger.info("EventManager: Zainicjalizowano")
        
    def subscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Rejestruje funkcję callback jako subskrybenta danego typu zdarzenia.
        
        Args:
            event_type: Nazwa typu zdarzenia
            callback: Funkcja wywoływana gdy zdarzenie nastąpi
            
        Returns:
            True jeśli dodano pomyślnie, False w przeciwnym wypadku
        """
        try:
            with self._subscribers_lock:
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = []
                
                if callback not in self._subscribers[event_type]:
                    self._subscribers[event_type].append(callback)
                    self._active_event_types.add(event_type)
                    logger.debug(f"EventManager: Subskrybowano {event_type}")
                    return True
                else:
                    logger.warning(f"EventManager: Subskrybent już istnieje dla {event_type}")
                    return False
        except Exception as e:
            logger.error(f"EventManager: Błąd podczas subskrypcji {event_type}: {str(e)}")
            return False
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Usuwa funkcję callback z listy subskrybentów danego typu zdarzenia.
        
        Args:
            event_type: Nazwa typu zdarzenia
            callback: Funkcja do usunięcia
            
        Returns:
            True jeśli usunięto pomyślnie, False w przeciwnym wypadku
        """
        try:
            with self._subscribers_lock:
                if event_type in self._subscribers and callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
                    
                    # Jeśli nie ma więcej subskrybentów, usuń typ zdarzenia
                    if not self._subscribers[event_type]:
                        del self._subscribers[event_type]
                        self._active_event_types.discard(event_type)
                    
                    logger.debug(f"EventManager: Anulowano subskrypcję {event_type}")
                    return True
                else:
                    logger.warning(f"EventManager: Nie znaleziono subskrypcji dla {event_type}")
                    return False
        except Exception as e:
            logger.error(f"EventManager: Błąd podczas anulowania subskrypcji {event_type}: {str(e)}")
            return False
    
    def publish(self, event_type: str, data: Any = None) -> bool:
        """
        Publikuje zdarzenie do kolejki zdarzeń.
        
        Args:
            event_type: Nazwa typu zdarzenia
            data: Dane zdarzenia
            
        Returns:
            True jeśli dodano do kolejki, False w przeciwnym wypadku
        """
        try:
            event = {
                'type': event_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            self._event_queue.put(event)
            logger.debug(f"EventManager: Opublikowano zdarzenie {event_type}")
            return True
        except Exception as e:
            logger.error(f"EventManager: Błąd podczas publikowania zdarzenia {event_type}: {str(e)}")
            return False
    
    def publish_sync(self, event_type: str, data: Any = None) -> bool:
        """
        Publikuje zdarzenie i od razu je przetwarza (synchronicznie).
        
        Args:
            event_type: Nazwa typu zdarzenia
            data: Dane zdarzenia
            
        Returns:
            True jeśli przetworzono pomyślnie, False w przeciwnym wypadku
        """
        try:
            event = {
                'type': event_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            self._process_event(event)
            logger.debug(f"EventManager: Przetworzono synchronicznie zdarzenie {event_type}")
            return True
        except Exception as e:
            logger.error(f"EventManager: Błąd podczas synchronicznego przetwarzania zdarzenia {event_type}: {str(e)}")
            return False
    
    def _process_event(self, event: Dict[str, Any]) -> None:
        """
        Przetwarza pojedyncze zdarzenie.
        
        Args:
            event: Słownik reprezentujący zdarzenie
        """
        event_type = event['type']
        event_data = event['data']
        
        with self._subscribers_lock:
            if event_type in self._subscribers:
                subscribers = self._subscribers[event_type].copy()
            else:
                subscribers = []
        
        for callback in subscribers:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"EventManager: Błąd w callbacku dla {event_type}: {str(e)}")
                logger.error(traceback.format_exc())
    
    def _event_processing_loop(self) -> None:
        """
        Główna pętla przetwarzania zdarzeń, uruchamiana w osobnym wątku.
        """
        logger.info("EventManager: Uruchomiono pętlę przetwarzania zdarzeń")
        
        while self._is_running:
            try:
                # Pobierz zdarzenie z timeoutem, aby móc reagować na zatrzymanie
                try:
                    event = self._event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Przetwórz zdarzenie
                self._process_event(event)
                
                # Oznacz zadanie jako ukończone
                self._event_queue.task_done()
                
            except Exception as e:
                logger.error(f"EventManager: Błąd w pętli przetwarzania zdarzeń: {str(e)}")
                logger.error(traceback.format_exc())
                # Krótka pauza aby uniknąć zapętlenia w przypadku błędu
                time.sleep(0.1)
    
    def start(self) -> bool:
        """
        Uruchamia wątek przetwarzania zdarzeń.
        
        Returns:
            True jeśli uruchomiono pomyślnie, False w przeciwnym wypadku
        """
        if self._is_running:
            logger.warning("EventManager: Przetwarzanie zdarzeń już jest uruchomione")
            return False
        
        try:
            self._is_running = True
            self._processing_thread = threading.Thread(
                target=self._event_processing_loop,
                name="EventManagerThread"
            )
            self._processing_thread.daemon = True
            self._processing_thread.start()
            
            logger.info("EventManager: Uruchomiono przetwarzanie zdarzeń")
            return True
        except Exception as e:
            self._is_running = False
            logger.error(f"EventManager: Błąd podczas uruchamiania przetwarzania zdarzeń: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Zatrzymuje wątek przetwarzania zdarzeń.
        
        Returns:
            True jeśli zatrzymano pomyślnie, False w przeciwnym wypadku
        """
        if not self._is_running:
            logger.warning("EventManager: Przetwarzanie zdarzeń nie jest uruchomione")
            return False
        
        try:
            self._is_running = False
            
            # Dodaj puste zdarzenie, aby wyjść z bloku get()
            self._event_queue.put({
                'type': '_system_stop',
                'data': None,
                'timestamp': datetime.now().isoformat()
            })
            
            # Czekaj na zakończenie wątku, ale z timeoutem
            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=5.0)
                
                if self._processing_thread.is_alive():
                    logger.warning("EventManager: Nie udało się zakończyć wątku w wyznaczonym czasie")
            
            logger.info("EventManager: Zatrzymano przetwarzanie zdarzeń")
            return True
        except Exception as e:
            logger.error(f"EventManager: Błąd podczas zatrzymywania przetwarzania zdarzeń: {str(e)}")
            return False
    
    def is_running(self) -> bool:
        """
        Sprawdza, czy wątek przetwarzania zdarzeń jest aktywny.
        
        Returns:
            True jeśli aktywny, False w przeciwnym wypadku
        """
        return self._is_running
    
    def get_queue_size(self) -> int:
        """
        Zwraca aktualną liczbę zdarzeń w kolejce.
        
        Returns:
            Liczba zdarzeń w kolejce
        """
        return self._event_queue.qsize()
    
    def get_subscribers_count(self) -> Dict[str, int]:
        """
        Zwraca liczbę subskrybentów dla każdego typu zdarzenia.
        
        Returns:
            Słownik {typ_zdarzenia: liczba_subskrybentów}
        """
        with self._subscribers_lock:
            return {event_type: len(subscribers) for event_type, subscribers in self._subscribers.items()}
    
    def get_active_event_types(self) -> List[str]:
        """
        Zwraca listę aktywnych typów zdarzeń.
        
        Returns:
            Lista nazw typów zdarzeń
        """
        with self._subscribers_lock:
            return list(self._active_event_types)
    
    def clear_queue(self) -> int:
        """
        Czyści kolejkę zdarzeń.
        
        Returns:
            Liczba usuniętych zdarzeń
        """
        count = 0
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
                count += 1
            except queue.Empty:
                break
        
        logger.info(f"EventManager: Wyczyszczono kolejkę zdarzeń ({count} zdarzeń)")
        return count


# Singleton dla łatwego dostępu globalnego
_event_manager_instance = None

def get_event_manager():
    """
    Zwraca globalną instancję EventManager.
    
    Returns:
        Instancja EventManager
    """
    global _event_manager_instance
    if _event_manager_instance is None:
        _event_manager_instance = EventManager()
        
    return _event_manager_instance
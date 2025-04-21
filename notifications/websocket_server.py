"""
Moduł serwera WebSocket do obsługi powiadomień w czasie rzeczywistym.
Umożliwia przesyłanie powiadomień, alertów i aktualizacji danych do klientów websocket.
"""

import logging
import asyncio
import json
import websockets
from datetime import datetime

logger = logging.getLogger(__name__)

class NotificationServer:
    """Serwer WebSocket do obsługi powiadomień w czasie rzeczywistym"""
    
    def __init__(self):
        self.clients = set()
        self.host = '0.0.0.0'
        self.port = 6789
        self.server = None
        self._running = False
        self.message_queue = asyncio.Queue()
        
    async def start_server(self) -> None:
        """Uruchamia serwer WebSocket"""
        self._running = True
        try:
            self.server = await websockets.serve(self.handle_client, self.host, self.port)
            logging.info(f"Serwer WebSocket uruchomiony na {self.host}:{self.port}")
            await self.server.wait_closed()
        except Exception as e:
            logging.error(f"Błąd podczas uruchamiania serwera WebSocket: {e}")
            self._running = False
            raise

    async def handle_client(self, websocket, path):
        """Obsługuje połączenie z klientem"""
        try:
            self.clients.add(websocket)
            logging.info(f"Nowy klient połączony, liczba klientów: {len(self.clients)}")
            
            while True:
                try:
                    message = await websocket.recv()
                    await self.process_message(websocket, message)
                except websockets.ConnectionClosed:
                    break
                except Exception as e:
                    logging.error(f"Błąd podczas obsługi wiadomości: {e}")
                    break
                    
        finally:
            self.clients.remove(websocket)
            logging.info(f"Klient rozłączony, pozostało klientów: {len(self.clients)}")
            
    async def process_message(self, websocket, message):
        """Przetwarza otrzymaną wiadomość"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'subscribe':
                # Obsługa subskrypcji
                await self.handle_subscription(websocket, data)
            elif msg_type == 'unsubscribe':
                # Obsługa anulowania subskrypcji
                await self.handle_unsubscription(websocket, data)
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Nieznany typ wiadomości'
                }))
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Nieprawidłowy format JSON'
            }))
            
    async def broadcast(self, message):
        """Wysyła wiadomość do wszystkich podłączonych klientów"""
        if not self.clients:
            return
            
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except websockets.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logging.error(f"Błąd podczas wysyłania wiadomości: {e}")
                disconnected_clients.add(client)
                
        # Usuń rozłączonych klientów
        self.clients -= disconnected_clients
        
    def stop(self):
        """Zatrzymuje serwer WebSocket"""
        self._running = False
        if self.server:
            self.server.close()
            
    @property
    def is_running(self):
        """Zwraca status serwera"""
        return self._running
        
    @property
    def client_count(self):
        """Zwraca liczbę podłączonych klientów"""
        return len(self.clients)

def create_notification_server():
    """Tworzy i zwraca nową instancję serwera powiadomień"""
    return NotificationServer()


# Przykładowe użycie jako moduł autonomiczny
if __name__ == "__main__":
    # Konfiguracja logowania
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Utworzenie i uruchomienie serwera
    server = create_notification_server()
    
    # Ustawienie handlera zamknięcia
    loop = asyncio.get_event_loop()
    
    try:
        loop.run_until_complete(server.start_server())
    except KeyboardInterrupt:
        logging.info("Zatrzymywanie serwera WebSocket...")
        loop.run_until_complete(server.stop())
    finally:
        loop.close()
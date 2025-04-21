// notifications.js - Obsługa powiadomień w czasie rzeczywistym

/**
 * Klasa zarządzająca powiadomieniami w czasie rzeczywistym
 */
class NotificationManager {
    constructor(wsUrl = 'ws://localhost:6789') {
        this.wsUrl = wsUrl;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.notifications = [];
        this.subscriptions = new Set();
        this.messageHandlers = new Map();
        
        // Zarejestruj domyślne handlery
        this.registerDefaultHandlers();
        
        // Inicjalizacja połączenia
        this.connect();
    }

    connect() {
        try {
            this.ws = new WebSocket(this.wsUrl);
            this.setupWebSocketHandlers();
        } catch (error) {
            console.error('Błąd podczas łączenia z serwerem WebSocket:', error);
            this.scheduleReconnect();
        }
    }

    setupWebSocketHandlers() {
        this.ws.onopen = () => {
            console.log('Połączono z serwerem WebSocket');
            this.reconnectAttempts = 0;
            this.resubscribe();
        };

        this.ws.onclose = () => {
            console.log('Rozłączono z serwerem WebSocket');
            this.scheduleReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('Błąd WebSocket:', error);
        };

        this.ws.onmessage = (event) => {
            try {
                const notification = JSON.parse(event.data);
                this.handleNotification(notification);
            } catch (error) {
                console.error('Błąd podczas przetwarzania powiadomienia:', error);
            }
        };
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            console.log(`Próba ponownego połączenia za ${delay}ms...`);
            setTimeout(() => this.connect(), delay);
        } else {
            console.error('Przekroczono maksymalną liczbę prób połączenia');
            this.showNotification('error', 'Nie można połączyć się z serwerem powiadomień');
        }
    }

    handleNotification(notification) {
        switch (notification.type) {
            case 'system':
                this.handleSystemNotification(notification.data);
                break;
            case 'price':
                this.handlePriceUpdate(notification.data);
                break;
            case 'trade':
                this.handleTradeExecution(notification.data);
                break;
            default:
                console.warn('Nieznany typ powiadomienia:', notification.type);
        }
    }

    handleSystemNotification(data) {
        // Dodaj powiadomienie do listy
        this.notifications.unshift(data);
        // Ogranicz liczbę przechowywanych powiadomień
        if (this.notifications.length > 100) {
            this.notifications.pop();
        }

        // Aktualizuj interfejs
        this.updateNotificationsUI();

        // Pokaż toast z powiadomieniem
        this.showToast(data);
    }

    handlePriceUpdate(data) {
        // Aktualizuj cenę w interfejsie
        const priceElement = document.querySelector(`[data-symbol="${data.symbol}"] .price`);
        if (priceElement) {
            const oldPrice = parseFloat(priceElement.textContent);
            const newPrice = data.price;
            
            // Dodaj klasę wskazującą kierunek zmiany
            priceElement.classList.remove('price-up', 'price-down');
            if (newPrice > oldPrice) {
                priceElement.classList.add('price-up');
            } else if (newPrice < oldPrice) {
                priceElement.classList.add('price-down');
            }
            
            priceElement.textContent = newPrice.toFixed(2);
        }
    }

    handleTradeExecution(data) {
        // Dodaj transakcję do listy ostatnich transakcji
        const tradesContainer = document.getElementById('recent-trades');
        if (tradesContainer) {
            const tradeElement = document.createElement('div');
            tradeElement.className = `trade-item ${data.side.toLowerCase()}`;
            tradeElement.innerHTML = `
                <div class="trade-time">${new Date(data.timestamp).toLocaleTimeString()}</div>
                <div class="trade-symbol">${data.symbol}</div>
                <div class="trade-side">${data.side}</div>
                <div class="trade-price">${data.price}</div>
                <div class="trade-amount">${data.amount}</div>
            `;
            tradesContainer.insertBefore(tradeElement, tradesContainer.firstChild);

            // Ogranicz liczbę wyświetlanych transakcji
            if (tradesContainer.children.length > 50) {
                tradesContainer.lastChild.remove();
            }
        }

        // Pokaż powiadomienie o transakcji
        this.showToast({
            message: `${data.side} ${data.amount} ${data.symbol} @ ${data.price}`,
            level: 'info'
        });
    }

    updateNotificationsUI() {
        const container = document.getElementById('notifications-list');
        if (!container) return;

        // Aktualizuj liczbę nieprzeczytanych powiadomień
        const badge = document.getElementById('notifications-badge');
        if (badge) {
            const unread = this.notifications.filter(n => !n.read).length;
            badge.textContent = unread;
            badge.style.display = unread > 0 ? 'inline' : 'none';
        }

        // Aktualizuj listę powiadomień
        container.innerHTML = this.notifications.map(notification => `
            <div class="notification-item ${notification.level}" ${notification.read ? 'data-read' : ''}>
                <div class="notification-time">${new Date(notification.timestamp).toLocaleString()}</div>
                <div class="notification-message">${notification.message}</div>
            </div>
        `).join('');
    }

    showToast(data) {
        const toast = document.createElement('div');
        toast.className = `toast-notification ${data.level}`;
        toast.innerHTML = `
            <div class="toast-message">${data.message}</div>
            <div class="toast-time">${new Date().toLocaleTimeString()}</div>
        `;

        document.body.appendChild(toast);

        // Animacja wejścia
        setTimeout(() => toast.classList.add('show'), 100);

        // Automatyczne ukrywanie
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    subscribe(symbols) {
        if (!Array.isArray(symbols)) {
            symbols = [symbols];
        }
        
        symbols.forEach(symbol => this.subscriptions.add(symbol));
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                action: 'subscribe',
                symbols: symbols
            }));
        }
    }

    unsubscribe(symbols) {
        if (!Array.isArray(symbols)) {
            symbols = [symbols];
        }
        
        symbols.forEach(symbol => this.subscriptions.delete(symbol));
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                action: 'unsubscribe',
                symbols: symbols
            }));
        }
    }
    
    resubscribe() {
        if (this.subscriptions.size > 0) {
            this.subscribe(Array.from(this.subscriptions));
        }
    }
    
    setAlert(symbol, condition, value) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'set_alert',
                alert: {
                    symbol,
                    condition,
                    value
                }
            }));
        } else {
            console.error('Nie można ustawić alertu - brak połączenia z serwerem');
            this.showNotification('error', 'Nie można ustawić alertu - brak połączenia');
        }
    }
    
    updatePrice(symbol, price) {
        // Aktualizacja ceny w interfejsie
        const priceElement = document.querySelector(`[data-symbol="${symbol}"] .price`);
        if (priceElement) {
            const oldPrice = parseFloat(priceElement.textContent);
            priceElement.textContent = price.toFixed(2);
            
            // Dodaj klasę wskazującą kierunek zmiany
            priceElement.classList.remove('price-up', 'price-down');
            if (price > oldPrice) {
                priceElement.classList.add('price-up');
            } else if (price < oldPrice) {
                priceElement.classList.add('price-down');
            }
        }
    }
    
    updateTrade(tradeData) {
        // Aktualizacja informacji o transakcji w interfejsie
        const tradesContainer = document.getElementById('recent-trades');
        if (tradesContainer) {
            const tradeElement = document.createElement('div');
            tradeElement.classList.add('trade-item');
            tradeElement.innerHTML = `
                <div class="trade-symbol">${tradeData.symbol}</div>
                <div class="trade-side ${tradeData.side.toLowerCase()}">${tradeData.side}</div>
                <div class="trade-price">${tradeData.price}</div>
                <div class="trade-quantity">${tradeData.quantity}</div>
                <div class="trade-time">${new Date(tradeData.timestamp).toLocaleTimeString()}</div>
            `;
            
            tradesContainer.insertBefore(tradeElement, tradesContainer.firstChild);
            
            // Usuń najstarsze transakcje, jeśli jest ich za dużo
            while (tradesContainer.children.length > 50) {
                tradesContainer.removeChild(tradesContainer.lastChild);
            }
        }
    }
    
    showNotification(type, message) {
        // Inicjalizacja kontenera powiadomień, jeśli nie istnieje
        let notificationsContainer = document.getElementById('notifications-container');
        if (!notificationsContainer) {
            notificationsContainer = document.createElement('div');
            notificationsContainer.id = 'notifications-container';
            document.body.appendChild(notificationsContainer);
        }
        
        // Tworzenie elementu powiadomienia
        const notification = document.createElement('div');
        notification.classList.add('notification', `notification-${type}`);
        notification.innerHTML = `
            <div class="notification-content">${message}</div>
            <button class="notification-close">&times;</button>
        `;
        
        // Dodanie obsługi zamykania
        const closeButton = notification.querySelector('.notification-close');
        closeButton.onclick = () => {
            notification.remove();
        };
        
        // Dodanie powiadomienia do kontenera
        notificationsContainer.appendChild(notification);
        
        // Automatyczne usunięcie po 5 sekundach
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
}

// Inicjalizacja managera powiadomień
document.addEventListener('DOMContentLoaded', () => {
    window.notificationManager = new NotificationManager();
    
    // Przykład użycia:
    // window.notificationManager.subscribe(['BTCUSDT', 'ETHUSDT']);
    // window.notificationManager.setAlert('BTCUSDT', 'above', 50000);
});
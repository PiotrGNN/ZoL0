
// Dashboard JavaScript dla obsługi UI, WebSocketów i API
document.addEventListener('DOMContentLoaded', function() {
    // Inicjalizacja zmiennych globalnych
    let socket = null;
    let baseApiUrl = '';
    let chartInstances = {};
    let marketData = {
        btc: { price: 0, change: 0 },
        eth: { price: 0, change: 0 },
        bnb: { price: 0, change: 0 }
    };
    
    // Formatery liczbowe
    const currencyFormatter = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
    
    const cryptoFormatter = new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 8
    });
    
    const percentFormatter = new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
    
    // Inicjalizacja elementów UI
    function initializeUI() {
        // Ustawiamy datę ostatniej aktualizacji
        document.getElementById('lastUpdateTime').textContent = new Date().toLocaleTimeString();
        document.getElementById('aiUpdateTime').textContent = new Date().toLocaleTimeString();
        
        // Inicjalizacja wykresów
        initializeCharts();
        
        // Przypisanie obsługi zdarzeń
        document.getElementById('refreshBtn').addEventListener('click', refreshData);
        document.getElementById('connectBtn').addEventListener('click', toggleConnection);
        
        // Przypisanie obsługi zdarzeń dla przycisków interwału czasowego
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                // Usuń klasę aktywną ze wszystkich przycisków
                document.querySelectorAll('.timeframe-btn').forEach(b => {
                    b.classList.remove('bg-blue-600');
                    b.classList.add('bg-gray-700');
                });
                
                // Dodaj klasę aktywną do klikniętego przycisku
                this.classList.remove('bg-gray-700');
                this.classList.add('bg-blue-600');
                
                // Pobierz dane dla wybranego interwału
                const interval = this.dataset.interval;
                loadPriceChart(interval);
            });
        });
        
        // Przypisanie obsługi zdarzeń dla przycisków typu zlecenia
        document.querySelectorAll('.order-type-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                // Usuń klasę aktywną ze wszystkich przycisków
                document.querySelectorAll('.order-type-btn').forEach(b => {
                    b.classList.remove('bg-blue-600');
                    b.classList.add('bg-gray-700');
                });
                
                // Dodaj klasę aktywną do klikniętego przycisku
                this.classList.remove('bg-gray-700');
                this.classList.add('bg-blue-600');
            });
        });
    }
    
    // Inicjalizacja wykresów
    function initializeCharts() {
        // Inicjalizacja wykresu salda
        const balanceChartCtx = document.getElementById('balanceChart').getContext('2d');
        chartInstances.balance = new Chart(balanceChartCtx, {
            type: 'line',
            data: {
                labels: Array.from({ length: 30 }, (_, i) => i + 1),
                datasets: [{
                    label: 'Balance',
                    data: generateRandomData(30, 10000, 11000),
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: {
                        target: 'origin',
                        above: 'rgba(59, 130, 246, 0.1)'
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        display: false
                    }
                }
            }
        });
        
        // Inicjalizacja wykresu ceny
        const priceChartCtx = document.getElementById('priceChart').getContext('2d');
        chartInstances.price = new Chart(priceChartCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Price',
                    data: [],
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: {
                        target: 'origin',
                        above: 'rgba(59, 130, 246, 0.1)'
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return currencyFormatter.format(context.raw);
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false,
                            drawBorder: false
                        },
                        ticks: {
                            color: '#9ca3af',
                            maxRotation: 0
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(75, 85, 99, 0.2)'
                        },
                        ticks: {
                            color: '#9ca3af',
                            callback: function(value) {
                                return currencyFormatter.format(value);
                            }
                        }
                    }
                }
            }
        });
        
        // Załaduj dane początkowe
        loadPriceChart('1d');
    }
    
    // Funkcja do generowania losowych danych
    function generateRandomData(points, min, max) {
        return Array.from({ length: points }, () => Math.random() * (max - min) + min);
    }
    
    // Funkcja do ładowania wykresu ceny
    function loadPriceChart(interval) {
        // Symulacja danych dla wykresu
        const now = new Date();
        const labels = [];
        const data = [];
        
        let timeIncrement;
        let points;
        
        switch (interval) {
            case '1h':
                timeIncrement = 5 * 60 * 1000; // 5 minut
                points = 12;
                break;
            case '1w':
                timeIncrement = 24 * 60 * 60 * 1000; // 1 dzień
                points = 7;
                break;
            case '1d':
            default:
                timeIncrement = 60 * 60 * 1000; // 1 godzina
                points = 24;
                break;
        }
        
        let currentTime = now.getTime() - (points * timeIncrement);
        let baseValue = 42500;
        
        for (let i = 0; i < points; i++) {
            const date = new Date(currentTime);
            let label;
            
            switch (interval) {
                case '1h':
                    label = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    break;
                case '1w':
                    label = date.toLocaleDateString([], { weekday: 'short' });
                    break;
                case '1d':
                default:
                    label = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    break;
            }
            
            labels.push(label);
            
            // Generuj losową zmianę ceny (-1% do +1%)
            const change = baseValue * (Math.random() * 0.02 - 0.01);
            baseValue += change;
            data.push(baseValue);
            
            currentTime += timeIncrement;
        }
        
        // Aktualizuj wykres
        chartInstances.price.data.labels = labels;
        chartInstances.price.data.datasets[0].data = data;
        chartInstances.price.update();
    }
    
    // Odświeżanie danych dashboardu
    function refreshData() {
        // Symulacja odświeżania danych
        document.getElementById('lastUpdateTime').textContent = new Date().toLocaleTimeString();
        
        // Pobierz saldo
        fetchBalance();
        
        // Pobierz ticker dla kryptowalut
        fetchTickers();
        
        // Pobierz portfolio
        fetchPortfolio();
        
        // Pobierz transakcje
        fetchTransactions();
    }
    
    // Połączenie WebSocket
    function toggleConnection() {
        const connectBtn = document.getElementById('connectBtn');
        
        if (socket && socket.readyState === WebSocket.OPEN) {
            // Zamknij połączenie
            socket.close();
            connectBtn.innerHTML = '<i class="fas fa-plug mr-2"></i> Connect';
            connectBtn.classList.remove('bg-red-600', 'hover:bg-red-500');
            connectBtn.classList.add('bg-blue-600', 'hover:bg-blue-500');
        } else {
            // Inicjalizuj WebSocket
            initWebSocket();
            connectBtn.innerHTML = '<i class="fas fa-times mr-2"></i> Disconnect';
            connectBtn.classList.remove('bg-blue-600', 'hover:bg-blue-500');
            connectBtn.classList.add('bg-red-600', 'hover:bg-red-500');
        }
    }
    
    // Inicjalizacja WebSocket
    function initWebSocket() {
        // Zamknij istniejące połączenie, jeśli istnieje
        if (socket) {
            socket.close();
        }
        
        // Utwórz nowe połączenie WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        socket = new WebSocket(wsUrl);
        
        socket.onopen = function() {
            console.log('WebSocket połączony');
        };
        
        socket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (e) {
                console.error('Błąd podczas przetwarzania wiadomości WebSocket:', e);
            }
        };
        
        socket.onclose = function(event) {
            console.log('WebSocket rozłączony', event.code, event.reason);
            
            // Aktualizuj przycisk, jeśli połączenie zostało zamknięte
            const connectBtn = document.getElementById('connectBtn');
            connectBtn.innerHTML = '<i class="fas fa-plug mr-2"></i> Connect';
            connectBtn.classList.remove('bg-red-600', 'hover:bg-red-500');
            connectBtn.classList.add('bg-blue-600', 'hover:bg-blue-500');
        };
        
        socket.onerror = function(error) {
            console.error('Błąd WebSocket:', error);
        };
    }
    
    // Obsługa wiadomości WebSocket
    function handleWebSocketMessage(data) {
        if (data.type === 'market_update') {
            updateMarketData(data.data);
        } else if (data.type === 'balance_update') {
            updateBalanceData(data.data);
        } else if (data.type === 'order_update') {
            updateOrderData(data.data);
        }
    }
    
    // Aktualizacja danych rynkowych
    function updateMarketData(data) {
        // Aktualizuj dane dla BTC
        if (data.btc) {
            document.getElementById('btcPrice').textContent = currencyFormatter.format(data.btc.price);
            const btcChangeElem = document.getElementById('btcChange');
            btcChangeElem.textContent = `${data.btc.change >= 0 ? '+' : ''}${(data.btc.change * 100).toFixed(2)}%`;
            btcChangeElem.className = data.btc.change >= 0 ? 'text-xs price-up' : 'text-xs price-down';
            
            // Zapisz dane dla obliczeń
            marketData.btc.price = data.btc.price;
            marketData.btc.change = data.btc.change;
        }
        
        // Aktualizuj dane dla ETH
        if (data.eth) {
            document.getElementById('ethPrice').textContent = currencyFormatter.format(data.eth.price);
            const ethChangeElem = document.getElementById('ethChange');
            ethChangeElem.textContent = `${data.eth.change >= 0 ? '+' : ''}${(data.eth.change * 100).toFixed(2)}%`;
            ethChangeElem.className = data.eth.change >= 0 ? 'text-xs price-up' : 'text-xs price-down';
            
            // Zapisz dane dla obliczeń
            marketData.eth.price = data.eth.price;
            marketData.eth.change = data.eth.change;
        }
        
        // Aktualizuj dane dla BNB
        if (data.bnb) {
            document.getElementById('bnbPrice').textContent = currencyFormatter.format(data.bnb.price);
            const bnbChangeElem = document.getElementById('bnbChange');
            bnbChangeElem.textContent = `${data.bnb.change >= 0 ? '+' : ''}${(data.bnb.change * 100).toFixed(2)}%`;
            bnbChangeElem.className = data.bnb.change >= 0 ? 'text-xs price-up' : 'text-xs price-down';
            
            // Zapisz dane dla obliczeń
            marketData.bnb.price = data.bnb.price;
            marketData.bnb.change = data.bnb.change;
        }
        
        // Aktualizuj czas ostatniej aktualizacji
        document.getElementById('lastUpdateTime').textContent = new Date().toLocaleTimeString();
    }
    
    // Aktualizacja danych salda
    function updateBalanceData(data) {
        document.getElementById('totalBalance').textContent = currencyFormatter.format(data.total);
        document.getElementById('balanceChange').textContent = `${(data.change * 100).toFixed(2)}%`;
        
        // Aktualizuj wykres salda z nowymi danymi
        if (data.history && data.history.length > 0) {
            chartInstances.balance.data.datasets[0].data = data.history;
            chartInstances.balance.update();
        }
    }
    
    // Aktualizacja danych zleceń
    function updateOrderData(data) {
        // Aktualizuj listę zleceń
        const openOrdersTable = document.getElementById('openOrdersTable');
        openOrdersTable.innerHTML = '';
        
        data.orders.forEach(order => {
            const row = document.createElement('tr');
            row.className = 'text-sm border-b border-gray-700';
            
            row.innerHTML = `
                <td class="py-3">${order.symbol}</td>
                <td class="py-3 ${order.side === 'Buy' ? 'text-green-500' : 'text-red-500'}">${order.side}</td>
                <td class="py-3">${order.price.toFixed(2)}</td>
                <td class="py-3">${order.amount.toFixed(8)}</td>
                <td class="py-3">
                    <button class="text-red-500 hover:text-red-400" data-order-id="${order.id}">
                        <i class="fas fa-times"></i>
                    </button>
                </td>
            `;
            
            openOrdersTable.appendChild(row);
        });
        
        // Dodaj obsługę zdarzeń dla przycisków anulowania
        openOrdersTable.querySelectorAll('button[data-order-id]').forEach(btn => {
            btn.addEventListener('click', function() {
                const orderId = this.dataset.orderId;
                cancelOrder(orderId);
            });
        });
    }
    
    // Pobieranie danych z API
    function fetchBalance() {
        try {
            // Symulacja pobierania danych z API
            const balanceData = {
                total: 10250.75,
                change: 0.0325,
                history: generateRandomData(30, 10000, 10500)
            };
            
            // Aktualizuj UI
            document.getElementById('totalBalance').textContent = currencyFormatter.format(balanceData.total);
            document.getElementById('balanceChange').textContent = `${(balanceData.change * 100).toFixed(2)}%`;
            
            // Aktualizuj wykres salda
            chartInstances.balance.data.datasets[0].data = balanceData.history;
            chartInstances.balance.update();
            
            // Aktualizuj dane PnL
            document.getElementById('totalPnL').textContent = currencyFormatter.format(325.50);
            document.getElementById('dailyPnL').textContent = '+3.28%';
            document.getElementById('change24h').textContent = '+$25.50';
            document.getElementById('winRate').textContent = '68%';
            document.getElementById('avgRoi').textContent = '2.5%';
            
            // Aktualizuj metryki wydajności
            document.getElementById('sharpeRatio').textContent = '1.42';
            document.getElementById('maxDrawdown').textContent = '8.2%';
            document.getElementById('totalTrades').textContent = '48';
            document.getElementById('successRate').textContent = '70%';
        } catch (error) {
            console.error('Błąd podczas pobierania danych salda:', error);
        }
    }
    
    function fetchTickers() {
        try {
            // Symulacja pobierania danych z API
            const tickersData = {
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "last_price": 45678.90,
                    "bid": 45670.50,
                    "ask": 45680.30,
                    "volume_24h": 12345.67,
                    "change": 0.0245
                },
                "ETHUSDT": {
                    "symbol": "ETHUSDT",
                    "last_price": 3456.78,
                    "bid": 3455.50,
                    "ask": 3457.30,
                    "volume_24h": 67890.12,
                    "change": 0.0312
                },
                "BNBUSDT": {
                    "symbol": "BNBUSDT",
                    "last_price": 456.78,
                    "bid": 456.50,
                    "ask": 457.30,
                    "volume_24h": 34567.89,
                    "change": -0.0118
                }
            };
            
            // Aktualizuj dane rynkowe
            updateMarketData({
                btc: {
                    price: tickersData.BTCUSDT.last_price,
                    change: tickersData.BTCUSDT.change
                },
                eth: {
                    price: tickersData.ETHUSDT.last_price,
                    change: tickersData.ETHUSDT.change
                },
                bnb: {
                    price: tickersData.BNBUSDT.last_price,
                    change: tickersData.BNBUSDT.change
                }
            });
        } catch (error) {
            console.error('Błąd podczas pobierania danych tickerów:', error);
        }
    }
    
    function fetchPortfolio() {
        try {
            // Symulacja pobierania danych z API
            const portfolioData = [
                {
                    "asset": "BTC",
                    "balance": 0.22,
                    "price": 45678.90,
                    "value": 10049.36,
                    "change": 0.0245
                },
                {
                    "asset": "ETH",
                    "balance": 3.5,
                    "price": 3456.78,
                    "value": 12098.73,
                    "change": 0.0312
                },
                {
                    "asset": "BNB",
                    "balance": 12.8,
                    "price": 456.78,
                    "value": 5846.78,
                    "change": -0.0118
                },
                {
                    "asset": "USDT",
                    "balance": 15250.45,
                    "price": 1,
                    "value": 15250.45,
                    "change": 0
                }
            ];
            
            // Aktualizuj dane portfolio
            const portfolioTable = document.getElementById('portfolioTable');
            portfolioTable.innerHTML = '';
            
            portfolioData.forEach(asset => {
                const row = document.createElement('tr');
                row.className = 'text-sm border-b border-gray-700';
                
                row.innerHTML = `
                    <td class="py-3">
                        <div class="flex items-center">
                            <div class="w-6 h-6 rounded-full mr-2 flex items-center justify-center ${getAssetColor(asset.asset)}">
                                <span class="text-xs font-bold">${asset.asset.charAt(0)}</span>
                            </div>
                            <span>${asset.asset}</span>
                        </div>
                    </td>
                    <td class="py-3">${cryptoFormatter.format(asset.balance)}</td>
                    <td class="py-3">${currencyFormatter.format(asset.price)}</td>
                    <td class="py-3">${currencyFormatter.format(asset.value)}</td>
                    <td class="py-3 ${asset.change >= 0 ? 'text-green-500' : 'text-red-500'}">
                        ${asset.change >= 0 ? '+' : ''}${(asset.change * 100).toFixed(2)}%
                    </td>
                `;
                
                portfolioTable.appendChild(row);
            });
        } catch (error) {
            console.error('Błąd podczas pobierania danych portfolio:', error);
        }
    }
    
    function fetchTransactions() {
        try {
            // Symulacja pobierania danych z API
            const transactionsData = [
                {
                    "type": "Buy",
                    "asset": "BTC",
                    "amount": 0.02,
                    "price": 45680.30,
                    "value": 913.61,
                    "time": "2025-04-16 13:45:22"
                },
                {
                    "type": "Sell",
                    "asset": "ETH",
                    "amount": 1.5,
                    "price": 3455.50,
                    "value": 5183.25,
                    "time": "2025-04-16 12:30:15"
                },
                {
                    "type": "Buy",
                    "asset": "BNB",
                    "amount": 3.8,
                    "price": 456.50,
                    "value": 1734.70,
                    "time": "2025-04-16 10:15:40"
                }
            ];
            
            // Aktualizuj dane transakcji
            const recentTransactions = document.getElementById('recentTransactions');
            recentTransactions.innerHTML = '';
            
            transactionsData.forEach(tx => {
                const div = document.createElement('div');
                div.className = 'flex justify-between items-center border-b border-gray-700 pb-3';
                
                div.innerHTML = `
                    <div class="flex items-center">
                        <div class="w-8 h-8 rounded-full ${tx.type === 'Buy' ? 'bg-green-500' : 'bg-red-500'} bg-opacity-20 flex items-center justify-center mr-3">
                            <i class="fas ${tx.type === 'Buy' ? 'fa-arrow-down text-green-500' : 'fa-arrow-up text-red-500'}"></i>
                        </div>
                        <div>
                            <p class="font-medium">${tx.type} ${tx.asset}</p>
                            <p class="text-xs text-gray-400">${tx.time}</p>
                        </div>
                    </div>
                    <div class="text-right">
                        <p class="font-medium">${cryptoFormatter.format(tx.amount)} ${tx.asset}</p>
                        <p class="text-xs text-gray-400">${currencyFormatter.format(tx.value)}</p>
                    </div>
                `;
                
                recentTransactions.appendChild(div);
            });
        } catch (error) {
            console.error('Błąd podczas pobierania danych transakcji:', error);
        }
    }
    
    // Anulowanie zlecenia
    function cancelOrder(orderId) {
        console.log(`Anulowanie zlecenia: ${orderId}`);
        // W rzeczywistej implementacji, tutaj byłoby wywołanie API do anulowania zlecenia
        
        // Symulacja usunięcia zlecenia z interfejsu
        const orderElement = document.querySelector(`button[data-order-id="${orderId}"]`).closest('tr');
        if (orderElement) {
            orderElement.remove();
        }
    }
    
    // Pomocnicza funkcja do wybierania koloru tła dla aktywów
    function getAssetColor(asset) {
        switch (asset) {
            case 'BTC':
                return 'bg-yellow-500';
            case 'ETH':
                return 'bg-indigo-500';
            case 'BNB':
                return 'bg-yellow-600';
            case 'USDT':
                return 'bg-green-500';
            default:
                return 'bg-blue-500';
        }
    }
    
    // Inicjalizacja
    initializeUI();
    fetchBalance();
    fetchTickers();
    fetchPortfolio();
    fetchTransactions();
    
    // Symulacja aktualizacji danych co 10 sekund
    setInterval(refreshData, 10000);
});

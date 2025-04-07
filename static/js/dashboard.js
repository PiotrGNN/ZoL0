/**
 * Dashboard interaktywny dla Trading Bota
 * Skrypt odpowiedzialny za wykresy, aktualizacje danych i interakcje.
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard załadowany');

    try {
        // Inicjalizacja wykresów
        initializeCharts();

        // Ustawienie interwału do pobierania danych (co 30 sekund)
        window.dashboardInterval = setInterval(updateDashboardData, 30000);

        // Obsługa zdarzeń
        setupEventListeners();
    } catch (error) {
        console.error("Błąd podczas inicjalizacji dashboardu:", error);
        showNotification("Wystąpił błąd podczas ładowania dashboardu. Odśwież stronę.", "error");
    }
});

/**
 * Inicjalizuje wykresy na dashboardzie
 */
function initializeCharts() {
    // Wykres aktywności systemu
    initializeActivityChart();

    // Jeśli są inne wykresy, tutaj można dodać ich inicjalizację
}

/**
 * Inicjalizuje wykres aktywności systemu
 */
function initializeActivityChart() {
    const ctx = document.getElementById('activityChart').getContext('2d');

    // Inicjalizacja pustego wykresu
    window.activityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Pobierz dane do wykresu
    fetchChartData();
}

/**
 * Aktualizuje dane dashboardu
 */
function updateDashboardData() {
    console.log('Aktualizacja danych dashboardu...');

    // Dodajemy prosty debounce, aby nie wysyłać zbyt wielu żądań
    clearTimeout(window.updateTimeout);
    window.updateTimeout = setTimeout(() => {
        // Aktualizacja wykresu aktywności
        fetchChartData();

        // Symulacja aktualizacji statusów komponentów
        updateComponentStatuses();
    }, 300);
}



/**
 * Aktualizuje wykres aktywności
 */
function updateActivityChart(data) {
    try {
        if (window.activityChart) {
            window.activityChart.data.labels = data.labels;
            window.activityChart.data.datasets = data.datasets;
            window.activityChart.update();
        } else {
            console.error("Wykres aktywności nie jest zainicjalizowany");
            initializeActivityChart(); // Próba ponownej inicjalizacji
        }
    } catch (error) {
        console.error("Błąd podczas aktualizacji wykresu:", error);
        // Próba naprawy wykresu
        const ctx = document.getElementById('activityChart');
        if (ctx) {
            ctx.getContext('2d').clearRect(0, 0, ctx.width, ctx.height);
            initializeActivityChart();
        }
    }
}

/**
 * Symulacja aktualizacji statusów komponentów
 */
function updateComponentStatuses() {
    // W rzeczywistej aplikacji pobralibyśmy te dane z API
    console.log('Aktualizacja statusów komponentów...');
    // Implementacja w pełnej wersji
}

/**
 * Konfiguruje obsługę zdarzeń interaktywnych
 */
function setupEventListeners() {
    // Przykład obsługi przycisku symulacji
    const simulationButton = document.getElementById('startSimulationBtn');
    if (simulationButton) {
        simulationButton.addEventListener('click', function() {
            fetch('/start-simulation')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        showNotification('Symulacja uruchomiona', 'success');
                    } else {
                        showNotification('Błąd uruchamiania symulacji', 'error');
                    }
                })
                .catch(error => {
                    console.error('Błąd:', error);
                    showNotification('Błąd połączenia z serwerem', 'error');
                });
        });
    }

    // Obsługa przycisku generowania raportu
    const reportButton = document.getElementById('downloadReportBtn');
    if (reportButton) {
        reportButton.addEventListener('click', function() {
            fetch('/download-report')
                .then(response => response.json())
                .then(data => {
                    showNotification(data.message, 'success');
                })
                .catch(error => {
                    console.error('Błąd:', error);
                    showNotification('Błąd generowania raportu', 'error');
                });
        });
    }

    // Obsługa przycisku odświeżania danych
    const refreshButton = document.getElementById('refreshDataBtn');
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            showNotification('Odświeżanie danych...', 'info');
            fetchChartData();
            updateComponentStatuses();
        });
    }
}

/**
 * Wyświetla powiadomienie na dashboardzie
 */
function showNotification(message, type = 'info') {
    // Implementacja prostego systemu powiadomień
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    const container = document.createElement('div');
    container.className = 'notification-container';
    container.appendChild(notification);

    document.body.appendChild(container);

    // Usuń powiadomienie po 3 sekundach
    setTimeout(() => {
        container.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(container);
        }, 300);
    }, 3000);
}

// Zmienne do obsługi błędów
let chartErrorCount = 0;
let statusErrorCount = 0;
const MAX_ERRORS = 3;
const RECONNECT_INTERVAL = 5000; // 5 sekund
const NORMAL_INTERVAL = 30000; // 30 sekund

// Funkcja do obsługi błędów i ponownych prób
function handleChartError() {
    chartErrorCount++;
    console.log(`Błąd wykresu #${chartErrorCount}. Próba ponownego połączenia...`);

    if (chartErrorCount <= MAX_ERRORS) {
        setTimeout(fetchChartData, RECONNECT_INTERVAL);
    } else {
        console.log("Przekroczono maksymalną liczbę prób. Powrót do normalnego interwału.");
        chartErrorCount = 0;
    }
}

function handleStatusError() {
    statusErrorCount++;
    console.log(`Błąd statusu #${statusErrorCount}. Próba ponownego połączenia...`);

    if (statusErrorCount <= MAX_ERRORS) {
        setTimeout(updateSystemStatus, RECONNECT_INTERVAL);
    } else {
        console.log("Przekroczono maksymalną liczbę prób. Powrót do normalnego interwału.");
        statusErrorCount = 0;
    }
}

// Funkcja do pobierania danych wykresu z API
function fetchChartData() {
    // Usuń istniejący komunikat o błędzie i przycisk retry, jeśli istnieją
    const existingError = document.querySelector('.chart-container .error-message');
    if (existingError) {
        existingError.remove();
    }

    const existingRetryBtn = document.querySelector('.chart-container .retry-button');
    if (existingRetryBtn) {
        existingRetryBtn.remove();
    }

    // Pokaż wykres (mógł być ukryty przez poprzedni błąd)
    const chartCanvas = document.getElementById('activityChart');
    if (chartCanvas) {
        chartCanvas.style.display = 'block';
    }

    // Dodaj wskaźnik ładowania
    const chartContainer = document.querySelector('.chart-container');
    if (chartContainer) {
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'loading-indicator';
        loadingIndicator.textContent = 'Ładowanie danych...';
        chartContainer.appendChild(loadingIndicator);
    }

    fetch('/api/chart-data')
        .then(response => {
            // Usuń wskaźnik ładowania
            const loadingIndicator = document.querySelector('.loading-indicator');
            if (loadingIndicator) {
                loadingIndicator.remove();
            }

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            chartErrorCount = 0; // Resetuj licznik błędów
            return response.json();
        })
        .then(data => {
            if (data && data.labels && data.datasets) {
                updateActivityChart(data);
            } else {
                throw new Error('Nieprawidłowy format danych');
            }
        })
        .catch(error => {
            console.error("Błąd podczas pobierania danych wykresu:", error);

            // Usuń wskaźnik ładowania jeśli wciąż istnieje
            const loadingIndicator = document.querySelector('.loading-indicator');
            if (loadingIndicator) {
                loadingIndicator.remove();
            }

            // Wyświetl komunikat o błędzie na stronie
            if (chartCanvas) {
                chartCanvas.style.display = 'none';
            }

            if (chartContainer) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = 'Nie udało się załadować danych wykresu. Spróbuj odświeżyć stronę.';
                chartContainer.appendChild(errorDiv);

                // Dodajmy też przycisk do ręcznego odświeżenia
                const retryButton = document.createElement('button');
                retryButton.className = 'btn btn-primary retry-button';
                retryButton.textContent = 'Odśwież dane';
                retryButton.onclick = fetchChartData;
                chartContainer.appendChild(retryButton);
            }
            handleChartError();
        });
}

// Placeholder function -  needs implementation based on actual system status updates.
function updateSystemStatus() {
    console.log('Updating system status...');
    // Add your system status update logic here.  For example, fetching from an API.
    statusErrorCount = 0; //Reset error count on successful completion
}

// Aktualizacja danych co 30 sekund
setInterval(fetchChartData, NORMAL_INTERVAL);
setInterval(updateSystemStatus, NORMAL_INTERVAL);
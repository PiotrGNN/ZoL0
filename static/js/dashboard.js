/**
 * Dashboard interaktywny dla Trading Bota
 * Skrypt odpowiedzialny za wykresy, aktualizacje danych i interakcje.
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard załadowany');

    // Inicjalizacja wykresów
    initializeCharts();

    // Ustawienie interwału do pobierania danych
    setInterval(updateDashboardData, 30000); // Co 30 sekund

    // Obsługa zdarzeń
    setupEventListeners();
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

    // Aktualizacja wykresu aktywności
    fetchChartData(); // Use the new function to fetch and handle errors

    // Symulacja aktualizacji statusów komponentów
    updateComponentStatuses();
}



/**
 * Aktualizuje wykres aktywności
 */
function updateActivityChart(data) { // Added data parameter
    if (window.activityChart) {
        window.activityChart.data.labels = data.labels;
        window.activityChart.data.datasets = data.datasets;
        window.activityChart.update();
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

// Funkcja do pobierania danych wykresu z API
function fetchChartData() {
    // Usuń istniejący komunikat o błędzie, jeśli istnieje
    const existingError = document.querySelector('.chart-container .error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Pokaż wykres (mógł być ukryty przez poprzedni błąd)
    const chartCanvas = document.getElementById('activityChart');
    if (chartCanvas) {
        chartCanvas.style.display = 'block';
    }
    
    fetch('/api/chart-data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
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
            
            // Wyświetl komunikat o błędzie na stronie
            if (chartCanvas) {
                chartCanvas.style.display = 'none';
            }
            
            const chartContainer = document.querySelector('.chart-container');
            if (chartContainer) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = 'Nie udało się załadować danych wykresu. Spróbuj odświeżyć stronę.';
                chartContainer.appendChild(errorDiv);
            }
            
            // Dodajmy też przycisk do ręcznego odświeżenia
            const retryButton = document.createElement('button');
            retryButton.className = 'btn btn-primary';
            retryButton.textContent = 'Odśwież dane';
            retryButton.onclick = fetchChartData;
            
            if (chartContainer) {
                chartContainer.appendChild(retryButton);
            }
        });
}
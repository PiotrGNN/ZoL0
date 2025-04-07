
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
    
    // Pobierz dane do wykresu
    fetch('/api/chart-data')
        .then(response => response.json())
        .then(data => {
            window.activityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: data.datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            grid: {
                                display: false
                            }
                        },
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'bottom'
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych wykresu:', error);
        });
}

/**
 * Aktualizuje dane dashboardu
 */
function updateDashboardData() {
    console.log('Aktualizacja danych dashboardu...');
    
    // Aktualizacja wykresu aktywności
    updateActivityChart();
    
    // Symulacja aktualizacji statusów komponentów
    updateComponentStatuses();
}

/**
 * Aktualizuje wykres aktywności
 */
function updateActivityChart() {
    fetch('/api/chart-data')
        .then(response => response.json())
        .then(data => {
            if (window.activityChart) {
                window.activityChart.data.labels = data.labels;
                window.activityChart.data.datasets = data.datasets;
                window.activityChart.update();
            }
        })
        .catch(error => {
            console.error('Błąd podczas aktualizacji danych wykresu:', error);
        });
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

"""
Performance monitoring system with advanced metrics and analysis.
"""

import time
import psutil
import threading
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import sqlite3
import os
from dataclasses import dataclass
from statistics import mean, median, stdev

# Import system logger
from data.logging.system_logger import get_logger
logger = get_logger()

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    cpu_percent: float
    memory_percent: float
    io_counters: Dict[str, int]
    network_counters: Dict[str, int]
    response_times: Dict[str, float]
    error_rates: Dict[str, float]
    timestamp: str

class PerformanceMonitor:
    """Advanced performance monitoring system."""
    
    def __init__(self, db_path: str = "logs/performance.db"):
        self.db_path = db_path
        self._ensure_db_exists()
        
        # Initialize metrics storage
        self.metrics_history = deque(maxlen=1000)
        self.component_response_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        
        # Monitoring settings
        self.monitoring_interval = 60  # seconds
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time_ms': 1000,
            'error_rate': 0.1
        }
        
        # Start monitoring thread
        self._stop_monitoring = threading.Event()
        self._monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.log_info("Performance monitor initialized")

    def _ensure_db_exists(self) -> None:
        """Ensure performance database exists and has correct schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create metrics table
        c.execute('''CREATE TABLE IF NOT EXISTS performance_metrics
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp DATETIME,
                     cpu_percent REAL,
                     memory_percent REAL,
                     io_counters TEXT,
                     network_counters TEXT,
                     response_times TEXT,
                     error_rates TEXT)''')
        
        # Create alerts table
        c.execute('''CREATE TABLE IF NOT EXISTS performance_alerts
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp DATETIME,
                     alert_type TEXT,
                     message TEXT,
                     metrics TEXT)''')
        
        conn.commit()
        conn.close()

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                metrics = self._collect_metrics()
                self._store_metrics(metrics)
                self._analyze_metrics(metrics)
            except Exception as e:
                logger.log_error(f"Error in monitoring loop: {e}")
            
            self._stop_monitoring.wait(self.monitoring_interval)

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            process = psutil.Process()
            
            # Basic system metrics
            cpu_percent = process.cpu_percent()
            memory_percent = process.memory_percent()
            
            # IO metrics
            io_counters = process.io_counters()._asdict()
            
            # Network metrics
            network = psutil.net_io_counters()._asdict()
            
            # Component response times
            response_times = {
                component: mean(times) if times else 0
                for component, times in self.component_response_times.items()
            }
            
            # Error rates
            total_requests = sum(len(times) for times in self.component_response_times.values())
            error_rates = {
                component: count / total_requests if total_requests > 0 else 0
                for component, count in self.error_counts.items()
            }
            
            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                io_counters=io_counters,
                network_counters=network,
                response_times=response_times,
                error_rates=error_rates,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.log_error(f"Error collecting metrics: {e}")
            return None

    def _store_metrics(self, metrics: PerformanceMetrics) -> None:
        """Store metrics in database."""
        if not metrics:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''INSERT INTO performance_metrics
                        (timestamp, cpu_percent, memory_percent, io_counters,
                         network_counters, response_times, error_rates)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (metrics.timestamp,
                      metrics.cpu_percent,
                      metrics.memory_percent,
                      json.dumps(metrics.io_counters),
                      json.dumps(metrics.network_counters),
                      json.dumps(metrics.response_times),
                      json.dumps(metrics.error_rates)))
            
            conn.commit()
            conn.close()
            
            # Store in memory buffer
            self.metrics_history.append(metrics)
        except Exception as e:
            logger.log_error(f"Error storing metrics: {e}")

    def _analyze_metrics(self, metrics: PerformanceMetrics) -> None:
        """Analyze metrics and generate alerts if needed."""
        if not metrics:
            return
            
        try:
            # Check CPU usage
            if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
                self._create_alert('cpu_usage', 
                                 f"High CPU usage: {metrics.cpu_percent}%",
                                 metrics)
            
            # Check memory usage
            if metrics.memory_percent > self.alert_thresholds['memory_percent']:
                self._create_alert('memory_usage',
                                 f"High memory usage: {metrics.memory_percent}%",
                                 metrics)
            
            # Check response times
            for component, time in metrics.response_times.items():
                if time > self.alert_thresholds['response_time_ms']:
                    self._create_alert('response_time',
                                     f"Slow response time for {component}: {time}ms",
                                     metrics)
            
            # Check error rates
            for component, rate in metrics.error_rates.items():
                if rate > self.alert_thresholds['error_rate']:
                    self._create_alert('error_rate',
                                     f"High error rate for {component}: {rate*100}%",
                                     metrics)
        except Exception as e:
            logger.log_error(f"Error analyzing metrics: {e}")

    def _create_alert(self, alert_type: str, message: str, metrics: PerformanceMetrics) -> None:
        """Create and store performance alert."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''INSERT INTO performance_alerts
                        (timestamp, alert_type, message, metrics)
                        VALUES (?, ?, ?, ?)''',
                     (datetime.now().isoformat(),
                      alert_type,
                      message,
                      json.dumps(metrics.__dict__)))
            
            conn.commit()
            conn.close()
            
            logger.log_warning(f"Performance alert: {message}")
        except Exception as e:
            logger.log_error(f"Error creating alert: {e}")

    def record_response_time(self, component: str, time_ms: float) -> None:
        """Record component response time."""
        if component not in self.component_response_times:
            self.component_response_times[component] = deque(maxlen=100)
        self.component_response_times[component].append(time_ms)

    def record_error(self, component: str) -> None:
        """Record component error."""
        self.error_counts[component] = self.error_counts.get(component, 0) + 1

    def get_metrics(self, 
                   hours: Optional[int] = None,
                   components: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get performance metrics for specified time period."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            query = "SELECT * FROM performance_metrics WHERE 1=1"
            params = []
            
            if hours:
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                query += " AND timestamp >= ?"
                params.append(cutoff)
            
            c.execute(query, params)
            columns = [description[0] for description in c.description]
            metrics = []
            
            for row in c.fetchall():
                metric = dict(zip(columns, row))
                # Parse JSON fields
                for field in ['io_counters', 'network_counters', 
                            'response_times', 'error_rates']:
                    metric[field] = json.loads(metric[field])
                metrics.append(metric)
            
            # Filter by components if specified
            if components:
                metrics = [
                    m for m in metrics
                    if any(c in m['response_times'] for c in components)
                ]
            
            # Calculate statistics
            stats = self._calculate_statistics(metrics)
            
            return {
                'metrics': metrics,
                'statistics': stats,
                'components': list(self.component_response_times.keys()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.log_error(f"Error getting metrics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_statistics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from metrics."""
        if not metrics:
            return {}
            
        try:
            cpu_values = [m['cpu_percent'] for m in metrics]
            memory_values = [m['memory_percent'] for m in metrics]
            
            stats = {
                'cpu': {
                    'mean': mean(cpu_values),
                    'median': median(cpu_values),
                    'std_dev': stdev(cpu_values) if len(cpu_values) > 1 else 0,
                    'min': min(cpu_values),
                    'max': max(cpu_values)
                },
                'memory': {
                    'mean': mean(memory_values),
                    'median': median(memory_values),
                    'std_dev': stdev(memory_values) if len(memory_values) > 1 else 0,
                    'min': min(memory_values),
                    'max': max(memory_values)
                },
                'response_times': {},
                'error_rates': {}
            }
            
            # Calculate component-specific statistics
            components = set()
            for m in metrics:
                components.update(m['response_times'].keys())
            
            for component in components:
                response_times = [
                    m['response_times'].get(component, 0) 
                    for m in metrics
                    if component in m['response_times']
                ]
                
                if response_times:
                    stats['response_times'][component] = {
                        'mean': mean(response_times),
                        'median': median(response_times),
                        'std_dev': stdev(response_times) if len(response_times) > 1 else 0,
                        'min': min(response_times),
                        'max': max(response_times)
                    }
                
                error_rates = [
                    m['error_rates'].get(component, 0)
                    for m in metrics
                    if component in m['error_rates']
                ]
                
                if error_rates:
                    stats['error_rates'][component] = {
                        'mean': mean(error_rates),
                        'max': max(error_rates)
                    }
            
            return stats
        except Exception as e:
            logger.log_error(f"Error calculating statistics: {e}")
            return {}

    def get_alerts(self, 
                  hours: Optional[int] = None,
                  alert_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get performance alerts for specified time period."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            query = "SELECT * FROM performance_alerts WHERE 1=1"
            params = []
            
            if hours:
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                query += " AND timestamp >= ?"
                params.append(cutoff)
            
            if alert_types:
                placeholders = ','.join('?' for _ in alert_types)
                query += f" AND alert_type IN ({placeholders})"
                params.extend(alert_types)
            
            query += " ORDER BY timestamp DESC"
            
            c.execute(query, params)
            columns = [description[0] for description in c.description]
            alerts = []
            
            for row in c.fetchall():
                alert = dict(zip(columns, row))
                alert['metrics'] = json.loads(alert['metrics'])
                alerts.append(alert)
            
            return alerts
        except Exception as e:
            logger.log_error(f"Error getting alerts: {e}")
            return []

    def cleanup_old_data(self, days: int = 30) -> None:
        """Clean up old performance data."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Clean up metrics
            c.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff,))
            metrics_deleted = c.rowcount
            
            # Clean up alerts
            c.execute("DELETE FROM performance_alerts WHERE timestamp < ?", (cutoff,))
            alerts_deleted = c.rowcount
            
            conn.commit()
            conn.close()
            
            logger.log_info(
                f"Cleaned up old performance data: {metrics_deleted} metrics, "
                f"{alerts_deleted} alerts deleted"
            )
        except Exception as e:
            logger.log_error(f"Error cleaning up old data: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        self._stop_monitoring.set()
        if hasattr(self, '_monitoring_thread'):
            self._monitoring_thread.join()

# Create global monitor instance
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return performance_monitor

# Example usage
if __name__ == "__main__":
    monitor = get_performance_monitor()
    
    # Simulate some activity
    for i in range(5):
        # Record response times
        monitor.record_response_time("api", 100 + i * 10)
        monitor.record_response_time("database", 50 + i * 5)
        
        # Record some errors
        if i % 2 == 0:
            monitor.record_error("api")
        
        time.sleep(2)
    
    # Get metrics
    metrics = monitor.get_metrics(hours=1)
    print("\nPerformance metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Get alerts
    alerts = monitor.get_alerts(hours=1)
    print("\nPerformance alerts:")
    print(json.dumps(alerts, indent=2))
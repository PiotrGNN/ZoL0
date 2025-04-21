"""
Analizator historycznych metryk API
"""
import os
import json
import glob
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging

class APIMetricsAnalyzer:
    """Analizuje historyczne metryki API i generuje raporty."""
    
    def __init__(self, metrics_dir: str = "reports"):
        self.metrics_dir = metrics_dir
        self.logger = logging.getLogger('api_analyzer')
        if not self.logger.handlers:
            handler = logging.FileHandler('logs/api_analyzer.log')
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def load_metrics(self, days: int = 7) -> List[Dict[str, Any]]:
        """Wczytuje metryki z ostatnich X dni."""
        metrics = []
        start_date = datetime.now() - timedelta(days=days)
        
        pattern = os.path.join(self.metrics_dir, "*_metrics_*.json")
        for file_path in glob.glob(pattern):
            try:
                file_date = datetime.strptime(
                    os.path.basename(file_path).split('_')[2].split('.')[0],
                    '%Y%m%d%H%M%S'
                )
                if file_date >= start_date:
                    with open(file_path, 'r') as f:
                        metrics.append({
                            'timestamp': file_date,
                            'data': json.load(f)
                        })
            except Exception as e:
                self.logger.error(f"Błąd podczas wczytywania pliku {file_path}: {e}")
                
        return sorted(metrics, key=lambda x: x['timestamp'])

    def analyze_performance(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analizuje wydajność API na podstawie metryk."""
        if not metrics:
            return {"error": "Brak dostępnych metryk"}
            
        df = pd.DataFrame([
            {
                'timestamp': m['timestamp'],
                'success_rate': float(m['data']['general']['success_rate'].rstrip('%')),
                'avg_response_time': float(m['data']['general']['avg_response_time'].rstrip('s')),
                'total_requests': m['data']['general']['total_requests'],
                'rate_limit_hits': m['data']['rate_limits']['total_hits']
            }
            for m in metrics
        ])
        
        analysis = {
            'period': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'summary': {
                'total_requests': int(df['total_requests'].sum()),
                'avg_success_rate': f"{df['success_rate'].mean():.1f}%",
                'avg_response_time': f"{df['avg_response_time'].mean():.3f}s",
                'total_rate_limit_hits': int(df['rate_limit_hits'].sum())
            },
            'trends': {
                'success_rate_trend': self._calculate_trend(df['success_rate']),
                'response_time_trend': self._calculate_trend(df['avg_response_time']),
                'request_volume_trend': self._calculate_trend(df['total_requests'])
            }
        }
        
        # Generowanie wykresów
        self._generate_performance_plots(df)
        
        return analysis

    def _calculate_trend(self, series: pd.Series) -> str:
        """Oblicza trend dla serii danych."""
        if len(series) < 2:
            return "stable"
            
        first_half = series[:len(series)//2].mean()
        second_half = series[len(series)//2:].mean()
        
        diff = second_half - first_half
        if abs(diff) < 0.05 * first_half:
            return "stable"
        return "improving" if diff < 0 else "degrading"

    def _generate_performance_plots(self, df: pd.DataFrame):
        """Generuje wykresy wydajności."""
        plt.figure(figsize=(15, 10))
        
        # Wykres success rate
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['success_rate'], 'g-')
        plt.title('Success Rate Over Time')
        plt.ylabel('Success Rate (%)')
        plt.grid(True)
        
        # Wykres średniego czasu odpowiedzi
        plt.subplot(3, 1, 2)
        plt.plot(df['timestamp'], df['avg_response_time'], 'b-')
        plt.title('Average Response Time Over Time')
        plt.ylabel('Response Time (s)')
        plt.grid(True)
        
        # Wykres liczby requestów
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['total_requests'], 'r-')
        plt.title('Request Volume Over Time')
        plt.ylabel('Number of Requests')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Zapisz wykres
        plot_path = os.path.join(self.metrics_dir, f"performance_plots_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(plot_path)
        plt.close()
        
    def generate_report(self, days: int = 7) -> Dict[str, Any]:
        """Generuje pełny raport z analizy metryk."""
        metrics = self.load_metrics(days)
        analysis = self.analyze_performance(metrics)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_analyzed': f"Last {days} days",
            'performance_analysis': analysis,
            'recommendations': self._generate_recommendations(analysis)
        }
        
        # Zapisz raport
        report_path = os.path.join(
            self.metrics_dir,
            f"performance_report_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Zapisano raport do {report_path}")
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania raportu: {e}")
            
        return report
        
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generuje zalecenia na podstawie analizy."""
        recommendations = []
        
        # Analiza success rate
        success_rate = float(analysis['summary']['avg_success_rate'].rstrip('%'))
        if success_rate < 95:
            recommendations.append(
                f"Success rate ({success_rate:.1f}%) jest poniżej oczekiwanego poziomu 95%. "
                "Zalecane sprawdzenie logów błędów i implementacji obsługi błędów."
            )
            
        # Analiza czasu odpowiedzi
        response_time = float(analysis['summary']['avg_response_time'].rstrip('s'))
        if response_time > 1.0:
            recommendations.append(
                f"Średni czas odpowiedzi ({response_time:.2f}s) jest wysoki. "
                "Rozważ optymalizację zapytań lub zwiększenie zasobów."
            )
            
        # Analiza trendów
        if analysis['trends']['success_rate_trend'] == 'degrading':
            recommendations.append(
                "Zaobserwowano spadający trend success rate. "
                "Zalecany przegląd zmian w systemie z ostatniego okresu."
            )
            
        if analysis['trends']['response_time_trend'] == 'degrading':
            recommendations.append(
                "Zaobserwowano rosnący trend czasów odpowiedzi. "
                "Sprawdź wykorzystanie zasobów i możliwe wąskie gardła."
            )
            
        return recommendations
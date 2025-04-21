"""
Moduł wyjaśnialności AI (Explainable AI) - generuje zrozumiałe wyjaśnienia
decyzji podejmowanych przez modele AI w systemie.

Ten moduł odpowiada za:
1. Tworzenie czytelnych wyjaśnień decyzji handlowych
2. Generowanie wizualizacji procesu decyzyjnego
3. Identyfikację kluczowych czynników wpływających na decyzję
4. Prezentację historycznych porównań
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import base64
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

class AIExplainer:
    """
    System wyjaśnialności AI, generujący zrozumiałe wyjaśnienia decyzji handlowych.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicjalizacja systemu wyjaśnialności.
        
        Args:
            config: Konfiguracja systemu wyjaśnialności
        """
        self.config = config or {
            "explanation_detail_level": "medium",  # "low", "medium", "high"
            "include_technical_details": True,
            "include_charts": True,
            "max_factors_to_show": 5,
            "language": "pl"
        }
        
        self.explanation_templates = self._load_explanation_templates()
        logger.info("Zainicjalizowano system wyjaśnialności AI")
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """
        Ładuje szablony wyjaśnień.
        
        Returns:
            Słownik szablonów wyjaśnień
        """
        # Podstawowe szablony wyjaśnień
        templates = {
            "buy_decision": "Decyzja o otwarciu pozycji LONG została podjęta głównie ze względu na {main_factor}. Dodatkowe czynniki to: {secondary_factors}.",
            "sell_decision": "Decyzja o otwarciu pozycji SHORT została podjęta głównie ze względu na {main_factor}. Dodatkowe czynniki to: {secondary_factors}.",
            "hold_decision": "Decyzja o wstrzymaniu się z transakcją wynika z {main_factor}. System zwraca uwagę na: {secondary_factors}.",
            "close_position": "Decyzja o zamknięciu pozycji została podjęta z powodu {main_factor}. Istotne czynniki: {secondary_factors}.",
            "confidence_high": "System ma wysoką pewność co do tej decyzji ({confidence_pct}%).",
            "confidence_medium": "System ma umiarkowaną pewność co do tej decyzji ({confidence_pct}%).",
            "confidence_low": "System ma niską pewność co do tej decyzji ({confidence_pct}%) i zaleca dodatkową weryfikację.",
            "technical_factor": "Wskaźnik {indicator_name} osiągnął wartość {indicator_value}, co sugeruje {interpretation}.",
            "ai_factor": "Model AI {model_name} przewiduje {prediction} z pewnością {confidence}%.",
            "market_condition": "Obecny stan rynku charakteryzuje się {market_description}.",
            "risk_assessment": "Ryzyko transakcji oceniono na {risk_level}/10, głównie z powodu {risk_factor}.",
            "historical_comparison": "Podobna sytuacja wystąpiła {date}, skutkując {outcome}."
        }
        
        return templates
    
    def explain_decision(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generuje wyjaśnienie dla decyzji handlowej.
        
        Args:
            decision_data: Dane decyzji zawierające typ decyzji, pewność, czynniki wpływające, itp.
            
        Returns:
            Słownik zawierający różne formy wyjaśnienia (tekst, HTML, komponenty wizualne)
        """
        # Wyekstrahuj podstawowe dane decyzji
        decision_type = decision_data.get('decision', 'UNKNOWN')
        confidence = decision_data.get('confidence', 0.0)
        factors = decision_data.get('factors', [])
        model_decisions = decision_data.get('details', {}).get('model_decisions', {})
        model_confidences = decision_data.get('details', {}).get('model_confidences', {})
        model_weights = decision_data.get('details', {}).get('model_weights', {})
        similar_contexts = decision_data.get('details', {}).get('similar_historical_contexts', [])
        
        # Generuj tekstowe wyjaśnienie
        text_explanation = self._generate_text_explanation(
            decision_type, confidence, factors, model_decisions, model_confidences
        )
        
        # Generuj HTML z wizualizacjami
        html_explanation = self._generate_html_explanation(
            decision_type, confidence, factors, model_decisions, 
            model_confidences, model_weights, similar_contexts,
            decision_data.get('explanation', '')
        )
        
        # Generuj komponenty wizualne
        visual_components = {}
        if self.config.get("include_charts", True):
            visual_components["decision_chart"] = self._generate_decision_chart(model_decisions, model_confidences, model_weights)
            if similar_contexts:
                visual_components["historical_comparison"] = self._generate_historical_comparison(similar_contexts)
        
        # Zwróć pełne wyjaśnienie
        return {
            "text_explanation": text_explanation,
            "html_explanation": html_explanation,
            "visual_components": visual_components,
            "raw_factors": factors,
            "decision_type": decision_type,
            "confidence": confidence
        }
    
    def _generate_text_explanation(
        self, 
        decision_type: str, 
        confidence: float,
        factors: List[Dict[str, Any]],
        model_decisions: Dict[str, Any],
        model_confidences: Dict[str, float]
    ) -> str:
        """
        Generuje tekstowe wyjaśnienie decyzji.
        
        Args:
            decision_type: Typ decyzji (np. BUY, SELL, HOLD)
            confidence: Pewność decyzji (0.0-1.0)
            factors: Lista czynników wpływających na decyzję
            model_decisions: Decyzje poszczególnych modeli
            model_confidences: Pewności decyzji modeli
            
        Returns:
            Tekstowe wyjaśnienie decyzji
        """
        explanation_parts = []
        
        # 1. Nagłówek decyzji
        decision_template = None
        if decision_type in ["BUY", "LONG"]:
            decision_template = self.explanation_templates["buy_decision"]
        elif decision_type in ["SELL", "SHORT"]:
            decision_template = self.explanation_templates["sell_decision"]
        elif decision_type in ["HOLD", "WAIT"]:
            decision_template = self.explanation_templates["hold_decision"]
        elif decision_type in ["CLOSE", "EXIT"]:
            decision_template = self.explanation_templates["close_position"]
            
        # Jeśli mamy odpowiedni szablon i czynniki
        if decision_template and factors:
            # Wybierz główny czynnik i drugorzędne czynniki
            main_factor = factors[0]['description'] if factors and 'description' in factors[0] else "złożonych warunków rynkowych"
            secondary_factors_list = [f['description'] for f in factors[1:self.config["max_factors_to_show"]] if 'description' in f]
            secondary_factors = ", ".join(secondary_factors_list) if secondary_factors_list else "brak wyraźnych dodatkowych sygnałów"
            
            # Wygeneruj główne wyjaśnienie
            main_explanation = decision_template.format(
                main_factor=main_factor,
                secondary_factors=secondary_factors
            )
            explanation_parts.append(main_explanation)
        else:
            # Fallback jeśli nie mamy szablonu lub czynników
            explanation_parts.append(f"System podjął decyzję {decision_type}.")
            
        # 2. Informacja o pewności
        confidence_pct = int(confidence * 100)
        if confidence >= 0.8:
            explanation_parts.append(self.explanation_templates["confidence_high"].format(confidence_pct=confidence_pct))
        elif confidence >= 0.5:
            explanation_parts.append(self.explanation_templates["confidence_medium"].format(confidence_pct=confidence_pct))
        else:
            explanation_parts.append(self.explanation_templates["confidence_low"].format(confidence_pct=confidence_pct))
            
        # 3. Podsumowanie decyzji modeli
        if model_decisions:
            model_summary = []
            model_summary.append("\nWkład poszczególnych modeli:")
            
            for model_name, decision in model_decisions.items():
                conf = model_confidences.get(model_name, 0.0)
                conf_pct = int(conf * 100)
                
                if isinstance(decision, (int, float)):
                    decision_str = f"{decision:.4f}"
                else:
                    decision_str = str(decision)
                    
                model_summary.append(f"- {model_name}: {decision_str} (pewność: {conf_pct}%)")
                
            explanation_parts.append("\n".join(model_summary))
            
        return "\n\n".join(explanation_parts)
    
    def _generate_html_explanation(
        self, 
        decision_type: str, 
        confidence: float,
        factors: List[Dict[str, Any]],
        model_decisions: Dict[str, Any],
        model_confidences: Dict[str, float],
        model_weights: Dict[str, float],
        similar_contexts: List[Dict[str, Any]],
        raw_explanation: str = ""
    ) -> str:
        """
        Generuje wyjaśnienie decyzji w formacie HTML.
        
        Args:
            decision_type: Typ decyzji (np. BUY, SELL, HOLD)
            confidence: Pewność decyzji (0.0-1.0)
            factors: Lista czynników wpływających na decyzję
            model_decisions: Decyzje poszczególnych modeli
            model_confidences: Pewności decyzji modeli
            model_weights: Wagi modeli
            similar_contexts: Podobne historyczne konteksty
            raw_explanation: Surowe wyjaśnienie z Meta-Agenta
            
        Returns:
            Wyjaśnienie w formacie HTML
        """
        # CSS dla wyjaśnienia
        css = """
        <style>
            .ai-explanation {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .decision-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            .decision-type {
                font-size: 24px;
                font-weight: bold;
            }
            .decision-buy { color: #28a745; }
            .decision-sell { color: #dc3545; }
            .decision-hold { color: #ffc107; }
            .decision-close { color: #6c757d; }
            .confidence-meter {
                width: 150px;
                height: 30px;
                background: linear-gradient(to right, #dc3545, #ffc107, #28a745);
                border-radius: 15px;
                position: relative;
            }
            .confidence-indicator {
                width: 20px;
                height: 20px;
                background-color: #fff;
                border: 2px solid #343a40;
                border-radius: 50%;
                position: absolute;
                top: 3px;
                transform: translateX(-50%);
            }
            .confidence-value {
                margin-top: 5px;
                text-align: center;
                font-weight: bold;
            }
            .factors-section, .models-section, .history-section {
                margin-top: 20px;
                padding-top: 10px;
                border-top: 1px solid #ddd;
            }
            .factor-item {
                margin-bottom: 10px;
                padding: 10px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .factor-importance-high { border-left: 5px solid #28a745; }
            .factor-importance-medium { border-left: 5px solid #ffc107; }
            .factor-importance-low { border-left: 5px solid #6c757d; }
            .model-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }
            .model-table th, .model-table td {
                padding: 8px;
                border: 1px solid #ddd;
                text-align: left;
            }
            .model-table th {
                background-color: #f1f1f1;
            }
            .historical-case {
                margin-top: 15px;
                padding: 10px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border-left: 5px solid #17a2b8;
            }
            .raw-explanation {
                margin-top: 20px;
                padding: 15px;
                background-color: #f1f1f1;
                border-radius: 5px;
                white-space: pre-wrap;
                font-family: monospace;
            }
        </style>
        """
        
        # Mapowanie typów decyzji na klasy CSS
        decision_classes = {
            "BUY": "decision-buy", "LONG": "decision-buy",
            "SELL": "decision-sell", "SHORT": "decision-sell",
            "HOLD": "decision-hold", "WAIT": "decision-hold",
            "CLOSE": "decision-close", "EXIT": "decision-close"
        }
        
        # Nagłówek decyzji
        decision_class = decision_classes.get(decision_type, "")
        confidence_pct = int(confidence * 100)
        confidence_position = confidence * 100  # Pozycja wskaźnika pewności w %
        
        header = """
        <div class="decision-header">
            <div class="decision-type {decision_class}">{decision_type}</div>
            <div>
                <div class="confidence-meter">
                    <div class="confidence-indicator" style="left: {confidence_position}%;"></div>
                </div>
                <div class="confidence-value">Pewność: {confidence_pct}%</div>
            </div>
        </div>
        """.format(
            decision_class=decision_class,
            decision_type=decision_type,
            confidence_position=confidence_position,
            confidence_pct=confidence_pct
        )
        
        # Sekcja czynników
        factors_html = """
        <div class="factors-section">
            <h3>Kluczowe czynniki</h3>
        """
        
        # Dodaj czynniki jeśli są dostępne
        if factors:
            for i, factor in enumerate(factors[:self.config["max_factors_to_show"]]):
                importance = "high" if i == 0 else ("medium" if i < 3 else "low")
                description = factor.get('description', 'Nieznany czynnik')
                impact = factor.get('impact', 0.0)
                impact_str = f"{impact:.2f}" if isinstance(impact, (int, float)) else str(impact)
                
                factor_html = """
                <div class="factor-item factor-importance-{importance}">
                    <strong>{description}</strong>
                    <div>Wpływ: {impact_str}</div>
                </div>
                """.format(
                    importance=importance,
                    description=description,
                    impact_str=impact_str
                )
                factors_html += factor_html
        else:
            factors_html += "<p>Brak dostępnych czynników do wyświetlenia.</p>"
            
        factors_html += "</div>"
        
        # Sekcja modeli
        models_html = """
        <div class="models-section">
            <h3>Decyzje modeli</h3>
        """
        
        if model_decisions:
            models_html += """
            <table class="model-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Decyzja</th>
                        <th>Pewność</th>
                        <th>Waga</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for model_name, decision in model_decisions.items():
                model_conf = model_confidences.get(model_name, 0.0)
                weight = model_weights.get(model_name, 0.0)
                
                if isinstance(decision, (int, float)):
                    decision_str = f"{decision:.4f}"
                else:
                    decision_str = str(decision)
                    
                tr_html = """
                <tr>
                    <td>{model_name}</td>
                    <td>{decision_str}</td>
                    <td>{confidence_pct}%</td>
                    <td>{weight:.2f}</td>
                </tr>
                """.format(
                    model_name=model_name,
                    decision_str=decision_str,
                    confidence_pct=int(model_conf * 100),
                    weight=weight
                )
                models_html += tr_html
                
            models_html += """
                </tbody>
            </table>
            """
        else:
            models_html += "<p>Brak dostępnych danych o modelach.</p>"
            
        models_html += "</div>"
        
        # Sekcja historycznych porównań
        history_html = """
        <div class="history-section">
            <h3>Podobne sytuacje historyczne</h3>
        """
        
        if similar_contexts:
            for i, context in enumerate(similar_contexts[:3]):  # Pokaż maksymalnie 3 podobne przypadki
                timestamp = context.get('timestamp', 'unknown date')
                similarity = context.get('similarity', 0.0)
                decision = context.get('decision', 'unknown')
                result = context.get('result', {})
                
                profit = result.get('profit', 'N/A')
                if isinstance(profit, (int, float)):
                    profit_str = f"{profit:.2f}%"
                    profit_color = "#28a745" if profit > 0 else "#dc3545"
                else:
                    profit_str = str(profit)
                    profit_color = "#6c757d"
                
                historical_case = """
                <div class="historical-case">
                    <div><strong>Data: </strong>{timestamp}</div>
                    <div><strong>Podobieństwo: </strong>{similarity_pct}%</div>
                    <div><strong>Decyzja: </strong>{decision}</div>
                    <div><strong>Wynik: </strong><span style="color: {profit_color}">{profit_str}</span></div>
                </div>
                """.format(
                    timestamp=timestamp,
                    similarity_pct=int(similarity * 100),
                    decision=decision,
                    profit_color=profit_color,
                    profit_str=profit_str
                )
                history_html += historical_case
        else:
            history_html += "<p>Brak podobnych przypadków historycznych.</p>"
            
        history_html += "</div>"
        
        # Surowe wyjaśnienie jeśli jest dostępne i włączone szczegóły techniczne
        raw_explanation_html = ""
        if raw_explanation and self.config.get("include_technical_details", True):
            raw_explanation_html = """
            <div class="raw-explanation">
                {raw_explanation}
            </div>
            """.format(raw_explanation=raw_explanation.replace("\n", "<br>"))
        
        # Złożenie pełnego HTML
        full_html = """
        {css}
        <div class="ai-explanation">
            {header}
            {factors_html}
            {models_html}
            {history_html}
            {raw_explanation_html}
        </div>
        """.format(
            css=css,
            header=header,
            factors_html=factors_html,
            models_html=models_html,
            history_html=history_html,
            raw_explanation_html=raw_explanation_html
        )
        
        return full_html
    
    def _generate_decision_chart(
        self, 
        model_decisions: Dict[str, Any], 
        model_confidences: Dict[str, float], 
        model_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generuje wykres decyzji modeli.
        
        Args:
            model_decisions: Decyzje poszczególnych modeli
            model_confidences: Pewności decyzji modeli
            model_weights: Wagi modeli
            
        Returns:
            Dane wykresu decyzji
        """
        # Przygotuj dane do wykresu
        models = []
        decisions = []
        confidences = []
        weights = []
        colors = []
        
        # Przygotuj dane tylko dla modeli z decyzjami numerycznymi
        numeric_models = {}
        for model, decision in model_decisions.items():
            if isinstance(decision, (int, float)):
                numeric_models[model] = decision
                
        if numeric_models:
            for model, decision in numeric_models.items():
                models.append(model)
                decisions.append(decision)
                conf = model_confidences.get(model, 0.5)
                confidences.append(conf)
                weights.append(model_weights.get(model, 1.0))
                
                # Kolor zależy od kierunku decyzji (zielony dla dodatnich, czerwony dla ujemnych)
                color = 'g' if decision > 0 else ('r' if decision < 0 else 'gray')
                colors.append(color)
            
            # Utwórz wykres
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Rozmiar kropek zależy od wagi * pewność
            sizes = [w * c * 200 for w, c in zip(weights, confidences)]
            
            # Przeskaluj decyzje do wspólnego zakresu (jeśli są bardzo różne)
            max_abs_decision = max(abs(d) for d in decisions) if decisions else 1
            normalized_decisions = [d / max_abs_decision for d in decisions]
            
            # Utwórz scatter plot
            scatter = ax.scatter(
                range(len(models)),
                normalized_decisions,
                s=sizes,
                c=colors,
                alpha=0.6
            )
            
            # Dodaj etykiety i tytuł
            ax.set_xlabel('Model')
            ax.set_ylabel('Znormalizowana decyzja')
            ax.set_title('Decyzje modeli (rozmiar = waga * pewność)')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Dodaj linię zerową
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Dodaj siatkę
            ax.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            
            # Konwertuj wykres do base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return {
                "type": "image",
                "format": "base64",
                "data": img_base64,
                "alt": "Wykres decyzji modeli"
            }
            
        return None
    
    def _generate_historical_comparison(self, similar_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generuje wykres porównania do podobnych sytuacji historycznych.
        
        Args:
            similar_contexts: Lista podobnych kontekstów historycznych
            
        Returns:
            Dane wykresu porównania
        """
        if not similar_contexts:
            return None
            
        # Przygotuj dane
        dates = []
        similarities = []
        profits = []
        
        for context in similar_contexts:
            try:
                timestamp = context.get('timestamp', '')
                date = pd.to_datetime(timestamp)
                similarity = context.get('similarity', 0)
                result = context.get('result', {})
                profit = result.get('profit', 0)
                
                dates.append(date)
                similarities.append(similarity)
                profits.append(profit)
            except:
                # Pomiń błędne wpisy
                pass
                
        if not dates:
            return None
        
        # Utwórz wykres
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Górny wykres: podobieństwa
        ax1.bar(dates, similarities, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Podobieństwo')
        ax1.set_title('Podobieństwo do bieżącej sytuacji')
        
        # Dolny wykres: zyski/straty
        colors = ['green' if p > 0 else 'red' for p in profits]
        ax2.bar(dates, profits, color=colors, alpha=0.7)
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Zysk/strata (%)')
        ax2.set_title('Wyniki podobnych przypadków historycznych')
        
        # Formatowanie dat
        fig.autofmt_xdate()
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_formatter(date_format)
        
        plt.tight_layout()
        
        # Konwertuj wykres do base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return {
            "type": "image",
            "format": "base64",
            "data": img_base64,
            "alt": "Porównanie z podobnymi sytuacjami historycznymi"
        }
    
    def extract_key_factors(self, market_data: Dict[str, Any], model_outputs: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Ekstrahuje kluczowe czynniki z danych rynkowych i wyników modeli.
        
        Args:
            market_data: Dane rynkowe
            model_outputs: Wyniki modeli
            limit: Maksymalna liczba czynników do zwrócenia
            
        Returns:
            Lista kluczowych czynników
        """
        factors = []
        
        # 1. Czynniki z danych rynkowych
        if isinstance(market_data, dict):
            # Trendy cenowe
            if 'close' in market_data and isinstance(market_data['close'], (list, np.ndarray)) and len(market_data['close']) >= 10:
                closes = market_data['close']
                current_price = closes[-1]
                prev_price = closes[-10] if len(closes) >= 10 else closes[0]
                price_change = (current_price - prev_price) / prev_price
                
                if abs(price_change) > 0.01:  # Zmiana ceny > 1%
                    direction = "wzrostowy" if price_change > 0 else "spadkowy"
                    factors.append({
                        'type': 'price_trend',
                        'description': f"Trend {direction} ({price_change:.2%} w ostatnich 10 okresach)",
                        'impact': price_change,
                        'importance': 'high'
                    })
            
            # Wskaźniki techniczne
            for indicator_name in ['rsi', 'macd', 'sma_20', 'ema_50']:
                if indicator_name in market_data and isinstance(market_data[indicator_name], (list, np.ndarray)) and len(market_data[indicator_name]) > 0:
                    value = market_data[indicator_name][-1]
                    
                    # Interpretacja RSI
                    if indicator_name == 'rsi' and isinstance(value, (int, float)):
                        if value > 70:
                            factors.append({
                                'type': 'technical_indicator',
                                'description': f"RSI wskazuje na wykupienie rynku (wartość: {value:.2f})",
                                'impact': -0.7,  # Negatywny wpływ - sugeruje sprzedaż
                                'importance': 'high'
                            })
                        elif value < 30:
                            factors.append({
                                'type': 'technical_indicator',
                                'description': f"RSI wskazuje na wyprzedanie rynku (wartość: {value:.2f})",
                                'impact': 0.7,  # Pozytywny wpływ - sugeruje kupno
                                'importance': 'high'
                            })
                    
                    # Interpretacja MACD
                    elif indicator_name == 'macd' and 'macd_signal' in market_data:
                        macd_val = value
                        signal = market_data['macd_signal'][-1] if isinstance(market_data['macd_signal'], (list, np.ndarray)) and len(market_data['macd_signal']) > 0 else 0
                        
                        if isinstance(macd_val, (int, float)) and isinstance(signal, (int, float)):
                            if macd_val > signal:
                                factors.append({
                                    'type': 'technical_indicator',
                                    'description': f"MACD powyżej linii sygnałowej (różnica: {macd_val - signal:.4f})",
                                    'impact': 0.5,  # Pozytywny wpływ - sugeruje kupno
                                    'importance': 'medium'
                                })
                            elif macd_val < signal:
                                factors.append({
                                    'type': 'technical_indicator',
                                    'description': f"MACD poniżej linii sygnałowej (różnica: {macd_val - signal:.4f})",
                                    'impact': -0.5,  # Negatywny wpływ - sugeruje sprzedaż
                                    'importance': 'medium'
                                })
        
        # 2. Czynniki z wyników modeli
        for model_name, output in model_outputs.items():
            if isinstance(output, dict) and 'explanation' in output and output['explanation']:
                factors.append({
                    'type': 'model_output',
                    'description': f"Model {model_name}: {output['explanation']}",
                    'impact': output.get('confidence', 0.5) * (1 if output.get('decision') in ['BUY', 'LONG'] else -1 if output.get('decision') in ['SELL', 'SHORT'] else 0),
                    'importance': 'high' if model_name.lower() in ['meta_agent', 'ensemble'] else 'medium'
                })
            
        # Sortuj czynniki według ważności (wartość bezwzględna wpływu)
        factors.sort(key=lambda x: abs(x.get('impact', 0)), reverse=True)
        
        return factors[:limit]
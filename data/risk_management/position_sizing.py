"""
position_sizing.py
------------------
Moduł obliczający optymalne rozmiary pozycji w zależności od ryzyka i dostępnego kapitału.

Funkcjonalności:
- Implementacja metod określania rozmiaru pozycji (fixed fractional, risk parity)
- Dynamiczne dostosowywanie wielkości pozycji na podstawie zmienności
- Integracja z mechanizmami stop-loss i dźwigni
"""

import logging
import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def calculate_position_size(account_balance: float, risk_per_trade: float, entry_price: float, stop_loss_price: float) -> float:
    """
    Oblicza wielkość pozycji na podstawie ryzyka i różnicy między ceną wejścia a stop-lossem.
    
    Parameters:
        account_balance (float): Całkowity dostępny kapitał
        risk_per_trade (float): Maksymalny procent kapitału ryzykowany na transakcję
        entry_price (float): Cena wejścia w pozycję
        stop_loss_price (float): Poziom stop-loss
        
    Returns:
        float: Wielkość pozycji w jednostkach bazowych
    """
    if account_balance <= 0 or risk_per_trade <= 0 or entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("Wszystkie parametry muszą być dodatnie")
        
    if stop_loss_price >= entry_price:
        raise ValueError("Stop-loss musi być poniżej ceny wejścia dla pozycji long")
        
    risk_amount = account_balance * risk_per_trade
    price_difference = abs(entry_price - stop_loss_price)
    position_size = risk_amount / price_difference
    
    logging.info(f"Obliczona wielkość pozycji: {position_size:.4f} jednostek")
    return position_size

def risk_parity_position_size(returns: pd.Series, account_balance: float, vol_target: float = 0.15) -> dict:
    """
    Oblicza wielkości pozycji używając metody równego ryzyka (risk parity).
    
    Parameters:
        returns (pd.Series): Historyczne zwroty
        account_balance (float): Dostępny kapitał
        vol_target (float): Docelowa zmienność portfela
        
    Returns:
        dict: Wielkości pozycji dla każdego aktywa
    """
    if len(returns) == 0:
        raise ValueError("Szereg zwrotów nie może być pusty")
        
    # Oblicz zmienności
    vols = returns.std()
    
    # Oblicz wagi inverse variance
    if isinstance(vols, (float, int)):
        weights = {returns.name if returns.name else 'asset': 1.0}
    else:
        inv_var = 1.0 / (vols ** 2)
        weights = (inv_var / inv_var.sum()).to_dict()
    
    # Przeskaluj do docelowej zmienności
    port_vol = np.sqrt((returns ** 2).mean())
    scaling = vol_target / port_vol if port_vol > 0 else 1.0
    
    # Oblicz finalne wielkości pozycji
    position_sizes = {asset: weight * account_balance * scaling 
                     for asset, weight in weights.items()}
    
    logging.info(f"Obliczone wielkości pozycji risk parity: {position_sizes}")
    return position_sizes

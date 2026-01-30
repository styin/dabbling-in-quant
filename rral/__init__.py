"""RRAL Framework - Regressive Risk-Adjusted Leverage for Backtesting"""

from .framework import (
    ewma_volatility,
    liquidation_ceiling,
    glide_path_multiplier,
    kelly_leverage,
    volatility_leverage,
    portfolio_yield,
    compute_final_leverage,
    compute_ewma_vol_series,
)
from .strategy import RRALStrategy
from .data_loader import fetch_data

__all__ = [
    'ewma_volatility',
    'liquidation_ceiling',
    'glide_path_multiplier',
    'kelly_leverage',
    'volatility_leverage',
    'portfolio_yield',
    'compute_final_leverage',
    'compute_ewma_vol_series',
    'RRALStrategy',
    'fetch_data',
]

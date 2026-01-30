"""
RRAL Framework v2.0 - Core Mathematical Functions

Regressive Risk-Adjusted Leverage framework for dynamic leverage calculation.
"""

import numpy as np
import pandas as pd
from typing import Optional


def ewma_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
    annualization_factor: float = 252
) -> pd.Series:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) volatility.
    
    Uses the RiskMetrics standard lambda=0.94 for daily returns.
    
    Args:
        returns: Series of log returns
        lambda_: Decay factor (0.94 is RiskMetrics standard)
        annualization_factor: Trading days per year (252 for daily)
    
    Returns:
        Series of annualized EWMA volatility
    """
    # Calculate squared returns
    squared_returns = returns ** 2
    
    # EWMA variance: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_t
    ewma_var = squared_returns.ewm(alpha=1 - lambda_, adjust=False).mean()
    
    # Annualize: daily vol * sqrt(252)
    ewma_vol = np.sqrt(ewma_var) * np.sqrt(annualization_factor)
    
    return ewma_vol


def liquidation_ceiling(
    d_gap: float = 0.30,
    m_stress: float = 0.40,
    buffer: float = 0.05
) -> float:
    """
    Calculate the maximum leverage that guarantees survival against a gap event.
    
    Formula: L_liq = (1 - B) / (D_gap + M_stress * (1 - D_gap))
    
    Args:
        d_gap: Gap buffer - maximum instantaneous drop (default 30%)
        m_stress: Stress-tested maintenance margin (default 40%)
        buffer: Safety buffer for residual equity (default 5%)
    
    Returns:
        Maximum leverage for liquidation survival
    """
    numerator = 1 - buffer
    denominator = d_gap + (m_stress * (1 - d_gap))
    return numerator / denominator


def glide_path_multiplier(
    years_to_retirement: float,
    max_years: float = 20
) -> float:
    """
    Calculate the lifecycle glide path multiplier for Kelly adjustment.
    
    Formula: C_age = 0.5 * min(1, years_to_retirement / max_years)
    
    Uses 0.5 (Half-Kelly) base for parameter safety.
    
    Args:
        years_to_retirement: Years until retirement
        max_years: Maximum years for full multiplier (default 20)
    
    Returns:
        Glide path multiplier C_age in [0, 0.5]
    """
    return 0.5 * min(1.0, years_to_retirement / max_years)


def kelly_leverage(
    mu_yield: float,
    r_debt: float,
    sigma: float,
    c_age: float
) -> float:
    """
    Calculate the lifecycle-adjusted Kelly leverage.
    
    Formula: L_kelly = C_age * (μ_yield - r_debt) / σ²
    
    If μ_yield < r_debt (negative expected return), returns 0.
    Kelly says "don't invest" when expected return doesn't cover debt cost.
    
    Args:
        mu_yield: Portfolio earnings yield (1/PE)
        r_debt: Cost of debt (margin interest rate)
        sigma: Current annualized volatility
        c_age: Glide path multiplier from lifecycle calculation
    
    Returns:
        Kelly-optimal leverage, adjusted for lifecycle.
        Returns 0 if expected return < debt cost.
    """
    if sigma <= 0:
        return 0.0
    
    excess_return = mu_yield - r_debt
    
    # If earnings yield doesn't cover debt cost, Kelly says "don't lever"
    if excess_return <= 0:
        return 0.0
    
    return c_age * excess_return / (sigma ** 2)


def volatility_leverage(
    l_kelly: float,
    sigma_bench: float,
    sigma_current: float
) -> float:
    """
    Calculate the volatility-matching leverage limit.
    
    Formula: L_vol = L_kelly * σ_bench / σ_current
    
    This ensures portfolio volatility doesn't exceed the implied
    volatility of the Kelly leverage applied to the benchmark.
    
    If L_kelly is 0 (no leverage justified), L_vol is also 0.
    
    Args:
        l_kelly: Kelly leverage from strategic calculation
        sigma_bench: Benchmark volatility (e.g., Nasdaq ~22%)
        sigma_current: Current portfolio volatility
    
    Returns:
        Volatility-matched leverage limit
    """
    if sigma_current <= 0 or l_kelly <= 0:
        return 0.0
    
    return l_kelly * sigma_bench / sigma_current


def portfolio_yield(
    weights: np.ndarray,
    pe_ratios: np.ndarray,
    s_bias: float = 0.15,
    default_pe: float = 100.0
) -> float:
    """
    Calculate weighted portfolio earnings yield with conservative penalty.
    
    Formula: μ_yield = Σ(w_i * (1/PE_i)) * (1 - S_bias)
    
    Args:
        weights: Portfolio weights (must sum to 1)
        pe_ratios: P/E ratios for each stock (negative/zero = unprofitable)
        s_bias: Penalty factor for overoptimism (default 15%)
        default_pe: Default P/E for unprofitable stocks (default 100)
    
    Returns:
        Portfolio earnings yield after penalty
    """
    # Handle unprofitable stocks (PE <= 0 or NaN)
    safe_pe = np.where(
        (pe_ratios <= 0) | np.isnan(pe_ratios),
        default_pe,
        pe_ratios
    )
    
    # Calculate weighted earnings yield
    yields = 1.0 / safe_pe
    weighted_yield = np.sum(weights * yields)
    
    # Apply penalty for overoptimism
    return weighted_yield * (1 - s_bias)


def compute_final_leverage(
    sigma_current: float,
    sigma_bench: float = 0.22,
    mu_yield: float = 0.04,
    r_debt: float = 0.07,
    years_to_retirement: float = 30,
    d_gap: float = 0.30,
    m_stress: float = 0.40,
    buffer: float = 0.05,
    min_leverage: float = 0.5,
    max_leverage: float = 2.0
) -> dict:
    """
    Compute the final RRAL leverage as the minimum of all constraints.
    
    Formula: L_final = min(L_liq, L_kelly, L_vol), clamped to [min, max]
    
    Args:
        sigma_current: Current annualized portfolio volatility
        sigma_bench: Benchmark volatility (default 22% for Nasdaq)
        mu_yield: Portfolio earnings yield (default 4%)
        r_debt: Cost of debt (default 7%)
        years_to_retirement: Years until retirement (default 30)
        d_gap: Gap buffer for liquidation (default 30%)
        m_stress: Stress margin (default 40%)
        buffer: Safety buffer (default 5%)
        min_leverage: Minimum allowed leverage (default 0.5)
        max_leverage: Maximum allowed leverage (default 2.0)
    
    Returns:
        Dictionary with L_liq, L_kelly, L_vol, L_final
    """
    # Step 1: Liquidation ceiling (hard floor)
    l_liq = liquidation_ceiling(d_gap, m_stress, buffer)
    
    # Step 2: Lifecycle-adjusted Kelly
    c_age = glide_path_multiplier(years_to_retirement)
    l_kelly = kelly_leverage(mu_yield, r_debt, sigma_current, c_age)
    
    # Step 3: Volatility matching
    l_vol = volatility_leverage(l_kelly, sigma_bench, sigma_current)
    
    # Step 4: Final leverage = min of all, clamped
    l_raw = min(l_liq, l_kelly, l_vol)
    l_final = np.clip(l_raw, min_leverage, max_leverage)
    
    return {
        'L_liq': l_liq,
        'L_kelly': l_kelly,
        'L_vol': l_vol,
        'L_raw': l_raw,
        'L_final': l_final,
        'C_age': c_age,
        'sigma_current': sigma_current
    }


def compute_ewma_vol_series(
    close_prices,
    lambda_: float = 0.94
) -> np.ndarray:
    """
    Compute EWMA volatility series from close prices.
    
    Convenience wrapper for use with Backtesting.py indicators.
    Works with both pandas Series and numpy arrays.
    
    Args:
        close_prices: Series or array of closing prices
        lambda_: EWMA decay factor
    
    Returns:
        Array of annualized EWMA volatility
    """
    # Convert to numpy array if needed (Backtesting.py uses _Array)
    prices = np.asarray(close_prices, dtype=float)
    
    # Calculate log returns using numpy
    # Shift manually: prepend a NaN to simulate shift(1)
    shifted_prices = np.concatenate([[np.nan], prices[:-1]])
    log_returns = np.log(prices / shifted_prices)
    
    # Replace NaN with 0 for the first element
    log_returns = np.nan_to_num(log_returns, nan=0.0)
    
    # Convert to pandas Series for EWMA calculation
    log_returns_series = pd.Series(log_returns)
    
    # Calculate EWMA volatility and return as numpy array
    vol_series = ewma_volatility(log_returns_series, lambda_)
    return vol_series.values

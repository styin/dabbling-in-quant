"""
RRAL Strategy - Backtesting.py Strategy Implementation

Implements the RRAL v2.0 framework as a Backtesting.py Strategy class.

Trading Logic:
- Start fully invested (100% position) on day 1
- Monitor optimal leverage (L_final) daily
- Rebalance ONLY when actual leverage deviates from optimal by ±threshold
- Log all trades with leverage details

Expected Return Calculation (Growth-Adjusted):
- μ = 1/ForwardPE + min(g_est, g_cap)
- g_cap prevents chasing hype/bubbles
"""

import numpy as np
from backtesting import Strategy
from .framework import (
    compute_ewma_vol_series,
    liquidation_ceiling,
    glide_path_multiplier,
    kelly_leverage,
    volatility_leverage,
)


def compute_leverage_series_dynamic(
    close_prices,
    forward_pe_series,
    growth_rate_series,
    risk_free_series,
    ewma_lambda: float,
    sigma_bench: float,
    volatility_surplus: float,
    margin_spread: float,
    years_to_retirement: float,
    d_gap: float,
    m_stress: float,
    buffer: float,
    s_bias: float,
    g_cap: float,
    min_leverage: float,
    max_leverage: float,
) -> tuple:
    """
    Compute all RRAL leverage components as series for plotting.
    
    Uses growth-adjusted expected return:
    μ = (1/ForwardPE) * (1 - s_bias) + min(g_est, g_cap)
    
    Returns tuple of (L_liq, L_kelly, L_vol, L_final) arrays.
    """
    # Convert to numpy
    prices = np.asarray(close_prices, dtype=float)
    fwd_pe_arr = np.asarray(forward_pe_series, dtype=float)
    growth_arr = np.asarray(growth_rate_series, dtype=float)
    rf_arr = np.asarray(risk_free_series, dtype=float)
    n = len(prices)
    
    # Calculate log returns
    shifted_prices = np.concatenate([[np.nan], prices[:-1]])
    log_returns = np.log(prices / shifted_prices)
    log_returns = np.nan_to_num(log_returns, nan=0.0)
    
    # EWMA variance calculation
    alpha = 1 - ewma_lambda
    ewma_var = np.zeros(n)
    squared_returns = log_returns ** 2
    
    ewma_var[0] = squared_returns[0]
    for i in range(1, n):
        ewma_var[i] = alpha * squared_returns[i] + ewma_lambda * ewma_var[i-1]
    
    # Annualized volatility
    sigma_current = np.sqrt(ewma_var) * np.sqrt(252)
    sigma_current = np.clip(sigma_current, 0.05, None)  # Floor at 5%
    
    # Compute static liquidation ceiling
    l_liq = liquidation_ceiling(d_gap, m_stress, buffer)
    
    # Compute glide path (static based on years)
    c_age = glide_path_multiplier(years_to_retirement)
    
    # Target volatility = sigma_bench + surplus
    sigma_target = sigma_bench + volatility_surplus
    
    # Compute dynamic values
    l_kelly = np.zeros(n)
    l_vol = np.zeros(n)
    l_final = np.zeros(n)
    
    for i in range(n):
        sig = sigma_current[i]
        
        # Forward P/E -> earnings yield with penalty
        fwd_pe = fwd_pe_arr[i] if not np.isnan(fwd_pe_arr[i]) and fwd_pe_arr[i] > 0 else 25.0
        earnings_yield = (1.0 / fwd_pe) * (1 - s_bias)
        
        # Growth rate, capped
        g_est = growth_arr[i] if not np.isnan(growth_arr[i]) else 0.05
        g_capped = min(g_est, g_cap)
        
        # Expected return μ = yield + capped growth
        mu_yield = earnings_yield + g_capped
        
        # Dynamic r_debt = risk_free + margin_spread
        rf = rf_arr[i] if not np.isnan(rf_arr[i]) else 0.04
        r_debt = rf + margin_spread
        
        # Kelly leverage (returns 0 if mu < r_debt)
        l_k = kelly_leverage(mu_yield, r_debt, sig, c_age)
        l_kelly[i] = l_k
        
        # Volatility leverage using target vol (sigma_bench + surplus)
        # If L_kelly is 0 (no leverage justified), L_vol is also 0
        if sig > 0 and l_k > 0:
            l_v = l_k * sigma_target / sig
        else:
            l_v = 0.0
        l_vol[i] = l_v
        
        # Final leverage = min of all, clamped
        l_raw = min(l_liq, l_k, l_v)
        l_final[i] = np.clip(l_raw, min_leverage, max_leverage)
    
    # Return L_liq as constant array for plotting
    l_liq_arr = np.full(n, l_liq)
    
    return l_liq_arr, l_kelly, l_vol, l_final


class RRALStrategy(Strategy):
    """
    RRAL (Regressive Risk-Adjusted Leverage) Strategy.
    
    Dynamically monitors leverage based on three risk horizons:
    1. Liquidation Floor (gap survival)
    2. Strategic Growth (lifecycle-adjusted Kelly with growth)
    3. Tactical Stability (volatility matching)
    
    Expected Return (Growth-Adjusted):
    μ = 1/ForwardPE * (1 - s_bias) + min(g_est, g_cap)
    
    Rebalances only when actual leverage deviates significantly from optimal.
    Logs all trades with leverage details.
    """
    
    # === Strategy Parameters (optimizable) ===
    
    # Lifecycle parameters
    years_to_retirement: int = 20  # Years until retirement
    
    # Valuation parameters
    # portfolio_pe: float = 25.0  # Fallback P/E if not in data
    s_bias: float = 0  # Penalty for overoptimism on yield
    g_cap: float = 0.15  # Growth cap (15%) to prevent hype-chasing
    
    # Liquidation parameters
    d_gap: float = 0.30  # Gap buffer (30% crash scenario)
    m_stress: float = 0.40  # Stress-tested margin requirement
    buffer: float = 0.05  # Safety buffer
    
    # Cost parameters
    margin_spread: float = 0.01  # Spread over risk-free rate (1%)
    fallback_risk_free: float = 0  # Fallback if not in data
    
    # Volatility parameters
    sigma_bench: float = 0.27  # Benchmark vol (Nasdaq ~22%)
    volatility_surplus: float = 0  # Desired surplus over benchmark
    ewma_lambda: float = 0.94  # RiskMetrics EWMA decay
    
    # Leverage bounds
    min_leverage: float = 1  # Minimum leverage floor
    max_leverage: float = 2.0  # Maximum leverage ceiling
    
    # Rebalancing threshold: rebalance when |actual - optimal| / optimal > threshold
    rebalance_threshold: float = 0.10  # 10% deviation triggers rebalance
    
    # Warmup period for EWMA volatility
    warmup_period: int = 20
    
    # Logging
    verbose_trades: bool = True  # Log trade details
    
    def init(self):
        """Initialize indicators and state."""
        
        # Get Forward PE from data or fallback to trailing PE
        if hasattr(self.data, 'ForwardPE'):
            fwd_pe_series = self.data.ForwardPE
        elif hasattr(self.data, 'PE'):
            fwd_pe_series = self.data.PE
        else:
            fwd_pe_series = np.full(len(self.data.Close), self.portfolio_pe)
        
        # Get growth rate from data or use default
        if hasattr(self.data, 'GrowthRate'):
            growth_series = self.data.GrowthRate
        else:
            growth_series = np.full(len(self.data.Close), 0.05)  # 5% default
        
        # Get risk-free rate from data or use static fallback
        if hasattr(self.data, 'RiskFreeRate'):
            rf_series = self.data.RiskFreeRate
        else:
            rf_series = np.full(len(self.data.Close), self.fallback_risk_free)
        
        # Precompute all leverage series as indicators
        leverage_tuple = self.I(
            compute_leverage_series_dynamic,
            self.data.Close,
            fwd_pe_series,
            growth_series,
            rf_series,
            self.ewma_lambda,
            self.sigma_bench,
            self.volatility_surplus,
            self.margin_spread,
            self.years_to_retirement,
            self.d_gap,
            self.m_stress,
            self.buffer,
            self.s_bias,
            self.g_cap,
            self.min_leverage,
            self.max_leverage,
            name=('L_liq', 'L_kelly', 'L_vol', 'L_final'),
            overlay=False,
            color=('#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'),
        )
        
        # Unpack the indicators
        self.L_liq = leverage_tuple[0]
        self.L_kelly = leverage_tuple[1]
        self.L_vol = leverage_tuple[2]
        self.L_final = leverage_tuple[3]
        
        # Also compute EWMA volatility for reference
        self.ewma_vol = self.I(
            compute_ewma_vol_series,
            self.data.Close,
            self.ewma_lambda,
            name='EWMA_Vol',
            overlay=False,
            color='#9B59B6',
        )
        
        # Track state
        self._initial_entry_done = False
        self._trade_count = 0
    
    def next(self):
        """Execute on each bar."""
        
        # Calculate actual leverage
        actual_lev = self._calculate_actual_leverage()
        
        # === Day 1: Enter full 100% position immediately ===
        if not self._initial_entry_done and not self.position:
            self._log_trade("INITIAL_ENTRY", actual_lev, 1.0, "Full 100% position")
            self.buy(size=0.9999)
            self._initial_entry_done = True
            return
        
        # === Warmup: Need volatility data before monitoring ===
        if len(self.data) < self.warmup_period:
            return
        
        # === Daily Monitoring: Get optimal leverage from precomputed indicator ===
        optimal_leverage = self.L_final[-1]
        
        # === Check actual vs optimal leverage ===
        if not self.position:
            self._log_trade("RE_ENTRY", actual_lev, optimal_leverage, "Position was closed")
            self._enter_at_leverage(optimal_leverage)
            return
        
        # Calculate deviation: |actual - optimal| / optimal
        if optimal_leverage > 0:
            deviation = abs(actual_lev - optimal_leverage) / optimal_leverage
        else:
            deviation = 1.0
        
        # === Rebalance only if deviation exceeds threshold ===
        if deviation > self.rebalance_threshold:
            reason = f"Deviation {deviation*100:.1f}% > threshold {self.rebalance_threshold*100:.0f}%"
            self._log_trade("REBALANCE", actual_lev, optimal_leverage, reason)
            self._rebalance_to_leverage(optimal_leverage)
    
    def _log_trade(self, action: str, actual_lev: float, target_lev: float, reason: str):
        """Log trade details."""
        if not self.verbose_trades:
            return
        
        self._trade_count += 1
        date = self.data.index[-1]
        price = self.data.Close[-1]
        equity = self.equity
        
        print(f"\n{'='*70}")
        print(f"TRADE #{self._trade_count}: {action}")
        print(f"{'='*70}")
        print(f"  Date:             {date}")
        print(f"  Price:            ${price:.2f}")
        print(f"  Equity:           ${equity:,.2f}")
        print(f"  Actual Leverage:  {actual_lev:.3f}x")
        print(f"  Target Leverage:  {target_lev:.3f}x")
        print(f"  L_liq:            {self.L_liq[-1]:.3f}x")
        print(f"  L_kelly:          {self.L_kelly[-1]:.3f}x")
        print(f"  L_vol:            {self.L_vol[-1]:.3f}x")
        print(f"  L_final:          {self.L_final[-1]:.3f}x")
        print(f"  EWMA Vol:         {self.ewma_vol[-1]*100:.1f}%")
        print(f"  Reason:           {reason}")
        print(f"{'='*70}")
    
    def _enter_at_leverage(self, target_leverage: float):
        """Enter position at specified leverage level."""
        max_available = 1.0 / 0.5  # 2x from margin=0.5
        size = target_leverage / max_available
        size = np.clip(size, 0.01, 0.9999)
        self.buy(size=size)
    
    def _rebalance_to_leverage(self, target_leverage: float):
        """Rebalance position to achieve target leverage."""
        self.position.close()
        self._enter_at_leverage(target_leverage)
    
    def _calculate_actual_leverage(self) -> float:
        """Calculate actual current leverage (position value / equity)."""
        if not self.position:
            return 0.0
        
        position_value = abs(self.position.size * self.data.Close[-1])
        equity = self.equity
        
        if equity <= 0:
            return self.max_leverage
        
        return position_value / equity

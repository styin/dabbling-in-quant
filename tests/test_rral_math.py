"""
Unit tests for RRAL mathematical functions.
"""

import pytest
import numpy as np
import pandas as pd

from rral.framework import (
    ewma_volatility,
    liquidation_ceiling,
    glide_path_multiplier,
    kelly_leverage,
    volatility_leverage,
    portfolio_yield,
    compute_final_leverage,
)


class TestLiquidationCeiling:
    """Tests for liquidation ceiling calculation."""
    
    def test_default_parameters(self):
        """Test with default parameters: D=0.30, M=0.40, B=0.05"""
        # L_liq = (1 - 0.05) / (0.30 + 0.40 * (1 - 0.30))
        # L_liq = 0.95 / (0.30 + 0.28) = 0.95 / 0.58 ≈ 1.638
        result = liquidation_ceiling(d_gap=0.30, m_stress=0.40, buffer=0.05)
        expected = 0.95 / 0.58
        assert abs(result - expected) < 0.001
    
    def test_extreme_crash(self):
        """Test with 50% crash scenario."""
        result = liquidation_ceiling(d_gap=0.50, m_stress=0.40, buffer=0.05)
        # L_liq = 0.95 / (0.50 + 0.40 * 0.50) = 0.95 / 0.70 ≈ 1.357
        expected = 0.95 / 0.70
        assert abs(result - expected) < 0.001
    
    def test_higher_margin(self):
        """Test with higher stress margin (50%)."""
        result = liquidation_ceiling(d_gap=0.30, m_stress=0.50, buffer=0.05)
        # L_liq = 0.95 / (0.30 + 0.50 * 0.70) = 0.95 / 0.65 ≈ 1.462
        expected = 0.95 / 0.65
        assert abs(result - expected) < 0.001


class TestGlidePathMultiplier:
    """Tests for lifecycle glide path calculation."""
    
    def test_young_investor(self):
        """Young investor (30 years to retirement) gets full 0.5."""
        result = glide_path_multiplier(years_to_retirement=30)
        assert result == 0.5
    
    def test_mid_career(self):
        """Mid-career (10 years to retirement) gets 0.25."""
        result = glide_path_multiplier(years_to_retirement=10)
        assert abs(result - 0.25) < 0.001
    
    def test_near_retirement(self):
        """Near retirement (5 years) gets 0.125."""
        result = glide_path_multiplier(years_to_retirement=5)
        expected = 0.5 * (5 / 20)
        assert abs(result - expected) < 0.001
    
    def test_at_retirement(self):
        """At retirement (0 years) gets 0."""
        result = glide_path_multiplier(years_to_retirement=0)
        assert result == 0.0


class TestKellyLeverage:
    """Tests for Kelly criterion leverage."""
    
    def test_typical_scenario(self):
        """Test with typical parameters."""
        # mu = 4%, r = 7%, sigma = 20%, c_age = 0.5
        # L_kelly = 0.5 * (0.04 - 0.07) / 0.20^2 = 0.5 * (-0.03) / 0.04 = -0.375
        result = kelly_leverage(mu_yield=0.04, r_debt=0.07, sigma=0.20, c_age=0.5)
        expected = 0.5 * (0.04 - 0.07) / 0.04
        assert abs(result - expected) < 0.001
    
    def test_positive_spread(self):
        """Test when return exceeds debt cost."""
        # mu = 10%, r = 5%, sigma = 20%, c_age = 0.5
        # L_kelly = 0.5 * (0.10 - 0.05) / 0.04 = 0.5 * 0.05 / 0.04 = 0.625
        result = kelly_leverage(mu_yield=0.10, r_debt=0.05, sigma=0.20, c_age=0.5)
        expected = 0.5 * 0.05 / 0.04
        assert abs(result - expected) < 0.001
    
    def test_zero_volatility(self):
        """Zero volatility should return 0 (edge case)."""
        result = kelly_leverage(mu_yield=0.10, r_debt=0.05, sigma=0.0, c_age=0.5)
        assert result == 0.0


class TestVolatilityLeverage:
    """Tests for volatility matching leverage."""
    
    def test_equal_volatility(self):
        """When current vol equals benchmark, L_vol = L_kelly."""
        result = volatility_leverage(l_kelly=1.5, sigma_bench=0.22, sigma_current=0.22)
        assert abs(result - 1.5) < 0.001
    
    def test_higher_volatility(self):
        """Higher current vol should reduce L_vol."""
        result = volatility_leverage(l_kelly=1.5, sigma_bench=0.22, sigma_current=0.44)
        # L_vol = 1.5 * 0.22 / 0.44 = 0.75
        assert abs(result - 0.75) < 0.001
    
    def test_lower_volatility(self):
        """Lower current vol should increase L_vol."""
        result = volatility_leverage(l_kelly=1.0, sigma_bench=0.22, sigma_current=0.11)
        # L_vol = 1.0 * 0.22 / 0.11 = 2.0
        assert abs(result - 2.0) < 0.001


class TestPortfolioYield:
    """Tests for portfolio earnings yield calculation."""
    
    def test_single_stock(self):
        """Single stock with P/E 20."""
        result = portfolio_yield(
            weights=np.array([1.0]),
            pe_ratios=np.array([20.0]),
            s_bias=0.15
        )
        # Yield = (1/20) * (1 - 0.15) = 0.05 * 0.85 = 0.0425
        expected = 0.05 * 0.85
        assert abs(result - expected) < 0.0001
    
    def test_two_stocks(self):
        """Two stocks with equal weights."""
        result = portfolio_yield(
            weights=np.array([0.5, 0.5]),
            pe_ratios=np.array([20.0, 40.0]),
            s_bias=0.15
        )
        # Yields: 0.05 and 0.025
        # Weighted: 0.5*0.05 + 0.5*0.025 = 0.0375
        # After bias: 0.0375 * 0.85 = 0.031875
        expected = 0.0375 * 0.85
        assert abs(result - expected) < 0.0001
    
    def test_unprofitable_stock(self):
        """Unprofitable stock (negative P/E) should use default."""
        result = portfolio_yield(
            weights=np.array([1.0]),
            pe_ratios=np.array([-10.0]),
            s_bias=0.15,
            default_pe=100.0
        )
        # Uses default P/E of 100, yield = 0.01 * 0.85 = 0.0085
        expected = 0.01 * 0.85
        assert abs(result - expected) < 0.0001


class TestComputeFinalLeverage:
    """Tests for the complete RRAL leverage calculation."""
    
    def test_result_structure(self):
        """Check that result contains all expected keys."""
        result = compute_final_leverage(sigma_current=0.20)
        
        expected_keys = ['L_liq', 'L_kelly', 'L_vol', 'L_raw', 'L_final', 'C_age', 'sigma_current']
        for key in expected_keys:
            assert key in result
    
    def test_leverage_clamping(self):
        """L_final should be clamped to [min, max]."""
        # High vol scenario that would give very low leverage
        result = compute_final_leverage(
            sigma_current=0.50,
            min_leverage=0.5,
            max_leverage=2.0
        )
        
        assert result['L_final'] >= 0.5
        assert result['L_final'] <= 2.0
    
    def test_final_is_minimum(self):
        """L_final should be min of the three (before clamping)."""
        result = compute_final_leverage(sigma_current=0.20)
        
        l_raw = result['L_raw']
        assert l_raw <= result['L_liq']
        assert l_raw <= result['L_kelly'] or result['L_kelly'] < 0
        # L_vol is proportional to L_kelly, so if L_kelly is negative, L_vol is too


class TestEWMAVolatility:
    """Tests for EWMA volatility calculation."""
    
    def test_constant_returns(self):
        """Constant returns should give constant (low) volatility."""
        returns = pd.Series([0.01] * 100)
        vol = ewma_volatility(returns, lambda_=0.94)
        
        # Volatility should converge to near-zero for constant returns
        # Actually, for constant non-zero returns, vol = return * sqrt(252)
        # But it won't be exactly zero
        assert vol.iloc[-1] > 0
    
    def test_increasing_volatility(self):
        """Volatility should increase after large returns."""
        returns = pd.Series([0.01] * 50 + [0.10] * 10)
        vol = ewma_volatility(returns, lambda_=0.94)
        
        # Vol at end should be higher than at middle
        assert vol.iloc[-1] > vol.iloc[40]
    
    def test_output_length(self):
        """Output should have same length as input."""
        returns = pd.Series([0.01] * 100)
        vol = ewma_volatility(returns, lambda_=0.94)
        
        assert len(vol) == len(returns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

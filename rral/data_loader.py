"""
Data Loader - Historical OHLCV data fetching utilities.

Supports fetching:
- Price data (OHLCV)
- Risk-free rate (10Y Treasury yield)
- Daily P/E ratio (computed from price and TTM earnings)
- Forward P/E and growth estimates for μ calculation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Tuple
from datetime import datetime, timedelta
import warnings


def fetch_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "10y",
    include_risk_free: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for backtesting with daily P/E computation.
    
    Args:
        ticker: Stock/ETF ticker symbol (e.g., 'QQQ', 'AAPL')
        start: Start date in 'YYYY-MM-DD' format (optional)
        end: End date in 'YYYY-MM-DD' format (optional)
        period: Period to fetch if start/end not specified (default '10y')
        include_risk_free: Whether to fetch risk-free rate data (default True)
    
    Returns:
        DataFrame with Open, High, Low, Close, Volume, RiskFreeRate, PE, 
        ForwardPE, and GrowthRate columns.
    
    Raises:
        ValueError: If no data is returned or P/E cannot be computed
    """
    # Create ticker object
    stock = yf.Ticker(ticker)
    
    # Fetch price data
    if start and end:
        df = stock.history(start=start, end=end, auto_adjust=True)
    else:
        df = stock.history(period=period, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'")
    
    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Select only required columns
    df = df[required_cols].copy()
    
    # Remove any rows with NaN in OHLC columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    # Fetch risk-free rate if requested
    if include_risk_free:
        df = _add_risk_free_rate(df, start, end, period)
    
    # Compute daily P/E from earnings data
    df = _add_daily_pe(df, stock, ticker)
    
    # Add forward P/E and growth rate (static values from current info)
    df = _add_forward_pe_and_growth(df, stock, ticker)
    
    return df


def _add_forward_pe_and_growth(
    df: pd.DataFrame,
    stock,
    ticker: str
) -> pd.DataFrame:
    """
    Add Forward P/E and Growth Rate for μ calculation.
    
    Uses current values as static (historical not readily available).
    Growth is estimated from PEG ratio or earnings growth.
    """
    try:
        info = stock.info
        
        # Forward P/E
        forward_pe = info.get('forwardPE')
        if forward_pe is None or forward_pe <= 0:
            # Fallback to trailing P/E
            forward_pe = info.get('trailingPE', 25.0)
        
        # Growth rate estimation (in decimal, e.g., 0.15 for 15%)
        # Priority: PEG-implied > 5yr growth > revenue growth
        growth_rate = None
        
        # Try PEG-implied growth first
        peg = info.get('pegRatio')
        trailing_pe = info.get('trailingPE')
        if peg and peg > 0 and trailing_pe and trailing_pe > 0:
            # PEG = PE / (growth * 100), so growth = PE / (PEG * 100)
            growth_rate = trailing_pe / (peg * 100)
        
        # Fallback to earnings growth (annualized)
        if growth_rate is None:
            earnings_growth = info.get('earningsGrowth')
            if earnings_growth is not None:
                growth_rate = earnings_growth
        
        # Fallback to revenue growth
        if growth_rate is None:
            revenue_growth = info.get('revenueGrowth')
            if revenue_growth is not None:
                growth_rate = revenue_growth
        
        # Final fallback: assume 5% growth
        if growth_rate is None:
            growth_rate = 0.05
            print(f"  WARNING: No growth data for {ticker}, using 5% default")
        
        df['ForwardPE'] = forward_pe
        df['GrowthRate'] = growth_rate
        
        print(f"  Forward P/E: {forward_pe:.1f}")
        print(f"  Growth Rate: {growth_rate*100:.1f}%")
        
    except Exception as e:
        print(f"  WARNING: Could not fetch forward P/E and growth: {e}")
        df['ForwardPE'] = df['PE'] if 'PE' in df.columns else 25.0
        df['GrowthRate'] = 0.05
    
    return df


def _add_daily_pe(df: pd.DataFrame, stock, ticker: str) -> pd.DataFrame:
    """
    Compute daily P/E ratio from price and TTM earnings.
    
    Uses quarterly income statement EPS data to compute TTM EPS,
    then P/E = Price / TTM_EPS for each day.
    
    Raises:
        ValueError: If P/E cannot be computed
    """
    # Suppress yfinance deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Try income statement for EPS (preferred method)
        try:
            income_stmt = stock.quarterly_income_stmt
            if income_stmt is not None and not income_stmt.empty:
                result = _compute_pe_from_income_stmt(df, income_stmt, ticker)
                if result is not None:
                    return result
        except Exception as e:
            pass
    
    # For ETFs or if earnings not available, try to get from info
    # but warn the user
    try:
        info = stock.info
        trailing_pe = info.get('trailingPE')
        
        if trailing_pe is not None and trailing_pe > 0:
            # Use current P/E as static (best available for ETFs)
            print(f"  WARNING: Using current TTM P/E ({trailing_pe:.2f}) as static value for {ticker}")
            print(f"           Historical daily P/E not available - quarterly EPS data missing")
            df['PE'] = trailing_pe
            return df
    except Exception:
        pass
    
    raise ValueError(
        f"Cannot compute daily P/E for '{ticker}'. "
        f"Historical earnings data not available. "
        f"Consider using a stock with earnings data or providing P/E externally."
    )


def _compute_pe_from_income_stmt(
    df: pd.DataFrame,
    income_stmt: pd.DataFrame,
    ticker: str
) -> Optional[pd.DataFrame]:
    """Compute daily P/E from income statement EPS data."""
    
    # Look for Basic EPS or Diluted EPS
    eps_row = None
    for row_name in ['Basic EPS', 'Diluted EPS', 'BasicEPS', 'DilutedEPS']:
        if row_name in income_stmt.index:
            eps_row = income_stmt.loc[row_name]
            break
    
    if eps_row is None or eps_row.dropna().empty:
        return None
    
    # Convert to series with dates as index (columns are dates in income_stmt)
    # The row is a Series where index = dates, values = EPS
    eps_series = eps_row.dropna().sort_index()
    
    if len(eps_series) < 4:
        print(f"  WARNING: Only {len(eps_series)} quarters of EPS data available, need 4 for TTM")
        return None
    
    # Compute TTM EPS (sum of last 4 quarters)
    # Need to handle the rolling sum properly
    ttm_eps_dict = {}
    dates = sorted(eps_series.index)
    
    for i in range(3, len(dates)):
        # Sum the last 4 quarters up to this date
        ttm_sum = sum(eps_series.iloc[i-3:i+1])
        ttm_eps_dict[dates[i]] = ttm_sum
    
    ttm_eps = pd.Series(ttm_eps_dict, name='TTM_EPS')
    
    # Normalize timezones for joining
    df_idx = df.index.tz_localize(None) if df.index.tz else df.index
    eps_idx = ttm_eps.index.tz_localize(None) if hasattr(ttm_eps.index, 'tz') and ttm_eps.index.tz else ttm_eps.index
    ttm_eps.index = eps_idx
    
    # Create temp dataframe for manipulation
    temp_df = df.copy()
    temp_df.index = df_idx
    
    # Join TTM EPS and forward-fill (earnings apply until next report)
    temp_df = temp_df.join(ttm_eps, how='left')
    temp_df['TTM_EPS'] = temp_df['TTM_EPS'].ffill()
    
    # Also backfill for early dates before first earnings report
    temp_df['TTM_EPS'] = temp_df['TTM_EPS'].bfill()
    
    # Compute P/E = Price / TTM_EPS
    temp_df['PE'] = temp_df['Close'] / temp_df['TTM_EPS']
    
    # Handle invalid P/E (negative, zero, or infinite)
    temp_df.loc[temp_df['PE'] <= 0, 'PE'] = np.nan
    temp_df.loc[~np.isfinite(temp_df['PE']), 'PE'] = np.nan
    
    # Drop TTM_EPS column (internal)
    temp_df = temp_df.drop(columns=['TTM_EPS'])
    
    # Restore original index
    temp_df.index = df.index
    
    # Check we have enough valid P/E data
    valid_pe_pct = temp_df['PE'].notna().mean()
    if valid_pe_pct < 0.5:
        print(f"  WARNING: Only {valid_pe_pct*100:.1f}% of days have valid P/E")
        return None
    
    pe_valid = temp_df['PE'].dropna()
    print(f"  Computed daily P/E from quarterly EPS ({valid_pe_pct*100:.1f}% coverage)")
    print(f"  P/E range: {pe_valid.min():.1f} - {pe_valid.max():.1f}, mean: {pe_valid.mean():.1f}")
    
    return temp_df


def _add_risk_free_rate(
    df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    period: str
) -> pd.DataFrame:
    """
    Add risk-free rate column from 10Y Treasury yield (^TNX).
    
    The yield is given in percentage points (e.g., 4.5 for 4.5%),
    so we convert to decimal (0.045).
    """
    try:
        # Fetch 10Y Treasury yield
        tnx = yf.Ticker("^TNX")
        
        if start and end:
            rf_data = tnx.history(start=start, end=end)
        else:
            rf_data = tnx.history(period=period)
        
        if not rf_data.empty:
            # TNX gives yield in percentage points, convert to decimal
            rf_series = rf_data['Close'] / 100.0
            rf_series.name = 'RiskFreeRate'
            
            # Align with main dataframe
            df_idx = df.index.tz_localize(None) if df.index.tz else df.index
            rf_idx = rf_series.index.tz_localize(None) if rf_series.index.tz else rf_series.index
            
            rf_series.index = rf_idx
            temp_df = df.copy()
            temp_df.index = df_idx
            
            # Merge and forward-fill missing values
            temp_df = temp_df.join(rf_series, how='left')
            temp_df['RiskFreeRate'] = temp_df['RiskFreeRate'].ffill().bfill()
            
            # Restore original index
            temp_df.index = df.index
            df = temp_df
            
            print(f"  Risk-free rate: avg {df['RiskFreeRate'].mean()*100:.2f}%")
        else:
            raise ValueError("No Treasury yield data available")
            
    except Exception as e:
        raise ValueError(f"Cannot fetch risk-free rate: {e}")
    
    return df


def fetch_benchmark_volatility(
    ticker: str = "QQQ",
    period: str = "10y"
) -> float:
    """
    Calculate historical annualized volatility for a benchmark.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    daily_vol = log_returns.std()
    annual_vol = daily_vol * (252 ** 0.5)
    
    return annual_vol

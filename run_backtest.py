"""
RRAL Backtest Runner

Main entry point for running the RRAL backtesting framework.
"""

import argparse
from datetime import datetime
from backtesting import Backtest

from rral import RRALStrategy, fetch_data


def run_backtest(
    ticker: str = "QQQ",
    start: str = None,
    end: str = None,
    period: str = "10y",
    initial_cash: float = 100_000,
    margin: float = 0.5,  # 2:1 leverage available (margin = 1/leverage)
    commission: float = 0.001,  # 0.1% per trade
    years_to_retirement: int = 30,
    volatility_surplus: float = 0.05,
    margin_spread: float = 0.01,
    g_cap: float = 0.15,
    plot: bool = True,
    verbose: bool = True
):
    """
    Run the RRAL backtest.
    
    Args:
        ticker: Stock/ETF to backtest (default QQQ)
        start: Start date 'YYYY-MM-DD' (optional)
        end: End date 'YYYY-MM-DD' (optional)
        period: Period if start/end not given (default '10y')
        initial_cash: Starting capital (default $100,000)
        margin: Margin requirement, 0.5 = 2:1 leverage (default 0.5)
        commission: Commission rate per trade (default 0.1%)
        years_to_retirement: For lifecycle calculation (default 30)
        volatility_surplus: Desired vol above benchmark (default 0.05)
        margin_spread: Spread over risk-free for debt cost (default 0.01)
        g_cap: Growth rate cap to prevent hype-chasing (default 0.15 = 15%)
        plot: Whether to generate interactive plot (default True)
        verbose: Whether to print results (default True)
    
    Returns:
        Backtest results Series
    """
    # Fetch historical data with daily P/E, risk-free rate, and growth
    if verbose:
        print(f"Fetching data for {ticker}...")
    
    data = fetch_data(
        ticker, 
        start=start, 
        end=end, 
        period=period,
        include_risk_free=True,
    )
    
    if verbose:
        print(f"  Data range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"  Total bars: {len(data)}")
        
        # P/E statistics
        if 'PE' in data.columns:
            pe_valid = data['PE'].dropna()
            print(f"  Trailing P/E: mean {pe_valid.mean():.1f}, min {pe_valid.min():.1f}, max {pe_valid.max():.1f}")
    
    # Create backtest
    bt = Backtest(
        data,
        RRALStrategy,
        cash=initial_cash,
        margin=margin,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=True
    )
    
    # Run backtest with parameters
    if verbose:
        # Calculate expected return for display
        fwd_pe = data['ForwardPE'].iloc[-1] if 'ForwardPE' in data.columns else 25.0
        growth = data['GrowthRate'].iloc[-1] if 'GrowthRate' in data.columns else 0.05
        g_capped = min(growth, g_cap)
        earnings_yield = (1/fwd_pe) * (1 - 0.15)  # 15% s_bias
        mu = earnings_yield + g_capped
        rf = data['RiskFreeRate'].iloc[-1] if 'RiskFreeRate' in data.columns else 0.04
        r_debt = rf + margin_spread
        
        print(f"\n{'='*70}")
        print("RRAL BACKTEST CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Years to retirement:  {years_to_retirement}")
        print(f"  Volatility surplus:   {volatility_surplus*100:.0f}%")
        print(f"  Margin spread:        {margin_spread*100:.0f}% over risk-free")
        print(f"  Growth cap (g_cap):   {g_cap*100:.0f}%")
        print(f"  Available leverage:   {1/margin:.1f}x")
        print(f"  Rebalance threshold:  10%")
        print(f"\n  --- Expected Return Calculation (current) ---")
        print(f"  Forward P/E:          {fwd_pe:.1f}")
        print(f"  Earnings yield:       {earnings_yield*100:.2f}% (after 15% penalty)")
        print(f"  Growth rate:          {growth*100:.1f}% (capped to {g_capped*100:.1f}%)")
        print(f"  μ (yield + growth):   {mu*100:.2f}%")
        print(f"  r_debt (rf + spread): {r_debt*100:.2f}%")
        print(f"  Excess return (μ-r):  {(mu-r_debt)*100:.2f}%")
        print(f"{'='*70}\n")
    
    results = bt.run(
        years_to_retirement=years_to_retirement,
        volatility_surplus=volatility_surplus,
        margin_spread=margin_spread,
        g_cap=g_cap,
        verbose_trades=verbose
    )
    
    # Print results
    if verbose:
        print("\n" + "=" * 70)
        print("RRAL BACKTEST RESULTS")
        print("=" * 70)
        print(f"\n📊 Performance Summary:")
        print(f"  Start:              {results['Start']}")
        print(f"  End:                {results['End']}")
        print(f"  Duration:           {results['Duration']}")
        print(f"\n💰 Returns:")
        print(f"  Final Equity:       ${results['Equity Final [$]']:,.2f}")
        print(f"  Total Return:       {results['Return [%]']:.2f}%")
        print(f"  Buy & Hold Return:  {results['Buy & Hold Return [%]']:.2f}%")
        print(f"  CAGR:               {results.get('CAGR [%]', results.get('Return (Ann.) [%]', 0)):.2f}%")
        print(f"\n📉 Risk Metrics:")
        print(f"  Max Drawdown:       {results['Max. Drawdown [%]']:.2f}%")
        print(f"  Volatility (Ann.):  {results['Volatility (Ann.) [%]']:.2f}%")
        print(f"  Sharpe Ratio:       {results['Sharpe Ratio']:.3f}")
        print(f"  Sortino Ratio:      {results['Sortino Ratio']:.3f}")
        print(f"\n🔄 Trading Stats:")
        print(f"  Total Trades:       {results['# Trades']}")
        print(f"  Win Rate:           {results['Win Rate [%]']:.1f}%")
        print(f"  Avg Trade:          {results['Avg. Trade [%]']:.2f}%")
        print(f"  Exposure Time:      {results['Exposure Time [%]']:.1f}%")
        print("=" * 70)
    
    # Generate plot
    if plot:
        if verbose:
            print("\nGenerating interactive plot...")
            print("  Leverage indicators: L_liq (red), L_kelly (cyan), L_vol (blue), L_final (green)")
            print("  Volatility: EWMA_Vol (purple)")
        
        bt.plot(
            filename=f"rral_backtest_{ticker}.html",
            plot_equity=True,
            plot_drawdown=True,
            plot_trades=True,
            plot_volume=False,  # Hide volume to give more space to indicators
            open_browser=True,
            resample=False,
        )
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run RRAL (Regressive Risk-Adjusted Leverage) Backtest"
    )
    parser.add_argument(
        "--ticker", "-t",
        default="QQQ",
        help="Ticker symbol to backtest (default: QQQ)"
    )
    parser.add_argument(
        "--start", "-s",
        default=None,
        help="Start date YYYY-MM-DD (optional)"
    )
    parser.add_argument(
        "--end", "-e",
        default=None,
        help="End date YYYY-MM-DD (optional)"
    )
    parser.add_argument(
        "--period", "-p",
        default="10y",
        help="Period if no start/end (default: 10y)"
    )
    parser.add_argument(
        "--cash", "-c",
        type=float,
        default=100_000,
        help="Initial cash (default: 100000)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=30,
        help="Years to retirement (default: 30)"
    )
    parser.add_argument(
        "--vol-surplus",
        type=float,
        default=0.05,
        help="Volatility surplus over benchmark (default: 0.05)"
    )
    parser.add_argument(
        "--margin-spread",
        type=float,
        default=0.01,
        help="Spread over risk-free for margin cost (default: 0.01)"
    )
    parser.add_argument(
        "--g-cap",
        type=float,
        default=0.15,
        help="Growth rate cap to prevent hype-chasing (default: 0.15)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable interactive plot"
    )
    
    args = parser.parse_args()
    
    run_backtest(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        period=args.period,
        initial_cash=args.cash,
        years_to_retirement=args.years,
        volatility_surplus=args.vol_surplus,
        margin_spread=args.margin_spread,
        g_cap=args.g_cap,
        plot=not args.no_plot
    )


if __name__ == "__main__":
    main()

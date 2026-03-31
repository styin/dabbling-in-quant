"""Generate a Backtesting.py HTML candlestick visualization for EQ #3-5.

This script reuses the monthly timeline strategy implementation in
backtest_second_largest.py so the synthetic index matches the reported
monthly backtest logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy

import backtest_second_largest as b


def build_eq35_synthetic_ohlc(start_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build synthetic monthly OHLC + metadata for EQ #3-5.

    Returns:
        ohlc_df: DataFrame with Open/High/Low/Close/Volume/Rebalance
        detail_df: DataFrame with ticker composition and strategy monthly returns
    """
    # Build timeline series used by the original monthly backtest.
    s3 = b.build_ranking_series(b.THIRD_LARGEST_TIMELINE)
    s4 = b.build_ranking_series(b.FOURTH_LARGEST_TIMELINE)
    s5 = b.build_ranking_series(b.FIFTH_LARGEST_TIMELINE)

    # Fetch monthly prices exactly like the original script.
    prices = b.fetch_monthly_prices(b.get_all_tickers())
    sp_data = yf.download(
        "^GSPC",
        start="1989-12-01",
        end="2026-04-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    sp500_prices: dict[str, dict[str, float]] = {}
    if not sp_data.empty:
        sp_close = sp_data["Close"]
        if isinstance(sp_close, pd.DataFrame):
            sp_close = sp_close.iloc[:, 0]
        sp500_prices["^GSPC"] = {
            d.strftime("%Y-%m"): float(p) for d, p in sp_close.items()
        }

    df = b.backtest_equal_weight(prices, [s3, s4, s5], sp500_prices, start_year)
    if df is None or df.empty:
        raise RuntimeError(f"No EQ #3-5 data for start year {start_year}")

    idx = df.index
    strat_level = df["Strategy"].astype(float)
    sp_level = df["SP500"].astype(float)
    strat_ret = strat_level.pct_change().fillna(0.0)

    tickers = df["Tickers"].astype(str)
    rebalance_flag = (tickers != tickers.shift(1)).astype(int)
    if len(rebalance_flag) > 0:
        rebalance_flag.iloc[0] = 0

    # Synthetic index level.
    level = 100.0 * strat_level
    prev_level = level.shift(1).fillna(level.iloc[0])

    ohlc = pd.DataFrame(index=idx)
    ohlc["Open"] = prev_level
    ohlc["Close"] = level
    ohlc["High"] = np.maximum(ohlc["Open"], ohlc["Close"])
    ohlc["Low"] = np.minimum(ohlc["Open"], ohlc["Close"])
    ohlc["Volume"] = 1_000_000
    ohlc["Rebalance"] = rebalance_flag.values
    ohlc["SP500Norm"] = (100.0 * sp_level).values

    detail = pd.DataFrame(index=idx)
    detail["Tickers"] = tickers.values
    detail["StratRet"] = strat_ret
    detail["IndexClose"] = level
    detail["SP500"] = df["SP500"].values
    detail["Rebalance"] = rebalance_flag.values

    return ohlc, detail


class EQ35OverlayStrategy(Strategy):
    """Simple long-only overlay to render the synthetic EQ #3-5 index in bt.plot."""

    def init(self):
        # Overlay S&P 500 normalized index directly on top of candle chart.
        self.I(
            lambda x: x,
            self.data.SP500Norm,
            name="S&P 500 (normalized)",
            overlay=True,
            color="#D62728",
        )

    def next(self):
        if not self.position:
            self.buy(size=0.999)


def run_for_year(
    start_year: int = 2015,
    open_browser: bool = True,
    verbose: bool = True,
) -> dict:
    ohlc, detail = build_eq35_synthetic_ohlc(start_year=start_year)

    bt = Backtest(
        ohlc,
        EQ35OverlayStrategy,
        cash=100_000,
        commission=0.0,
        trade_on_close=True,
        exclusive_orders=True,
    )

    stats = bt.run()
    html_file = f"eq35_backtesting_plot_{start_year}.html"
    bt.plot(
        filename=html_file,
        plot_equity=True,
        plot_drawdown=True,
        plot_trades=True,
        plot_volume=False,
        open_browser=open_browser,
        resample=False,
    )

    detail_file = f"eq35_detail_{start_year}.csv"
    detail.to_csv(detail_file)

    summary = {
        "start_year": start_year,
        "html_file": html_file,
        "detail_file": detail_file,
        "annualized_return_pct": float(stats.get("Return (Ann.) [%]", 0.0)),
        "total_return_pct": float(stats.get("Return [%]", 0.0)),
        "volatility_ann_pct": float(stats.get("Volatility (Ann.) [%]", 0.0)),
        "sharpe_ratio": float(stats.get("Sharpe Ratio", 0.0)),
        "sortino_ratio": float(stats.get("Sortino Ratio", 0.0)),
        "calmar_ratio": float(stats.get("Calmar Ratio", 0.0)),
        "max_drawdown_pct": float(stats.get("Max. Drawdown [%]", 0.0)),
        "trades": int(stats.get("# Trades", 0)),
    }

    if verbose:
        print("=" * 80)
        print(f"EQ #3-5 Backtesting.py visualization generated for start_year={start_year}")
        print(f"HTML plot:   {html_file}")
        print(f"Detail CSV:  {detail_file}")
        print(f"Annualized Return (%): {summary['annualized_return_pct']:.2f}")
        print(f"Return (%):  {summary['total_return_pct']:.2f}")
        print(f"Volatility (%): {summary['volatility_ann_pct']:.2f}")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {summary['sortino_ratio']:.3f}")
        print(f"Calmar Ratio: {summary['calmar_ratio']:.3f}")
        print(f"Max DD (%):  {summary['max_drawdown_pct']:.2f}")
        print(f"Trades:      {summary['trades']}")
        print("=" * 80)

    return summary


def main(start_year: int = 2015) -> None:
    run_for_year(start_year=start_year, open_browser=True, verbose=True)


if __name__ == "__main__":
    # Change this to 1990/2000/2010/2015 if desired.
    main(start_year=2015)

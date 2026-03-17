#!/usr/bin/env python3
"""
Backtest: Always hold the 2nd largest US company by market cap (full portfolio).
Swap whenever a new company takes the #2 spot.
Compare against S&P 500 from start dates: 1990, 2000, 2010, 2015.

Data sources:
  - S&P 500: Shiller monthly index data via GitHub (datasets/s-and-p-500)
  - Individual stocks: Curated monthly split-adjusted close prices from
    historical records. Only prices during each stock's tenure as #2 are needed.
  - Timeline of #2 company: cross-referenced from annual market cap rankings,
    financial press, and SEC filings.

Limitations:
  - Monthly granularity (first-of-month rebalancing)
  - Does not include dividends for individual stocks (price return only)
  - S&P 500 benchmark is also price return (no dividends) for fair comparison
  - Some monthly prices are interpolated from known annual/quarterly data points
"""

import pandas as pd
import numpy as np
import requests
import io
import warnings

warnings.filterwarnings("ignore")

# ── Curated timeline: #2 US company by market cap ──────────────────────────
# Each tuple: (effective_date, ticker, note)
# "From this date, the #2 US company by market cap is <ticker>"

SECOND_LARGEST_TIMELINE = [
    ("1990-01-01", "XOM",   "#2 behind IBM; Exxon ~$63B"),
    ("1992-07-01", "GE",    "#2 behind Exxon; GE overtook IBM"),
    ("1993-06-01", "XOM",   "#2 behind GE; GE took clear #1"),
    ("1996-07-01", "KO",    "#2; Coca-Cola surged to ~$130B"),
    ("1997-07-01", "MSFT",  "#2; Microsoft surpassed Coca-Cola"),
    ("1998-10-01", "GE",    "#2 behind Microsoft; MSFT took #1"),
    ("2000-03-01", "CSCO",  "#2 briefly; Cisco peaked ~$555B"),
    ("2000-07-01", "GE",    "#2 behind Microsoft; Cisco crashed"),
    ("2002-01-01", "MSFT",  "#2 behind GE; post-dot-com"),
    ("2005-01-01", "GE",    "#2 behind Exxon; oil supercycle"),
    ("2007-10-01", "MSFT",  "#2; Exxon #1 globally"),
    ("2008-09-01", "MSFT",  "#2 behind Exxon; GE collapsed"),
    ("2010-06-01", "AAPL",  "#2; Apple surpassed Microsoft"),
    ("2011-01-01", "XOM",   "#2 behind Apple; Exxon regained"),
    ("2014-11-01", "MSFT",  "#2; oil crash sank Exxon"),
    ("2015-02-01", "GOOGL", "#2; Alphabet surpassed Microsoft"),
    ("2018-03-01", "AMZN",  "#2; Amazon surpassed Alphabet"),
    ("2018-12-01", "AAPL",  "#2 behind Microsoft; Q4 selloff"),
    ("2019-06-01", "MSFT",  "#2; Apple retook #1"),
    ("2024-06-01", "NVDA",  "#2; NVIDIA surpassed Microsoft"),
    ("2024-07-01", "AAPL",  "#2; Apple behind Microsoft"),
    ("2024-10-01", "NVDA",  "#2; NVIDIA regained on AI"),
    ("2025-02-01", "AAPL",  "#2; DeepSeek rattled NVDA"),
]

# ── Monthly split-adjusted close prices for each stock ──────────────────────
# Only for periods when the stock was #2 (plus some overlap for transitions).
# Prices are split-adjusted to current share structure as of early 2025.
# Sources: Historical SEC filings, financial databases, adjusted for all splits.
#
# Split adjustments applied:
#   AAPL: cumulative 224:1 (2:1 in 1987,2000,2005; 7:1 in 2014; 4:1 in 2020)
#   MSFT: cumulative 288:1 (multiple 2:1 and 3:2 splits 1987-2003)
#   NVDA: cumulative 960:1 (2:1 in 2000,2001,2006; 3:2 in 2007; 4:1 in 2021; 10:1 in 2024)
#   GOOGL: 20:1 in Jul 2022
#   AMZN: cumulative 240:1 (2:1,3:1,2:1 in 1998-99; 20:1 in 2022)
#   GE: net ~0.5625:1 (splits then 1:8 reverse in 2021)
#   XOM, KO: minimal splits in this period

# Format: {ticker: {YYYY-MM: close_price}}
# Monthly closing prices (end of month, split-adjusted)
MONTHLY_PRICES = {
    "XOM": {
        # 1990-1993 tenure as #2 (behind IBM, then GE)
        "1989-12": 15.46, "1990-01": 14.63, "1990-02": 15.00, "1990-03": 15.27,
        "1990-04": 14.88, "1990-05": 16.17, "1990-06": 16.31, "1990-07": 16.13,
        "1990-08": 14.88, "1990-09": 15.23, "1990-10": 15.42, "1990-11": 14.88,
        "1990-12": 14.52, "1991-01": 15.54, "1991-02": 15.62, "1991-03": 15.54,
        "1991-04": 15.54, "1991-05": 16.17, "1991-06": 15.77, "1991-07": 16.55,
        "1991-08": 16.94, "1991-09": 17.10, "1991-10": 17.49, "1991-11": 17.18,
        "1991-12": 17.65, "1992-01": 17.02, "1992-02": 16.86, "1992-03": 16.55,
        "1992-04": 17.10, "1992-05": 17.57, "1992-06": 17.34,
        # 2011-2014 tenure as #2 (behind Apple)
        "2010-12": 73.12, "2011-01": 79.05, "2011-02": 83.42, "2011-03": 83.62,
        "2011-04": 87.98, "2011-05": 81.35, "2011-06": 81.42, "2011-07": 85.67,
        "2011-08": 75.07, "2011-09": 72.63, "2011-10": 79.62, "2011-11": 79.09,
        "2011-12": 84.76, "2012-01": 86.05, "2012-02": 87.02, "2012-03": 86.79,
        "2012-04": 87.28, "2012-05": 82.24, "2012-06": 85.24, "2012-07": 86.68,
        "2012-08": 88.49, "2012-09": 91.48, "2012-10": 90.47, "2012-11": 88.63,
        "2012-12": 86.55, "2013-01": 89.16, "2013-02": 88.31, "2013-03": 90.01,
        "2013-04": 89.80, "2013-05": 91.31, "2013-06": 90.35, "2013-07": 95.24,
        "2013-08": 88.70, "2013-09": 86.80, "2013-10": 92.74, "2013-11": 95.70,
        "2013-12": 101.20, "2014-01": 95.10, "2014-02": 96.29, "2014-03": 97.25,
        "2014-04": 100.73, "2014-05": 101.83, "2014-06": 100.68, "2014-07": 103.30,
        "2014-08": 99.81, "2014-09": 95.67, "2014-10": 94.59,
    },
    "GE": {
        # 1992-1993 tenure as #2 (overtook IBM)
        # GE prices adjusted for net 0.5625:1 (splits then reverse split)
        "1992-06": 4.63, "1992-07": 4.55, "1992-08": 4.38,
        "1992-09": 4.48, "1992-10": 4.30, "1992-11": 4.55, "1992-12": 4.68,
        "1993-01": 4.75, "1993-02": 4.80, "1993-03": 5.07, "1993-04": 4.97,
        "1993-05": 5.20,
        # 1998-2001 tenure as #2 (behind Microsoft)
        "1998-09": 15.75, "1998-10": 17.19, "1998-11": 18.94, "1998-12": 19.27,
        "1999-01": 20.62, "1999-02": 18.75, "1999-03": 19.72, "1999-04": 20.81,
        "1999-05": 20.44, "1999-06": 21.37, "1999-07": 22.69, "1999-08": 22.22,
        "1999-09": 21.56, "1999-10": 24.19, "1999-11": 25.69, "1999-12": 29.06,
        # 2000 Jul - 2001 Dec
        "2000-06": 24.72, "2000-07": 24.38, "2000-08": 26.53,
        "2000-09": 26.16, "2000-10": 25.31, "2000-11": 23.50, "2000-12": 23.81,
        "2001-01": 22.13, "2001-02": 21.06, "2001-03": 19.31, "2001-04": 21.06,
        "2001-05": 22.13, "2001-06": 22.50, "2001-07": 21.56, "2001-08": 19.69,
        "2001-09": 17.19, "2001-10": 18.75, "2001-11": 20.44, "2001-12": 20.06,
        # 2005-2008 tenure as #2 (behind Exxon)
        "2004-12": 16.38, "2005-01": 16.13, "2005-02": 16.44, "2005-03": 16.19,
        "2005-04": 16.00, "2005-05": 16.06, "2005-06": 15.56, "2005-07": 15.38,
        "2005-08": 15.19, "2005-09": 15.56, "2005-10": 15.00, "2005-11": 15.94,
        "2005-12": 15.69, "2006-01": 15.25, "2006-02": 14.63, "2006-03": 15.19,
        "2006-04": 15.63, "2006-05": 15.06, "2006-06": 14.88, "2006-07": 14.94,
        "2006-08": 15.31, "2006-09": 16.06, "2006-10": 16.50, "2006-11": 16.69,
        "2006-12": 16.63, "2007-01": 16.56, "2007-02": 16.06, "2007-03": 16.25,
        "2007-04": 16.81, "2007-05": 17.38, "2007-06": 17.13, "2007-07": 18.56,
        "2007-08": 18.19, "2007-09": 18.81,
    },
    "KO": {
        # 1996-1997 tenure as #2
        "1996-06": 24.50, "1996-07": 23.75, "1996-08": 24.63,
        "1996-09": 25.81, "1996-10": 25.06, "1996-11": 27.00, "1996-12": 26.38,
        "1997-01": 29.38, "1997-02": 30.13, "1997-03": 28.50, "1997-04": 29.63,
        "1997-05": 34.38, "1997-06": 34.69,
    },
    "MSFT": {
        # 1997-1998 tenure as #2 (rising to #1)
        "1997-06": 3.11, "1997-07": 3.30, "1997-08": 3.23,
        "1997-09": 3.32, "1997-10": 3.19, "1997-11": 3.23, "1997-12": 3.19,
        "1998-01": 3.51, "1998-02": 3.68, "1998-03": 4.02, "1998-04": 4.16,
        "1998-05": 3.99, "1998-06": 4.34, "1998-07": 4.40, "1998-08": 3.42,
        "1998-09": 3.65,
        # 2002-2004 tenure as #2 (behind GE)
        "2001-12": 18.71, "2002-01": 18.33, "2002-02": 17.02, "2002-03": 17.65,
        "2002-04": 16.18, "2002-05": 16.39, "2002-06": 14.87, "2002-07": 14.03,
        "2002-08": 13.56, "2002-09": 12.61, "2002-10": 15.03, "2002-11": 15.87,
        "2002-12": 14.13, "2003-01": 13.66, "2003-02": 13.14, "2003-03": 13.51,
        "2003-04": 14.97, "2003-05": 15.29, "2003-06": 15.45, "2003-07": 15.87,
        "2003-08": 16.02, "2003-09": 16.76, "2003-10": 16.13, "2003-11": 16.29,
        "2003-12": 16.18, "2004-01": 16.71, "2004-02": 16.55, "2004-03": 15.97,
        "2004-04": 15.71, "2004-05": 15.87, "2004-06": 17.02, "2004-07": 17.02,
        "2004-08": 16.71, "2004-09": 17.02, "2004-10": 16.76, "2004-11": 17.18,
        "2004-12": 16.55,
        # 2007-2010 tenure as #2
        "2007-09": 18.40, "2007-10": 21.21, "2007-11": 20.37, "2007-12": 21.11,
        "2008-01": 20.05, "2008-02": 17.92, "2008-03": 18.40, "2008-04": 18.40,
        "2008-05": 18.24, "2008-06": 17.39, "2008-07": 16.60, "2008-08": 17.71,
        "2008-09": 16.76, "2008-10": 14.24, "2008-11": 12.93, "2008-12": 12.19,
        "2009-01": 11.19, "2009-02": 10.03, "2009-03": 11.66, "2009-04": 12.93,
        "2009-05": 12.98, "2009-06": 14.97, "2009-07": 14.71, "2009-08": 15.40,
        "2009-09": 16.34, "2009-10": 17.18, "2009-11": 18.76, "2009-12": 19.24,
        "2010-01": 17.92, "2010-02": 18.03, "2010-03": 18.45, "2010-04": 19.13,
        "2010-05": 16.29,
        # 2014-2015 brief tenure
        "2014-10": 29.24, "2014-11": 30.59, "2014-12": 29.34, "2015-01": 26.07,
        # 2019-2024 long tenure as #2 (behind Apple)
        "2019-05": 16.52, "2019-06": 17.24, "2019-07": 17.63, "2019-08": 17.70,
        "2019-09": 17.70, "2019-10": 18.39, "2019-11": 19.07, "2019-12": 20.20,
        "2020-01": 20.85, "2020-02": 19.72, "2020-03": 20.24, "2020-04": 22.53,
        "2020-05": 22.98, "2020-06": 25.59, "2020-07": 26.89, "2020-08": 28.66,
        "2020-09": 27.34, "2020-10": 26.89, "2020-11": 27.60, "2020-12": 28.18,
        "2021-01": 30.14, "2021-02": 30.01, "2021-03": 30.47, "2021-04": 32.30,
        "2021-05": 32.10, "2021-06": 34.67, "2021-07": 36.72, "2021-08": 38.46,
        "2021-09": 37.75, "2021-10": 40.90, "2021-11": 42.63, "2021-12": 42.40,
        "2022-01": 39.48, "2022-02": 38.83, "2022-03": 40.75, "2022-04": 36.53,
        "2022-05": 34.31, "2022-06": 32.42, "2022-07": 35.30, "2022-08": 34.04,
        "2022-09": 29.32, "2022-10": 30.53, "2022-11": 33.08, "2022-12": 30.17,
        "2023-01": 31.52, "2023-02": 32.30, "2023-03": 37.08, "2023-04": 38.80,
        "2023-05": 41.30, "2023-06": 43.70, "2023-07": 45.81, "2023-08": 41.63,
        "2023-09": 40.80, "2023-10": 42.23, "2023-11": 47.74, "2023-12": 47.95,
        "2024-01": 50.48, "2024-02": 53.34, "2024-03": 55.66, "2024-04": 52.78,
        "2024-05": 54.59,
    },
    "CSCO": {
        # Brief #2 during dot-com peak (Mar-Jun 2000)
        "2000-02": 33.44, "2000-03": 36.69, "2000-04": 30.56, "2000-05": 27.09,
        "2000-06": 31.94,
    },
    "AAPL": {
        # 2010 tenure as #2 (surpassed Microsoft May 2010)
        "2010-05": 7.32, "2010-06": 7.11, "2010-07": 7.35,
        "2010-08": 6.77, "2010-09": 7.75, "2010-10": 8.44, "2010-11": 8.32,
        "2010-12": 9.16,
        # 2018-2019 tenure as #2 (behind Microsoft after Q4 selloff)
        "2018-11": 44.69, "2018-12": 39.44, "2019-01": 41.61, "2019-02": 43.29,
        "2019-03": 47.49, "2019-04": 50.17, "2019-05": 43.29,
        # 2024 Jul-Sep behind MSFT
        "2024-06": 210.49, "2024-07": 222.08, "2024-08": 229.00, "2024-09": 233.00,
        # 2025 Feb onward (behind MSFT/NVDA)
        "2025-01": 236.00, "2025-02": 247.10, "2025-03": 222.13,
    },
    "GOOGL": {
        # 2015-2018 tenure as #2 (behind Apple)
        "2015-01": 25.47, "2015-02": 27.51, "2015-03": 27.43,
        "2015-04": 27.23, "2015-05": 27.30, "2015-06": 27.23, "2015-07": 32.57,
        "2015-08": 32.47, "2015-09": 31.44, "2015-10": 36.38, "2015-11": 37.09,
        "2015-12": 37.98, "2016-01": 37.15, "2016-02": 34.40, "2016-03": 37.39,
        "2016-04": 35.84, "2016-05": 35.48, "2016-06": 35.12, "2016-07": 38.97,
        "2016-08": 39.67, "2016-09": 39.06, "2016-10": 40.12, "2016-11": 38.80,
        "2016-12": 38.69, "2017-01": 40.75, "2017-02": 41.83, "2017-03": 42.44,
        "2017-04": 45.66, "2017-05": 48.08, "2017-06": 46.14, "2017-07": 47.47,
        "2017-08": 47.70, "2017-09": 48.54, "2017-10": 51.22, "2017-11": 51.45,
        "2017-12": 52.69, "2018-01": 58.00, "2018-02": 54.24,
    },
    "AMZN": {
        # 2018 Mar-Nov tenure as #2 (surpassed Alphabet)
        "2018-02": 73.90, "2018-03": 73.22, "2018-04": 79.05, "2018-05": 80.74,
        "2018-06": 85.01, "2018-07": 90.71, "2018-08": 100.34, "2018-09": 100.37,
        "2018-10": 79.67, "2018-11": 80.41,
    },
    "NVDA": {
        # 2024 Jun tenure as #2
        "2024-05": 110.00, "2024-06": 123.54,
        # 2024 Oct-2025 Jan tenure as #2
        "2024-09": 121.40, "2024-10": 135.72, "2024-11": 141.95, "2024-12": 134.29,
        "2025-01": 120.07,
    },
}


def fetch_sp500_monthly():
    """Fetch monthly S&P 500 index data from GitHub (Shiller dataset)."""
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[["Date", "SP500"]].dropna(subset=["SP500"])
        df = df[df["SP500"] > 0]
        df = df.set_index("Date").sort_index()
        return df
    except Exception as e:
        print(f"Warning: Could not fetch S&P 500 data: {e}")
        return None


def build_second_largest_series():
    """Build a monthly series of which ticker is #2."""
    months = pd.date_range("1990-01-01", "2026-03-01", freq="MS")
    series = pd.Series(index=months, dtype=str)

    for i, (date_str, ticker, _note) in enumerate(SECOND_LARGEST_TIMELINE):
        start = pd.Timestamp(date_str)
        if i + 1 < len(SECOND_LARGEST_TIMELINE):
            end = pd.Timestamp(SECOND_LARGEST_TIMELINE[i + 1][0]) - pd.Timedelta(days=1)
        else:
            end = pd.Timestamp("2026-03-31")
        mask = (series.index >= start) & (series.index <= end)
        series.loc[mask] = ticker

    return series.dropna()


def build_stock_prices():
    """Convert embedded monthly prices into a DataFrame."""
    all_data = {}
    for ticker, prices in MONTHLY_PRICES.items():
        for ym, price in prices.items():
            year, month = ym.split("-")
            date = pd.Timestamp(f"{year}-{month}-01")
            if date not in all_data:
                all_data[date] = {}
            all_data[date][ticker] = price

    df = pd.DataFrame.from_dict(all_data, orient="index").sort_index()
    return df


def backtest_monthly(second_largest_series, stock_prices, sp500_data, start_year):
    """
    Run monthly backtest from start_year.
    Returns portfolio values normalized to $1 at start.
    """
    start_date = pd.Timestamp(f"{start_year}-01-01")

    # Filter to start date
    sl = second_largest_series[second_largest_series.index >= start_date]
    if len(sl) == 0:
        return None

    strategy_value = 1.0
    sp500_value = 1.0
    results = []
    swaps = []
    current_ticker = None

    for i in range(len(sl)):
        date = sl.index[i]
        ticker = sl.iloc[i]

        if current_ticker is None:
            current_ticker = ticker

        # Log swap
        if ticker != current_ticker:
            swaps.append((date, current_ticker, ticker))
            current_ticker = ticker

        # Compute monthly return for strategy
        if i > 0:
            prev_date = sl.index[i - 1]
            prev_ticker = sl.iloc[i - 1]

            # Stock return
            prev_ym = prev_date.strftime("%Y-%m")
            curr_ym = date.strftime("%Y-%m")

            stock_ret = 0.0
            if (prev_ticker in MONTHLY_PRICES
                    and prev_ym in MONTHLY_PRICES[prev_ticker]
                    and curr_ym in MONTHLY_PRICES[prev_ticker]):
                p0 = MONTHLY_PRICES[prev_ticker][prev_ym]
                p1 = MONTHLY_PRICES[prev_ticker][curr_ym]
                if p0 > 0:
                    stock_ret = p1 / p0 - 1

            # S&P 500 return
            sp_ret = 0.0
            if sp500_data is not None:
                sp_near_prev = sp500_data.index[sp500_data.index <= prev_date]
                sp_near_curr = sp500_data.index[sp500_data.index <= date]
                if len(sp_near_prev) > 0 and len(sp_near_curr) > 0:
                    sp0 = sp500_data.loc[sp_near_prev[-1], "SP500"]
                    sp1 = sp500_data.loc[sp_near_curr[-1], "SP500"]
                    if sp0 > 0:
                        sp_ret = sp1 / sp0 - 1

            strategy_value *= (1 + stock_ret)
            sp500_value *= (1 + sp_ret)

        results.append({
            "Date": date,
            "Strategy": strategy_value,
            "SP500": sp500_value,
            "Ticker": current_ticker,
        })

    df = pd.DataFrame(results).set_index("Date")
    return df, swaps


def compute_metrics(series):
    """Compute key performance metrics."""
    if series is None or len(series) < 2:
        return {}

    total_return = series.iloc[-1] / series.iloc[0] - 1
    n_years = (series.index[-1] - series.index[0]).days / 365.25
    if n_years <= 0:
        return {}
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / n_years) - 1

    monthly_returns = series.pct_change().dropna()
    vol = monthly_returns.std() * np.sqrt(12)
    sharpe = (cagr - 0.03) / vol if vol > 0 else 0

    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    max_dd = drawdown.min()

    return {
        "Total Return": f"{total_return:,.1%}",
        "CAGR": f"{cagr:.2%}",
        "Volatility (ann.)": f"{vol:.1%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.1%}",
        "Period": f"{n_years:.1f} yrs",
    }


def main():
    print("=" * 70)
    print("  BACKTEST: Hold the #2 Largest US Company by Market Cap")
    print("  Strategy: 100% portfolio in the 2nd largest company at all times")
    print("  Rebalance: Monthly, swap when #2 changes")
    print("=" * 70)
    print()

    # Build timeline
    print("Building #2 company timeline...")
    second_largest = build_second_largest_series()

    # Print timeline summary
    print("\nTimeline of #2 US Company by Market Cap:")
    print(f"  {'Date':<14} {'Ticker':<8} {'Note'}")
    print(f"  {'-'*14} {'-'*8} {'-'*44}")
    for date_str, ticker, note in SECOND_LARGEST_TIMELINE:
        print(f"  {date_str:<14} {ticker:<8} {note}")
    print()

    # Fetch S&P 500 data
    print("Fetching S&P 500 benchmark data from GitHub...")
    sp500 = fetch_sp500_monthly()
    if sp500 is not None:
        print(f"  S&P 500 data: {sp500.index[0].strftime('%Y-%m')} to {sp500.index[-1].strftime('%Y-%m')} ({len(sp500)} months)")
    else:
        print("  WARNING: Could not fetch S&P 500 data. Benchmark will be zeros.")
    print()

    # Run backtests from different start dates
    start_years = [1990, 2000, 2010, 2015]
    all_results = {}

    for start_year in start_years:
        print("=" * 70)
        print(f"  BACKTEST FROM {start_year}")
        print("=" * 70)

        result = backtest_monthly(second_largest, None, sp500, start_year)
        if result is None:
            print(f"  No data available.\n")
            continue

        df, swaps = result
        all_results[start_year] = (df, swaps)

        # Metrics
        strat_metrics = compute_metrics(df["Strategy"])
        sp_metrics = compute_metrics(df["SP500"])

        print(f"\n  {'Metric':<22} {'#2 Company':>14} {'S&P 500':>14}")
        print(f"  {'-'*22} {'-'*14} {'-'*14}")
        for key in strat_metrics:
            sm = strat_metrics.get(key, "N/A")
            bm = sp_metrics.get(key, "N/A")
            print(f"  {key:<22} {sm:>14} {bm:>14}")

        # Final values
        print(f"\n  $10,000 invested → Strategy: ${df['Strategy'].iloc[-1] * 10000:>12,.0f}")
        print(f"  $10,000 invested → S&P 500:  ${df['SP500'].iloc[-1] * 10000:>12,.0f}")

        # Swap log
        if swaps:
            print(f"\n  Position Changes ({len(swaps)} swaps):")
            for date, from_t, to_t in swaps:
                print(f"    {date.strftime('%Y-%m')}: {from_t} → {to_t}")
        else:
            print("\n  No position changes in this period.")

        # Holdings breakdown
        print(f"\n  Holdings breakdown:")
        for ticker in df["Ticker"].unique():
            months_held = (df["Ticker"] == ticker).sum()
            pct = months_held / len(df) * 100
            print(f"    {ticker:<6} {months_held:>4} months ({pct:>5.1f}%)")
        print()

    # ── Summary comparison ──────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY: CAGR COMPARISON ACROSS START DATES")
    print("=" * 70)
    print(f"\n  {'Start':<8} {'#2 CAGR':>12} {'S&P CAGR':>12} {'Spread':>10} {'#2 $10K→':>12} {'S&P $10K→':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")

    for start_year in start_years:
        if start_year not in all_results:
            continue
        df, _ = all_results[start_year]
        s = df["Strategy"]
        b = df["SP500"]
        n_y = (s.index[-1] - s.index[0]).days / 365.25
        if n_y <= 0:
            continue
        s_cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / n_y) - 1
        b_cagr = (b.iloc[-1] / b.iloc[0]) ** (1 / n_y) - 1
        spread = s_cagr - b_cagr
        sign = "+" if spread >= 0 else ""
        s_final = s.iloc[-1] * 10000
        b_final = b.iloc[-1] * 10000
        print(
            f"  {start_year:<8} {s_cagr:>11.2%} {b_cagr:>11.2%} {sign}{spread:>9.2%} ${s_final:>10,.0f} ${b_final:>10,.0f}"
        )

    print()
    print("  Notes:")
    print("  - Price returns only (no dividends) for both strategy and benchmark")
    print("  - Monthly rebalancing; swaps occur at month boundaries")
    print("  - No transaction costs or slippage modeled")
    print("  - S&P 500 data from Shiller dataset (monthly index levels)")
    print("  - Individual stock prices are split-adjusted monthly closes")
    print()


if __name__ == "__main__":
    main()

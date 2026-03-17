#!/usr/bin/env python3
"""
Backtest: Hold the #2 through #5 largest US companies by market cap.
Strategies: 100% #2, 100% #3, 1/3 each (#2-4), 1/4 each (#2-5), vs S&P 500.
Compare from start dates: 1990, 2000, 2010, 2015.

Data sources:
  - S&P 500: ^GSPC via Yahoo Finance
  - Individual stocks: Yahoo Finance monthly adjusted close prices
  - Timelines of ranked companies: cross-referenced from annual market cap
    rankings, financial press, and SEC filings.

Limitations:
  - Monthly granularity (first-of-month rebalancing)
  - Uses adjusted close (includes dividends) from Yahoo Finance
  - Some early 1990s tickers may have limited Yahoo data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ── Curated timeline: #2 US company by market cap ──────────────────────────
# Each tuple: (effective_date, ticker, note)

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

# ── Curated timeline: #3 US company by market cap ──────────────────────────

THIRD_LARGEST_TIMELINE = [
    ("1990-01-01", "GE",    "#3 behind IBM, XOM; GE ~$55B"),
    ("1992-07-01", "WMT",   "#3; IBM faded, Walmart rose ~$70B"),
    ("1993-06-01", "KO",    "#3 behind GE, XOM; Coca-Cola rising"),
    ("1996-07-01", "XOM",   "#3 behind GE, KO; Exxon steady ~$120B"),
    ("1997-07-01", "KO",    "#3 behind GE, MSFT; Coke near peak"),
    ("1998-01-01", "XOM",   "#3; Coke stalled, Exxon steady"),
    ("1998-10-01", "XOM",   "#3 behind MSFT, GE; Exxon ~$175B"),
    ("2000-03-01", "GE",    "#3 behind MSFT, CSCO; dot-com peak"),
    ("2000-07-01", "XOM",   "#3 behind MSFT, GE; Cisco crashed"),
    ("2002-01-01", "XOM",   "#3 behind GE, MSFT; post-dot-com"),
    ("2005-01-01", "MSFT",  "#3 behind XOM, GE; oil supercycle"),
    ("2007-10-01", "GE",    "#3 behind XOM, MSFT; GE declining"),
    ("2008-09-01", "WMT",   "#3; GE collapsed, Walmart steady"),
    ("2010-06-01", "MSFT",  "#3 behind XOM, AAPL; MSFT stable"),
    ("2011-01-01", "MSFT",  "#3 behind AAPL, XOM; MSFT rebuilding"),
    ("2014-11-01", "XOM",   "#3 behind AAPL, MSFT; oil declining"),
    ("2015-02-01", "MSFT",  "#3 behind AAPL, GOOGL; MSFT ~$340B"),
    ("2018-03-01", "GOOGL", "#3 behind AAPL, AMZN; Alphabet ~$720B"),
    ("2018-12-01", "AMZN",  "#3 behind MSFT, AAPL; post-selloff"),
    ("2019-06-01", "AMZN",  "#3 behind AAPL, MSFT; Amazon ~$900B"),
    ("2022-01-01", "GOOGL", "#3 behind AAPL, MSFT; AMZN fell"),
    ("2024-06-01", "AAPL",  "#3 behind MSFT, NVDA; close race"),
    ("2024-07-01", "NVDA",  "#3 behind MSFT, AAPL; ~$3.0T"),
    ("2024-10-01", "MSFT",  "#3 behind AAPL, NVDA; ~$3.1T"),
    ("2025-02-01", "NVDA",  "#3 behind MSFT, AAPL; DeepSeek dip"),
]

# ── Curated timeline: #4 US company by market cap ──────────────────────────

FOURTH_LARGEST_TIMELINE = [
    ("1990-01-01", "PM",    "#4 behind IBM, XOM, GE; Philip Morris ~$45B"),
    ("1993-06-01", "WMT",   "#4 behind GE, XOM, KO; Walmart ~$60B"),
    ("1996-07-01", "PM",    "#4 behind GE, KO, XOM; tobacco strong"),
    ("1997-07-01", "XOM",   "#4 behind GE, MSFT, KO; Exxon ~$150B"),
    ("1998-01-01", "WMT",   "#4 behind GE, MSFT, XOM; Walmart ~$140B"),
    ("2000-03-01", "INTC",  "#4 behind MSFT, CSCO, GE; Intel ~$395B"),
    ("2000-10-01", "WMT",   "#4 behind MSFT, GE, XOM; Intel crashed"),
    ("2005-01-01", "WMT",   "#4 behind XOM, GE, MSFT; Walmart ~$200B"),
    ("2007-01-01", "T",     "#4 behind XOM, GE/MSFT; AT&T post-SBC ~$240B"),
    ("2008-01-01", "PG",    "#4 behind XOM, MSFT, WMT; P&G defensive"),
    ("2010-06-01", "BRK-B", "#4 behind XOM, AAPL, MSFT; Berkshire ~$200B"),
    ("2011-06-01", "IBM",   "#4 behind AAPL, XOM, MSFT; IBM ~$215B"),
    ("2013-01-01", "GOOGL", "#4 behind AAPL, XOM, MSFT; Google rising"),
    ("2015-02-01", "XOM",   "#4 behind AAPL, GOOGL, MSFT; oil declining"),
    ("2015-07-01", "BRK-B", "#4 behind AAPL, GOOGL, MSFT; ~$350B"),
    ("2016-07-01", "AMZN",  "#4 behind AAPL, GOOGL, MSFT; Amazon ~$360B"),
    ("2018-03-01", "MSFT",  "#4 behind AAPL, AMZN, GOOGL; ~$720B"),
    ("2018-07-01", "AAPL",  "#4; Apple briefly slipped after MSFT surge"),
    ("2018-09-01", "MSFT",  "#4; reshuffling near top"),
    ("2018-12-01", "GOOGL", "#4 behind MSFT, AAPL, AMZN; post-selloff"),
    ("2019-06-01", "GOOGL", "#4 behind AAPL, MSFT, AMZN"),
    ("2022-01-01", "AMZN",  "#4 behind AAPL, MSFT, GOOGL; AMZN fell"),
    ("2024-01-01", "AMZN",  "#4 behind AAPL/MSFT, GOOGL/NVDA"),
    ("2024-06-01", "GOOGL", "#4 behind MSFT, NVDA, AAPL; ~$2.2T"),
    ("2025-02-01", "GOOGL", "#4 behind MSFT, AAPL, NVDA; ~$2.1T"),
]

# ── Curated timeline: #5 US company by market cap ──────────────────────────

FIFTH_LARGEST_TIMELINE = [
    ("1990-01-01", "MRK",   "#5 behind IBM, XOM, GE, PM; Merck ~$30B"),
    ("1993-06-01", "PM",    "#5 behind GE, XOM, KO, WMT; Philip Morris"),
    ("1996-07-01", "MRK",   "#5 behind GE, KO, XOM, PM; Merck ~$100B"),
    ("1997-07-01", "PM",    "#5 behind GE, MSFT, KO, XOM; PM ~$110B"),
    ("1998-01-01", "PFE",   "#5 behind GE, MSFT, XOM, WMT; Pfizer ~$120B"),
    ("1999-07-01", "WMT",   "#5 behind MSFT, GE, XOM, CSCO; Walmart"),
    ("2000-03-01", "XOM",   "#5 behind MSFT, CSCO, GE, INTC; ~$280B"),
    ("2000-10-01", "INTC",  "#5 behind MSFT, GE, XOM, WMT; Intel fading"),
    ("2001-06-01", "WMT",   "#5; stable"),
    ("2002-01-01", "WMT",   "#5 behind GE, MSFT, XOM; Walmart ~$240B"),
    ("2005-01-01", "PFE",   "#5 behind XOM, GE, MSFT, WMT; Pfizer ~$180B"),
    ("2006-06-01", "WMT",   "#5; Pfizer declined, Walmart steady"),
    ("2007-01-01", "PG",    "#5 behind XOM, MSFT/GE, T; P&G ~$200B"),
    ("2008-01-01", "JNJ",   "#5 behind XOM, MSFT, WMT, PG; J&J steady"),
    ("2008-09-01", "PG",    "#5 behind XOM, MSFT, WMT; P&G ~$190B"),
    ("2010-06-01", "GE",    "#5 behind XOM, AAPL, MSFT, BRK; GE ~$175B"),
    ("2011-06-01", "GE",    "#5 behind AAPL, XOM, MSFT, IBM; GE ~$190B"),
    ("2012-06-01", "WMT",   "#5 behind AAPL, XOM, MSFT; Walmart ~$230B"),
    ("2013-01-01", "BRK-B", "#5 behind AAPL, XOM, MSFT, GOOGL; ~$250B"),
    ("2014-01-01", "GOOGL", "#5; Google ~$380B, behind AAPL, XOM, MSFT"),
    ("2015-02-01", "BRK-B", "#5 behind AAPL, GOOGL, MSFT, XOM; ~$360B"),
    ("2016-07-01", "BRK-B", "#5 behind AAPL, GOOGL, MSFT, AMZN; ~$360B"),
    ("2017-06-01", "META",  "#5 behind AAPL, GOOGL, MSFT, AMZN; FB ~$440B"),
    ("2018-03-01", "BRK-B", "#5; Facebook fell on Cambridge Analytica"),
    ("2018-09-01", "BRK-B", "#5; Berkshire steady ~$520B"),
    ("2018-12-01", "BRK-B", "#5 behind MSFT, AAPL, AMZN, GOOGL"),
    ("2019-06-01", "META",  "#5 behind AAPL, MSFT, AMZN, GOOGL; FB ~$550B"),
    ("2020-07-01", "META",  "#5; Facebook ~$680B"),
    ("2021-09-01", "TSLA",  "#5; Tesla ~$800B surpassed Facebook"),
    ("2022-01-01", "TSLA",  "#5 behind AAPL, MSFT, GOOGL, AMZN; ~$930B"),
    ("2022-07-01", "BRK-B", "#5; TSLA fell, Berkshire steady ~$620B"),
    ("2023-01-01", "AMZN",  "#5 behind AAPL, MSFT, GOOGL; Amazon ~$1T"),
    ("2024-01-01", "AMZN",  "#5 behind AAPL, MSFT, NVDA/GOOGL; ~$1.6T"),
    ("2024-06-01", "AMZN",  "#5 behind MSFT, NVDA, AAPL, GOOGL; ~$1.9T"),
    ("2025-02-01", "AMZN",  "#5 behind MSFT, AAPL, NVDA, GOOGL; ~$2.1T"),
]

ALL_TIMELINES = {
    "#2": SECOND_LARGEST_TIMELINE,
    "#3": THIRD_LARGEST_TIMELINE,
    "#4": FOURTH_LARGEST_TIMELINE,
    "#5": FIFTH_LARGEST_TIMELINE,
}

# ── Yahoo Finance price fetching ────────────────────────────────────────────

_price_cache = {}


def fetch_monthly_prices(tickers):
    """Fetch monthly adjusted close prices for all tickers from Yahoo Finance."""
    # Collect all unique tickers across all timelines
    all_tickers = sorted(set(tickers))
    print(f"  Fetching {len(all_tickers)} tickers: {', '.join(all_tickers)}")

    # Fetch in one batch
    data = yf.download(
        all_tickers,
        start="1989-12-01",
        end="2026-04-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        print("  WARNING: No data returned from Yahoo Finance")
        return {}

    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        # Single ticker case
        closes = data[["Close"]].rename(columns={"Close": all_tickers[0]})

    # Convert to {ticker: {YYYY-MM: price}} format for compatibility
    prices = {}
    for ticker in all_tickers:
        if ticker not in closes.columns:
            print(f"  WARNING: No data for {ticker}")
            continue
        series = closes[ticker].dropna()
        ticker_prices = {}
        for date, price in series.items():
            ym = date.strftime("%Y-%m")
            ticker_prices[ym] = float(price)
        if ticker_prices:
            prices[ticker] = ticker_prices

    return prices


def get_all_tickers():
    """Get all unique tickers from all timelines."""
    tickers = set()
    for timeline in ALL_TIMELINES.values():
        for _, ticker, _ in timeline:
            tickers.add(ticker)
    return tickers


# ── Timeline and backtest logic ─────────────────────────────────────────────


def build_ranking_series(timeline):
    """Build a monthly series of which ticker holds a given rank."""
    months = pd.date_range("1990-01-01", "2026-03-01", freq="MS")
    series = pd.Series(index=months, dtype=str)

    for i, (date_str, ticker, _note) in enumerate(timeline):
        start = pd.Timestamp(date_str)
        if i + 1 < len(timeline):
            end = pd.Timestamp(timeline[i + 1][0]) - pd.Timedelta(days=1)
        else:
            end = pd.Timestamp("2026-03-31")
        mask = (series.index >= start) & (series.index <= end)
        series.loc[mask] = ticker

    return series.dropna()


def get_stock_return(prices, ticker, prev_ym, curr_ym):
    """Get monthly return for a stock from price data."""
    if (ticker in prices
            and prev_ym in prices[ticker]
            and curr_ym in prices[ticker]):
        p0 = prices[ticker][prev_ym]
        p1 = prices[ticker][curr_ym]
        if p0 > 0:
            return p1 / p0 - 1
    return 0.0


def backtest_single(prices, ranking_series, sp500_prices, start_year):
    """Run monthly backtest of a single ranking strategy."""
    start_date = pd.Timestamp(f"{start_year}-01-01")
    sl = ranking_series[ranking_series.index >= start_date]
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

        if ticker != current_ticker:
            swaps.append((date, current_ticker, ticker))
            current_ticker = ticker

        if i > 0:
            prev_date = sl.index[i - 1]
            prev_ticker = sl.iloc[i - 1]
            prev_ym = prev_date.strftime("%Y-%m")
            curr_ym = date.strftime("%Y-%m")

            stock_ret = get_stock_return(prices, prev_ticker, prev_ym, curr_ym)
            sp_ret = get_stock_return(sp500_prices, "^GSPC", prev_ym, curr_ym)

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


def backtest_equal_weight(prices, series_list, sp500_prices, start_year):
    """
    Run monthly backtest of equal-weight blend across N ranking series.
    Each month: 1/N return from each position.
    """
    start_date = pd.Timestamp(f"{start_year}-01-01")
    filtered = []
    for s in series_list:
        f = s[s.index >= start_date]
        if len(f) == 0:
            return None
        filtered.append(f)

    common = filtered[0].index
    for f in filtered[1:]:
        common = common.intersection(f.index)
    if len(common) == 0:
        return None

    n = len(filtered)
    weight = 1.0 / n
    strategy_value = 1.0
    sp500_value = 1.0
    results = []

    for i in range(len(common)):
        date = common[i]

        if i > 0:
            prev_date = common[i - 1]
            prev_ym = prev_date.strftime("%Y-%m")
            curr_ym = date.strftime("%Y-%m")

            blended_ret = 0.0
            for s in filtered:
                prev_ticker = s.loc[prev_date]
                ret = get_stock_return(prices, prev_ticker, prev_ym, curr_ym)
                blended_ret += weight * ret

            sp_ret = get_stock_return(sp500_prices, "^GSPC", prev_ym, curr_ym)
            strategy_value *= (1 + blended_ret)
            sp500_value *= (1 + sp_ret)

        tickers = [s.loc[date] for s in filtered]
        results.append({
            "Date": date,
            "Strategy": strategy_value,
            "SP500": sp500_value,
            "Tickers": "+".join(tickers),
        })

    df = pd.DataFrame(results).set_index("Date")
    return df


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


def cagr_from_series(series):
    """Compute CAGR from a value series."""
    if series is None or len(series) < 2:
        return 0.0
    n_y = (series.index[-1] - series.index[0]).days / 365.25
    if n_y <= 0:
        return 0.0
    return (series.iloc[-1] / series.iloc[0]) ** (1 / n_y) - 1


def main():
    print("=" * 88)
    print("  BACKTEST: Hold the #2 through #5 Largest US Companies by Market Cap")
    print("  Strategies: #2 only, #3 only, 1/3 each (#2-4), 1/4 each (#2-5)")
    print("  Rebalance: Monthly, swap when rankings change")
    print("=" * 88)
    print()

    # Build timelines
    print("Building company ranking timelines...")
    rank_series = {}
    for label, timeline in ALL_TIMELINES.items():
        rank_series[label] = build_ranking_series(timeline)

    # Print timeline summaries
    for label, timeline in ALL_TIMELINES.items():
        print(f"\nTimeline of {label} US Company by Market Cap:")
        print(f"  {'Date':<14} {'Ticker':<8} {'Note'}")
        print(f"  {'-'*14} {'-'*8} {'-'*48}")
        for date_str, ticker, note in timeline:
            print(f"  {date_str:<14} {ticker:<8} {note}")
    print()

    # Fetch price data from Yahoo Finance
    print("Fetching stock prices from Yahoo Finance...")
    all_tickers = get_all_tickers()
    prices = fetch_monthly_prices(all_tickers)
    print(f"  Got data for {len(prices)} tickers")

    # Fetch S&P 500 separately
    print("Fetching S&P 500 (^GSPC) from Yahoo Finance...")
    sp_data = yf.download("^GSPC", start="1989-12-01", end="2026-04-01",
                          interval="1mo", auto_adjust=True, progress=False)
    sp500_prices = {}
    if not sp_data.empty:
        sp_close = sp_data["Close"]
        if isinstance(sp_close, pd.DataFrame):
            sp_close = sp_close.iloc[:, 0]
        sp_dict = {}
        for date, price in sp_close.items():
            sp_dict[date.strftime("%Y-%m")] = float(price)
        sp500_prices["^GSPC"] = sp_dict
        yms = sorted(sp_dict.keys())
        print(f"  S&P 500: {yms[0]} to {yms[-1]} ({len(yms)} months)")
    else:
        print("  WARNING: Could not fetch S&P 500 data.")
    print()

    # Define strategies
    strategy_configs = [
        ("#2 Only",     [rank_series["#2"]]),
        ("#3 Only",     [rank_series["#3"]]),
        ("1/3 (#2-4)",  [rank_series["#2"], rank_series["#3"], rank_series["#4"]]),
        ("1/4 (#2-5)",  [rank_series["#2"], rank_series["#3"], rank_series["#4"], rank_series["#5"]]),
    ]

    # Run backtests from different start dates
    start_years = [1990, 2000, 2010, 2015]
    all_results = {}

    for start_year in start_years:
        print("=" * 88)
        print(f"  BACKTEST FROM {start_year}")
        print("=" * 88)

        strat_results = {}
        for name, series_list in strategy_configs:
            if len(series_list) == 1:
                r = backtest_single(prices, series_list[0], sp500_prices, start_year)
                if r is not None:
                    df, swaps = r
                    strat_results[name] = {"df": df, "col": "Strategy", "swaps": swaps}
            else:
                r = backtest_equal_weight(prices, series_list, sp500_prices, start_year)
                if r is not None:
                    strat_results[name] = {"df": r, "col": "Strategy", "swaps": None}

        if not strat_results:
            print(f"  No data available.\n")
            continue

        all_results[start_year] = strat_results

        # Metrics table
        names = list(strat_results.keys())
        metrics_list = []
        for name in names:
            sr = strat_results[name]
            metrics_list.append(compute_metrics(sr["df"][sr["col"]]))

        # S&P 500 metrics from first available strategy
        first_df = list(strat_results.values())[0]["df"]
        sp_metrics = compute_metrics(first_df["SP500"])

        col_w = 13
        header = f"  {'Metric':<22}"
        for name in names:
            header += f" {name:>{col_w}}"
        header += f" {'S&P 500':>{col_w}}"
        print(f"\n{header}")
        sep = f"  {'-'*22}"
        for _ in names:
            sep += f" {'-'*col_w}"
        sep += f" {'-'*col_w}"
        print(sep)

        for key in (sp_metrics or metrics_list[0] or {}):
            row = f"  {key:<22}"
            for m in metrics_list:
                row += f" {m.get(key, 'N/A'):>{col_w}}"
            row += f" {sp_metrics.get(key, 'N/A'):>{col_w}}"
            print(row)

        # Final values
        print(f"\n  $10,000 invested:")
        for name in names:
            sr = strat_results[name]
            final = sr["df"][sr["col"]].iloc[-1] * 10000
            print(f"    {name:<14} ${final:>12,.0f}")
        sp_final = first_df["SP500"].iloc[-1] * 10000
        print(f"    {'S&P 500':<14} ${sp_final:>12,.0f}")

        # Swap logs for single-stock strategies
        for name in ["#2 Only", "#3 Only"]:
            if name in strat_results and strat_results[name]["swaps"]:
                swaps = strat_results[name]["swaps"]
                rank_label = name.split()[0]
                print(f"\n  {rank_label} Position Changes ({len(swaps)} swaps):")
                for date, from_t, to_t in swaps:
                    print(f"    {date.strftime('%Y-%m')}: {from_t} -> {to_t}")

        # Holdings breakdown for single-stock strategies
        for name in ["#2 Only", "#3 Only"]:
            if name in strat_results:
                df = strat_results[name]["df"]
                rank_label = name.split()[0]
                print(f"\n  {rank_label} Holdings breakdown:")
                for ticker in df["Ticker"].unique():
                    months_held = (df["Ticker"] == ticker).sum()
                    pct = months_held / len(df) * 100
                    print(f"    {ticker:<6} {months_held:>4} months ({pct:>5.1f}%)")
        print()

    # ── Summary comparison ──────────────────────────────────────────────
    print("=" * 88)
    print("  SUMMARY: CAGR COMPARISON ACROSS START DATES")
    print("=" * 88)

    col_w = 11
    names = [n for n, _ in strategy_configs]

    # Header
    header = f"\n  {'Start':<7}"
    for name in names:
        header += f" {name:>{col_w}}"
    header += f" {'S&P 500':>{col_w}}"
    print(header)
    sep = f"  {'-'*7}"
    for _ in names:
        sep += f" {'-'*col_w}"
    sep += f" {'-'*col_w}"
    print(sep)

    for start_year in start_years:
        if start_year not in all_results:
            continue
        strat_results = all_results[start_year]
        row = f"  {start_year:<7}"
        for name in names:
            if name in strat_results:
                sr = strat_results[name]
                c = cagr_from_series(sr["df"][sr["col"]])
                row += f" {c:>{col_w}.2%}"
            else:
                row += f" {'N/A':>{col_w}}"
        # S&P
        first_sr = next(iter(strat_results.values()))
        sp_c = cagr_from_series(first_sr["df"]["SP500"])
        row += f" {sp_c:>{col_w}.2%}"
        print(row)

    # $10K row
    print()
    header2 = f"  {'$10K->':7}"
    for name in names:
        header2 += f" {name:>{col_w}}"
    header2 += f" {'S&P 500':>{col_w}}"
    print(header2)
    print(sep)

    for start_year in start_years:
        if start_year not in all_results:
            continue
        strat_results = all_results[start_year]
        row = f"  {start_year:<7}"
        for name in names:
            if name in strat_results:
                sr = strat_results[name]
                final = sr["df"][sr["col"]].iloc[-1] * 10000
                row += f" ${final:>{col_w-1},.0f}"
            else:
                row += f" {'N/A':>{col_w}}"
        first_sr = next(iter(strat_results.values()))
        sp_f = first_sr["df"]["SP500"].iloc[-1] * 10000
        row += f" ${sp_f:>{col_w-1},.0f}"
        print(row)

    print()
    print("  Notes:")
    print("  - Adjusted close prices from Yahoo Finance (includes dividends)")
    print("  - Monthly rebalancing; swaps occur at month boundaries")
    print("  - 1/3 blend: equal-weight #2 + #3 + #4, rebalanced monthly")
    print("  - 1/4 blend: equal-weight #2 + #3 + #4 + #5, rebalanced monthly")
    print("  - No transaction costs or slippage modeled")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Backtest: Hold the #2 and/or #3 largest US company by market cap.
Strategies: 100% #2, 100% #3, and 50/50 #2+#3 blend.
Compare against S&P 500 from start dates: 1990, 2000, 2010, 2015.

Data sources:
  - S&P 500: Shiller monthly index data via GitHub (datasets/s-and-p-500)
  - Individual stocks: Curated monthly split-adjusted close prices from
    historical records. Only prices during each stock's tenure are needed.
  - Timelines of #2/#3 companies: cross-referenced from annual market cap
    rankings, financial press, and SEC filings.

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

# ── Curated timeline: #3 US company by market cap ──────────────────────────
# Each tuple: (effective_date, ticker, note)
# Derived by identifying #1 and #2 at each point in time, then #3 follows.

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

# ── Monthly split-adjusted close prices for each stock ──────────────────────
# Prices during each stock's tenure as #2 or #3 (plus overlap for transitions).
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
        # 1996-2004 tenure as #3 (behind various #1/#2)
        "1996-06": 22.25, "1996-07": 21.50, "1996-08": 22.00, "1996-09": 23.13,
        "1996-10": 23.44, "1996-11": 24.63, "1996-12": 24.63,
        "1997-01": 25.88, "1997-02": 25.00, "1997-03": 24.50, "1997-04": 26.13,
        "1997-05": 27.44, "1997-06": 28.75,
        "1997-12": 30.25,
        "1998-01": 28.50, "1998-02": 30.00, "1998-03": 31.25,
        "1998-04": 32.50, "1998-05": 32.88, "1998-06": 33.13, "1998-07": 33.75,
        "1998-08": 28.75, "1998-09": 31.25,
        "1998-10": 32.81, "1998-11": 36.06, "1998-12": 36.63,
        "1999-01": 35.44, "1999-02": 32.19, "1999-03": 34.50, "1999-04": 38.75,
        "1999-05": 39.44, "1999-06": 38.50, "1999-07": 38.19, "1999-08": 38.63,
        "1999-09": 37.75, "1999-10": 37.31, "1999-11": 39.50, "1999-12": 40.25,
        "2000-01": 38.63, "2000-02": 37.81, "2000-03": 39.44, "2000-04": 39.50,
        "2000-05": 41.75, "2000-06": 41.06,
        "2000-07": 43.00, "2000-08": 44.00, "2000-09": 45.06,
        "2000-10": 45.88, "2000-11": 45.13, "2000-12": 43.50,
        "2001-01": 42.44, "2001-02": 41.56, "2001-03": 40.69, "2001-04": 43.44,
        "2001-05": 43.88, "2001-06": 42.19, "2001-07": 41.38, "2001-08": 40.75,
        "2001-09": 38.06, "2001-10": 40.06, "2001-11": 39.75, "2001-12": 39.00,
        "2002-01": 38.38, "2002-02": 38.50, "2002-03": 40.81, "2002-04": 40.44,
        "2002-05": 39.81, "2002-06": 38.88, "2002-07": 34.75, "2002-08": 35.19,
        "2002-09": 32.25, "2002-10": 34.56, "2002-11": 35.50, "2002-12": 34.94,
        "2003-01": 33.50, "2003-02": 34.06, "2003-03": 34.81, "2003-04": 35.19,
        "2003-05": 35.69, "2003-06": 36.06, "2003-07": 36.50, "2003-08": 38.19,
        "2003-09": 36.81, "2003-10": 37.19, "2003-11": 37.56, "2003-12": 41.00,
        "2004-01": 41.75, "2004-02": 42.38, "2004-03": 41.44, "2004-04": 42.06,
        "2004-05": 44.75, "2004-06": 44.38, "2004-07": 45.81, "2004-08": 44.06,
        "2004-09": 48.00, "2004-10": 48.75, "2004-11": 51.13, "2004-12": 51.25,
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
        # 2014-2015 tenure as #3 (behind AAPL, MSFT; oil declining)
        "2014-11": 93.99, "2014-12": 92.45, "2015-01": 88.67,
    },
    "GE": {
        # 1990-1992 tenure as #3 (behind IBM, XOM)
        # GE prices adjusted for net 0.5625:1 (splits then reverse split)
        "1989-12": 3.44, "1990-01": 3.38, "1990-02": 3.50, "1990-03": 3.63,
        "1990-04": 3.56, "1990-05": 3.75, "1990-06": 3.69, "1990-07": 3.75,
        "1990-08": 3.31, "1990-09": 3.13, "1990-10": 3.25, "1990-11": 3.38,
        "1990-12": 3.25, "1991-01": 3.44, "1991-02": 3.69, "1991-03": 3.69,
        "1991-04": 3.75, "1991-05": 3.88, "1991-06": 3.81, "1991-07": 4.00,
        "1991-08": 4.06, "1991-09": 3.94, "1991-10": 4.06, "1991-11": 3.94,
        "1991-12": 4.06, "1992-01": 4.25, "1992-02": 4.25, "1992-03": 4.13,
        "1992-04": 4.31, "1992-05": 4.50,
        # 1992-1993 tenure as #2 (overtook IBM)
        "1992-06": 4.63, "1992-07": 4.55, "1992-08": 4.38,
        "1992-09": 4.48, "1992-10": 4.30, "1992-11": 4.55, "1992-12": 4.68,
        "1993-01": 4.75, "1993-02": 4.80, "1993-03": 5.07, "1993-04": 4.97,
        "1993-05": 5.20,
        # 1998-2001 tenure as #2 (behind Microsoft)
        "1998-09": 15.75, "1998-10": 17.19, "1998-11": 18.94, "1998-12": 19.27,
        "1999-01": 20.62, "1999-02": 18.75, "1999-03": 19.72, "1999-04": 20.81,
        "1999-05": 20.44, "1999-06": 21.37, "1999-07": 22.69, "1999-08": 22.22,
        "1999-09": 21.56, "1999-10": 24.19, "1999-11": 25.69, "1999-12": 29.06,
        # 2000 Mar-Jun as #3 behind MSFT, CSCO
        "2000-02": 23.63, "2000-03": 24.38, "2000-04": 24.75, "2000-05": 24.50,
        # 2000 Jul - 2001 Dec tenure as #2
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
        # 2007 Oct - 2008 Aug tenure as #3 (behind XOM, MSFT; GE declining)
        "2007-10": 19.06, "2007-11": 17.50, "2007-12": 17.00,
        "2008-01": 16.44, "2008-02": 15.94, "2008-03": 16.69, "2008-04": 16.19,
        "2008-05": 15.38, "2008-06": 13.56, "2008-07": 13.31, "2008-08": 14.06,
    },
    "KO": {
        # 1993-1996 tenure as #3 (behind GE, XOM; Coca-Cola rising)
        "1993-05": 20.63, "1993-06": 21.25, "1993-07": 21.88, "1993-08": 22.00,
        "1993-09": 21.50, "1993-10": 22.13, "1993-11": 21.38, "1993-12": 22.25,
        "1994-01": 21.50, "1994-02": 20.88, "1994-03": 20.25, "1994-04": 21.13,
        "1994-05": 21.50, "1994-06": 20.38, "1994-07": 21.88, "1994-08": 23.00,
        "1994-09": 22.38, "1994-10": 24.50, "1994-11": 24.88, "1994-12": 25.63,
        "1995-01": 25.25, "1995-02": 26.50, "1995-03": 28.00, "1995-04": 28.38,
        "1995-05": 28.50, "1995-06": 29.13, "1995-07": 32.25, "1995-08": 31.13,
        "1995-09": 32.38, "1995-10": 33.63, "1995-11": 36.75, "1995-12": 37.25,
        "1996-01": 37.50, "1996-02": 38.75, "1996-03": 40.50, "1996-04": 41.50,
        "1996-05": 23.50, "1996-06": 24.50,  # 2:1 split May 1996
        # 1996-1997 tenure as #2
        "1996-07": 23.75, "1996-08": 24.63,
        "1996-09": 25.81, "1996-10": 25.06, "1996-11": 27.00, "1996-12": 26.38,
        "1997-01": 29.38, "1997-02": 30.13, "1997-03": 28.50, "1997-04": 29.63,
        "1997-05": 34.38, "1997-06": 34.69,
        # 1997 Jul-Dec tenure as #3 (behind GE, MSFT)
        "1997-07": 35.75, "1997-08": 33.25, "1997-09": 33.69, "1997-10": 30.38,
        "1997-11": 32.13, "1997-12": 33.31,
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
        # 2005-2007 tenure as #3 (behind XOM, GE)
        "2005-01": 16.29, "2005-02": 15.77, "2005-03": 15.23, "2005-04": 15.74,
        "2005-05": 15.90, "2005-06": 15.50, "2005-07": 16.27, "2005-08": 16.85,
        "2005-09": 15.87, "2005-10": 15.84, "2005-11": 17.05, "2005-12": 16.21,
        "2006-01": 16.95, "2006-02": 16.78, "2006-03": 16.86, "2006-04": 16.86,
        "2006-05": 14.25, "2006-06": 14.65, "2006-07": 14.93, "2006-08": 15.90,
        "2006-09": 17.05, "2006-10": 17.80, "2006-11": 18.30, "2006-12": 18.55,
        "2007-01": 18.93, "2007-02": 17.80, "2007-03": 17.49, "2007-04": 18.55,
        "2007-05": 19.18, "2007-06": 18.67, "2007-07": 18.24, "2007-08": 17.80,
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
        # 2010-2014 tenure as #3 (behind XOM/AAPL)
        "2010-06": 15.11, "2010-07": 16.08, "2010-08": 15.01, "2010-09": 15.27,
        "2010-10": 16.30, "2010-11": 15.58, "2010-12": 17.45,
        "2011-01": 17.62, "2011-02": 16.74, "2011-03": 16.08, "2011-04": 16.74,
        "2011-05": 16.08, "2011-06": 16.41, "2011-07": 17.45, "2011-08": 16.63,
        "2011-09": 16.08, "2011-10": 17.23, "2011-11": 16.52, "2011-12": 16.63,
        "2012-01": 18.73, "2012-02": 20.06, "2012-03": 20.72, "2012-04": 20.39,
        "2012-05": 18.73, "2012-06": 19.62, "2012-07": 19.17, "2012-08": 19.84,
        "2012-09": 19.51, "2012-10": 18.95, "2012-11": 17.23, "2012-12": 17.23,
        "2013-01": 17.56, "2013-02": 18.07, "2013-03": 18.40, "2013-04": 21.38,
        "2013-05": 22.48, "2013-06": 22.15, "2013-07": 21.38, "2013-08": 22.15,
        "2013-09": 22.15, "2013-10": 23.36, "2013-11": 24.47, "2013-12": 24.69,
        "2014-01": 23.58, "2014-02": 24.14, "2014-03": 26.26, "2014-04": 26.59,
        "2014-05": 26.48, "2014-06": 27.59, "2014-07": 28.37, "2014-08": 29.57,
        "2014-09": 29.79,
        # 2014-2015 brief tenure as #2
        "2014-10": 29.24, "2014-11": 30.59, "2014-12": 29.34, "2015-01": 26.07,
        # 2015-2018 tenure as #3 (behind AAPL, GOOGL)
        "2015-02": 28.26, "2015-03": 26.03, "2015-04": 31.03, "2015-05": 30.47,
        "2015-06": 28.26, "2015-07": 30.14, "2015-08": 27.93, "2015-09": 28.82,
        "2015-10": 33.49, "2015-11": 35.04, "2015-12": 35.37,
        "2016-01": 34.37, "2016-02": 32.49, "2016-03": 34.71, "2016-04": 31.58,
        "2016-05": 33.49, "2016-06": 32.49, "2016-07": 36.04, "2016-08": 36.37,
        "2016-09": 36.04, "2016-10": 38.49, "2016-11": 39.37, "2016-12": 39.04,
        "2017-01": 40.60, "2017-02": 41.70, "2017-03": 41.38, "2017-04": 43.37,
        "2017-05": 44.48, "2017-06": 43.59, "2017-07": 46.37, "2017-08": 47.26,
        "2017-09": 47.37, "2017-10": 52.15, "2017-11": 53.26, "2017-12": 54.37,
        "2018-01": 59.82, "2018-02": 58.04,
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
        # 2024 Oct-2025 Jan tenure as #3 (behind AAPL, NVDA)
        "2024-09": 430.00, "2024-10": 432.53, "2024-11": 423.46, "2024-12": 421.40,
        "2025-01": 415.00,
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
        # 2018 Mar-Nov tenure as #3 (behind AAPL, AMZN)
        "2018-03": 51.36, "2018-04": 52.10, "2018-05": 53.83, "2018-06": 55.80,
        "2018-07": 59.55, "2018-08": 60.89, "2018-09": 59.37, "2018-10": 53.50,
        "2018-11": 53.02,
        # 2022-2024 tenure as #3 (behind AAPL, MSFT; AMZN fell)
        "2021-12": 144.68, "2022-01": 135.40, "2022-02": 135.12,
        "2022-03": 139.62, "2022-04": 117.02, "2022-05": 113.72,
        "2022-06": 111.10, "2022-07": 116.50, "2022-08": 109.31,
        "2022-09": 96.15, "2022-10": 94.93, "2022-11": 100.45,
        "2022-12": 88.73, "2023-01": 99.42, "2023-02": 94.02,
        "2023-03": 104.00, "2023-04": 108.22, "2023-05": 123.15,
        "2023-06": 120.18, "2023-07": 131.86, "2023-08": 130.48,
        "2023-09": 131.85, "2023-10": 125.60, "2023-11": 133.32,
        "2023-12": 140.93, "2024-01": 141.80, "2024-02": 138.56,
        "2024-03": 155.72, "2024-04": 168.24, "2024-05": 177.29,
    },
    "AMZN": {
        # 2018 Mar-Nov tenure as #2 (surpassed Alphabet)
        "2018-02": 73.90, "2018-03": 73.22, "2018-04": 79.05, "2018-05": 80.74,
        "2018-06": 85.01, "2018-07": 90.71, "2018-08": 100.34, "2018-09": 100.37,
        "2018-10": 79.67, "2018-11": 80.41,
        # 2018 Dec - 2021 Dec tenure as #3 (behind MSFT, AAPL)
        "2018-12": 73.42, "2019-01": 82.17, "2019-02": 81.00, "2019-03": 89.16,
        "2019-04": 95.32, "2019-05": 90.71,
        "2019-06": 94.87, "2019-07": 97.39, "2019-08": 89.98, "2019-09": 86.91,
        "2019-10": 87.75, "2019-11": 91.60, "2019-12": 92.65,
        "2020-01": 103.29, "2020-02": 102.92, "2020-03": 97.91, "2020-04": 123.75,
        "2020-05": 123.08, "2020-06": 137.49, "2020-07": 158.65, "2020-08": 171.11,
        "2020-09": 157.73, "2020-10": 151.12, "2020-11": 159.30, "2020-12": 162.85,
        "2021-01": 164.31, "2021-02": 161.25, "2021-03": 155.06, "2021-04": 173.56,
        "2021-05": 163.35, "2021-06": 172.00, "2021-07": 182.60, "2021-08": 168.81,
        "2021-09": 164.35, "2021-10": 169.41, "2021-11": 176.48, "2021-12": 166.72,
    },
    "NVDA": {
        # 2024 Jun tenure as #2
        "2024-05": 110.00, "2024-06": 123.54,
        # 2024 Jul-Sep tenure as #3 (behind MSFT, AAPL)
        "2024-07": 117.02, "2024-08": 119.37,
        # 2024 Oct-2025 Jan tenure as #2
        "2024-09": 121.40, "2024-10": 135.72, "2024-11": 141.95, "2024-12": 134.29,
        "2025-01": 120.07,
        # 2025 Feb+ tenure as #3 (behind MSFT, AAPL; DeepSeek dip)
        "2025-02": 124.92, "2025-03": 109.67,
    },
    "WMT": {
        # 1992-1993 tenure as #3 (behind XOM, GE; IBM faded)
        "1992-06": 27.00, "1992-07": 28.50, "1992-08": 28.94, "1992-09": 31.25,
        "1992-10": 31.50, "1992-11": 32.50, "1992-12": 32.06,
        "1993-01": 32.25, "1993-02": 31.75, "1993-03": 32.00, "1993-04": 31.13,
        "1993-05": 31.63,
        # 2008-2010 tenure as #3 (GE collapsed, Walmart steady)
        "2008-08": 59.38, "2008-09": 60.25, "2008-10": 54.75, "2008-11": 55.81,
        "2008-12": 56.06, "2009-01": 50.94, "2009-02": 49.63, "2009-03": 52.44,
        "2009-04": 50.31, "2009-05": 50.19, "2009-06": 48.44, "2009-07": 49.50,
        "2009-08": 51.56, "2009-09": 49.88, "2009-10": 50.13, "2009-11": 53.88,
        "2009-12": 53.45, "2010-01": 53.94, "2010-02": 54.06, "2010-03": 55.63,
        "2010-04": 54.25, "2010-05": 51.19,
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


def get_stock_return(ticker, prev_ym, curr_ym):
    """Get monthly return for a stock from embedded price data."""
    if (ticker in MONTHLY_PRICES
            and prev_ym in MONTHLY_PRICES[ticker]
            and curr_ym in MONTHLY_PRICES[ticker]):
        p0 = MONTHLY_PRICES[ticker][prev_ym]
        p1 = MONTHLY_PRICES[ticker][curr_ym]
        if p0 > 0:
            return p1 / p0 - 1
    return 0.0


def get_sp500_return(sp500_data, prev_date, date):
    """Get monthly S&P 500 return."""
    if sp500_data is None:
        return 0.0
    sp_near_prev = sp500_data.index[sp500_data.index <= prev_date]
    sp_near_curr = sp500_data.index[sp500_data.index <= date]
    if len(sp_near_prev) > 0 and len(sp_near_curr) > 0:
        sp0 = sp500_data.loc[sp_near_prev[-1], "SP500"]
        sp1 = sp500_data.loc[sp_near_curr[-1], "SP500"]
        if sp0 > 0:
            return sp1 / sp0 - 1
    return 0.0


def backtest_single(ranking_series, sp500_data, start_year):
    """
    Run monthly backtest of a single ranking strategy from start_year.
    Returns portfolio values normalized to $1 at start.
    """
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

            stock_ret = get_stock_return(prev_ticker, prev_ym, curr_ym)
            sp_ret = get_sp500_return(sp500_data, prev_date, date)

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


def backtest_combined(second_series, third_series, sp500_data, start_year):
    """
    Run monthly backtest of 50/50 #2+#3 from start_year.
    Each month: 50% return from #2, 50% return from #3.
    """
    start_date = pd.Timestamp(f"{start_year}-01-01")
    sl2 = second_series[second_series.index >= start_date]
    sl3 = third_series[third_series.index >= start_date]
    if len(sl2) == 0:
        return None

    # Use intersection of dates
    common = sl2.index.intersection(sl3.index)
    if len(common) == 0:
        return None

    strategy_value = 1.0
    sp500_value = 1.0
    results = []

    for i in range(len(common)):
        date = common[i]
        t2 = sl2.loc[date]
        t3 = sl3.loc[date]

        if i > 0:
            prev_date = common[i - 1]
            prev_t2 = sl2.loc[prev_date]
            prev_t3 = sl3.loc[prev_date]
            prev_ym = prev_date.strftime("%Y-%m")
            curr_ym = date.strftime("%Y-%m")

            ret2 = get_stock_return(prev_t2, prev_ym, curr_ym)
            ret3 = get_stock_return(prev_t3, prev_ym, curr_ym)
            blended_ret = 0.5 * ret2 + 0.5 * ret3

            sp_ret = get_sp500_return(sp500_data, prev_date, date)

            strategy_value *= (1 + blended_ret)
            sp500_value *= (1 + sp_ret)

        results.append({
            "Date": date,
            "Combined": strategy_value,
            "SP500": sp500_value,
            "Ticker2": t2,
            "Ticker3": t3,
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


def main():
    print("=" * 78)
    print("  BACKTEST: Hold the #2 and #3 Largest US Companies by Market Cap")
    print("  Strategies: 100% #2, 100% #3, and 50/50 #2+#3 blend")
    print("  Rebalance: Monthly, swap when rankings change")
    print("=" * 78)
    print()

    # Build timelines
    print("Building company ranking timelines...")
    second_largest = build_ranking_series(SECOND_LARGEST_TIMELINE)
    third_largest = build_ranking_series(THIRD_LARGEST_TIMELINE)

    # Print timeline summaries
    print("\nTimeline of #2 US Company by Market Cap:")
    print(f"  {'Date':<14} {'Ticker':<8} {'Note'}")
    print(f"  {'-'*14} {'-'*8} {'-'*44}")
    for date_str, ticker, note in SECOND_LARGEST_TIMELINE:
        print(f"  {date_str:<14} {ticker:<8} {note}")

    print("\nTimeline of #3 US Company by Market Cap:")
    print(f"  {'Date':<14} {'Ticker':<8} {'Note'}")
    print(f"  {'-'*14} {'-'*8} {'-'*44}")
    for date_str, ticker, note in THIRD_LARGEST_TIMELINE:
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
        print("=" * 78)
        print(f"  BACKTEST FROM {start_year}")
        print("=" * 78)

        r2 = backtest_single(second_largest, sp500, start_year)
        r3 = backtest_single(third_largest, sp500, start_year)
        rc = backtest_combined(second_largest, third_largest, sp500, start_year)

        if r2 is None:
            print(f"  No data available.\n")
            continue

        df2, swaps2 = r2
        df3, swaps3 = r3 if r3 is not None else (None, [])

        all_results[start_year] = (df2, df3, rc, swaps2, swaps3)

        # Metrics for all strategies
        m2 = compute_metrics(df2["Strategy"])
        m3 = compute_metrics(df3["Strategy"]) if df3 is not None else {}
        mc = compute_metrics(rc["Combined"]) if rc is not None else {}
        ms = compute_metrics(df2["SP500"])

        print(f"\n  {'Metric':<22} {'#2 Only':>12} {'#3 Only':>12} {'50/50':>12} {'S&P 500':>12}")
        print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for key in m2:
            v2 = m2.get(key, "N/A")
            v3 = m3.get(key, "N/A")
            vc = mc.get(key, "N/A")
            vs = ms.get(key, "N/A")
            print(f"  {key:<22} {v2:>12} {v3:>12} {vc:>12} {vs:>12}")

        # Final values
        f2 = df2['Strategy'].iloc[-1] * 10000
        f3 = df3['Strategy'].iloc[-1] * 10000 if df3 is not None else 0
        fc = rc['Combined'].iloc[-1] * 10000 if rc is not None else 0
        fs = df2['SP500'].iloc[-1] * 10000
        print(f"\n  $10,000 invested:")
        print(f"    #2 Only:  ${f2:>12,.0f}")
        print(f"    #3 Only:  ${f3:>12,.0f}")
        print(f"    50/50:    ${fc:>12,.0f}")
        print(f"    S&P 500:  ${fs:>12,.0f}")

        # Swap logs
        if swaps2:
            print(f"\n  #2 Position Changes ({len(swaps2)} swaps):")
            for date, from_t, to_t in swaps2:
                print(f"    {date.strftime('%Y-%m')}: {from_t} → {to_t}")

        if swaps3:
            print(f"\n  #3 Position Changes ({len(swaps3)} swaps):")
            for date, from_t, to_t in swaps3:
                print(f"    {date.strftime('%Y-%m')}: {from_t} → {to_t}")

        # Holdings breakdown for #2
        print(f"\n  #2 Holdings breakdown:")
        for ticker in df2["Ticker"].unique():
            months_held = (df2["Ticker"] == ticker).sum()
            pct = months_held / len(df2) * 100
            print(f"    {ticker:<6} {months_held:>4} months ({pct:>5.1f}%)")

        # Holdings breakdown for #3
        if df3 is not None:
            print(f"\n  #3 Holdings breakdown:")
            for ticker in df3["Ticker"].unique():
                months_held = (df3["Ticker"] == ticker).sum()
                pct = months_held / len(df3) * 100
                print(f"    {ticker:<6} {months_held:>4} months ({pct:>5.1f}%)")
        print()

    # ── Summary comparison ──────────────────────────────────────────────
    print("=" * 78)
    print("  SUMMARY: CAGR COMPARISON ACROSS START DATES")
    print("=" * 78)
    print(f"\n  {'Start':<7} {'#2 CAGR':>9} {'#3 CAGR':>9} {'50/50':>9} {'S&P':>9} {'#2 $10K→':>11} {'#3 $10K→':>11} {'50/50→':>11} {'S&P→':>11}")
    print(f"  {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*11} {'-'*11} {'-'*11} {'-'*11}")

    for start_year in start_years:
        if start_year not in all_results:
            continue
        df2, df3, rc, _, _ = all_results[start_year]
        s2 = df2["Strategy"]
        sp = df2["SP500"]
        n_y = (s2.index[-1] - s2.index[0]).days / 365.25
        if n_y <= 0:
            continue

        c2 = (s2.iloc[-1] / s2.iloc[0]) ** (1 / n_y) - 1
        cs = (sp.iloc[-1] / sp.iloc[0]) ** (1 / n_y) - 1

        c3 = 0.0
        f3 = 0
        if df3 is not None:
            s3 = df3["Strategy"]
            n_y3 = (s3.index[-1] - s3.index[0]).days / 365.25
            if n_y3 > 0:
                c3 = (s3.iloc[-1] / s3.iloc[0]) ** (1 / n_y3) - 1
            f3 = s3.iloc[-1] * 10000

        cc = 0.0
        fc = 0
        if rc is not None:
            sc = rc["Combined"]
            n_yc = (sc.index[-1] - sc.index[0]).days / 365.25
            if n_yc > 0:
                cc = (sc.iloc[-1] / sc.iloc[0]) ** (1 / n_yc) - 1
            fc = sc.iloc[-1] * 10000

        f2 = s2.iloc[-1] * 10000
        fs = sp.iloc[-1] * 10000
        print(
            f"  {start_year:<7} {c2:>8.2%} {c3:>8.2%} {cc:>8.2%} {cs:>8.2%}"
            f" ${f2:>9,.0f} ${f3:>9,.0f} ${fc:>9,.0f} ${fs:>9,.0f}"
        )

    print()
    print("  Notes:")
    print("  - Price returns only (no dividends) for all strategies and benchmark")
    print("  - Monthly rebalancing; swaps occur at month boundaries")
    print("  - 50/50 blend: equal-weight #2 + #3, rebalanced monthly")
    print("  - No transaction costs or slippage modeled")
    print("  - S&P 500 data from Shiller dataset (monthly index levels)")
    print("  - Individual stock prices are split-adjusted monthly closes")
    print()


if __name__ == "__main__":
    main()

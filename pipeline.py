import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_sp500_data():
    """
    Fetch S&P 500 tickers and industry data from Wikipedia.

    Returns:
        dict: A dictionary mapping ticker symbols (str) to their industry data.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500 = pd.read_html(url)[0]

    # Create dictionary with requested fields
    ticker_data_dict = {
        t.replace(".", "-"): {
            "ticker": t.replace(".", "-"),
            "name": n,
            "industry": ind,
            "sub_industry": sub,
        }
        for t, n, ind, sub in zip(
            sp500["Symbol"],
            sp500["Security"],
            sp500["GICS Sector"],
            sp500["GICS Sub-Industry"],
        )
    }

    return ticker_data_dict


def fetch_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for the data (YYYY-MM-DD).
        end_date (str): End date for the data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame containing historical stock data for the given ticker.
    """
    data = yf.download(tickers=ticker, start=start_date, end=end_date, interval="1d", threads=True)

    return data


if __name__ == "__main__":
    # get S&P 500 data
    data = get_sp500_data()

    # check first ticker symbol
    first_ticker = list(data.keys())[0]
    print(f"Stock: {first_ticker}")

    # find today and look back dates
    today = datetime.today()
    look_back = today - relativedelta(months=6)

    # convert to strings
    today = today.strftime("%Y-%m-%d")
    look_back = look_back.strftime("%Y-%m-%d")

    # fetch stock data for first ticker
    fetch_ticker_data(ticker=first_ticker, start_date=look_back, end_date=today)

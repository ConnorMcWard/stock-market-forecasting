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
            "Ticker": t.replace(".", "-"),
            "Name": n,
            "Industry": ind,
            "Sub_industry": sub,
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
    data = data.reset_index()
    data.columns = [col[0] for col in data.columns]
    data['Ticker'] = ticker

    return data


def combine_stock_data(ticker_data_dict: dict, ticker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine historical ticker data with industry and subindustry.

    Args:
        ticker (str): Stock ticker symbol.
        ticker_data_dict (dict): Dictionary of Ticker data. This includes Industry and sub-industry.
        ticker_df (pd.DataFrame): DataFrame of historical ticker data.
    """
    ticker_data_dict_df = pd.DataFrame(ticker_data_dict).T.reset_index(drop=True)

    combined_df = pd.merge(ticker_data_dict_df, ticker_df, how="right", on="Ticker")

    return combined_df


def save_stock_data(ticker: str, full_df: pd.DataFrame, output_path: str) -> None:
    """
    Save historical and descriptive stock data to data output folder for model training

    Args:
        ticker (str): Stock ticker symbol.
        full_df (pd.DataFrame): DataFrame after being combined using combine_stock_data.
        output_path (str): output directory for storing data.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    full_file_path = os.path.join(output_path, f"{ticker}.csv")
    full_df.to_csv(path_or_buf=full_file_path, index=False)


if __name__ == "__main__":
    # get S&P 500 data
    sp_500 = get_sp500_data()

    # check first ticker symbol
    first_ticker = list(sp_500.keys())[0]
    print(f"Stock: {first_ticker}")

    # find today and look back dates
    today = datetime.today()
    look_back = today - relativedelta(months=6)

    # convert to strings
    today = today.strftime("%Y-%m-%d")
    look_back = look_back.strftime("%Y-%m-%d")

    # fetch stock data for first ticker
    df = fetch_ticker_data(ticker=first_ticker, start_date=look_back, end_date=today)

    combined_df = combine_stock_data(ticker_data_dict=sp_500, ticker_df=df)

    save_stock_data(first_ticker, full_df=combined_df, output_path="data/")

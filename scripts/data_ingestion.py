import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_sp500_data():
    """
    Fetch S&P 500 tickers and metadata from Wikipedia.

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
            "Sub_Industry": sub,
        }
        for t, n, ind, sub in zip(
            sp500["Symbol"],
            sp500["Security"],
            sp500["GICS Sector"],
            sp500["GICS Sub-Industry"],
        )
    }

    return ticker_data_dict


def fetch_ticker_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol.
        period (str): How far to look back from last available trading day. \n
                        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

    Returns:
        data (pd.DataFrame): A DataFrame containing historical stock data for the given ticker.
    """
    try:
        # download historical ticker data
        data = yf.download(
            tickers=ticker, 
            period=period,
            interval="1d",
            actions=True, 
            threads=True, 
            auto_adjust=False,
            multi_level_index=False,
        ).reset_index()

        # add ticker column
        data['Ticker'] = ticker

    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        data=pd.DataFrame()

    return data


def combine_stock_data(ticker_data_dict: dict, ticker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine historical ticker data with industry and subindustry metadata.

    Args:
        ticker (str): Stock ticker symbol.
        ticker_data_dict (dict): Dictionary of Ticker data. This includes Industry and sub-industry.
        ticker_df (pd.DataFrame): DataFrame of historical ticker data.
    """
    # convert ticker_data_dict to a DataFrame and Transpose it, while dropping unecessary index
    ticker_data_dict_df = pd.DataFrame(ticker_data_dict).T.reset_index(drop=True)

    # merge DataFrames together
    combined_df = pd.merge(ticker_data_dict_df, ticker_df, how="right", on="Ticker")

    return combined_df


def save_stock_data(ticker: str, full_df: pd.DataFrame, output_path: str) -> None:
    """
    Save metadata and historical stock data to data output folder for model training

    Args:
        ticker (str): Stock ticker symbol.
        full_df (pd.DataFrame): DataFrame after being combined using combine_stock_data.
        output_path (str): output directory for storing data.
    """
    # create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    full_file_path = os.path.join(output_path, f"{ticker}.csv")
    full_df.to_csv(path_or_buf=full_file_path, index=False)


if __name__ == "__main__":
    # get S&P 500 data
    sp_500 = get_sp500_data()

    for ticker in list(sp_500.keys()):
        print(f"Ticker: {ticker}")

        df = fetch_ticker_data(ticker=ticker, period='2y')

        combined_df = combine_stock_data(ticker_data_dict=sp_500, ticker_df=df)

        save_stock_data(ticker, full_df=combined_df, output_path="data/input/")

    print("S&P 500 stocks download complete!")


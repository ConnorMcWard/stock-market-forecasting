import yfinance as yf
import pandas as pd
import numpy as np

def get_sp500_data():
    """
    Fetch S&P 500 tickers and industry data

    :return ticker_data: dictionary of ticker information
    :rtype dict:
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500 = pd.read_html(url)[0]
    tickers = [t.replace('.', '-') for t in sp500['Symbol'].tolist()]
    industries = sp500['GICS Sector'].tolist()
    sub_industries = sp500['GICS Sub-Industry'].tolist()
    
    # Create mappings
    industry_map = {ticker: idx for idx, ticker in enumerate(tickers)}
    unique_industries = list(set(industries))
    unique_sub_industries = list(set(sub_industries))
    industry_to_idx = {ind: i for i, ind in enumerate(unique_industries)}
    sub_industry_to_idx = {sub: i for i, sub in enumerate(unique_sub_industries)}
    
    ticker_info = {
        ticker: {
            'industry_idx': industry_to_idx[ind],
            'sub_industry_idx': sub_industry_to_idx[sub]
        }
        for ticker, ind, sub in zip(tickers, industries, sub_industries)
    }
    return tickers, ticker_info, len(unique_industries), len(unique_sub_industries)





if __name__ == "__main__":
    get_sp500_data()
    


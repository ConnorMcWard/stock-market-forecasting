# scripts/data_preparation.py
import os
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_concat_data(data_folder):
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    df_list = [pd.read_csv(file) for file in csv_files]
    return pd.concat(df_list, ignore_index=True)

def prepare_data(df, window_size=10):
    # Convert Date column and sort
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Ticker", "Date"])

    # One-hot encode static features
    static_features = df[["Ticker", "Industry", "Sub_Industry"]].drop_duplicates()
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoded_static = encoder.fit_transform(static_features)
    static_columns = encoder.get_feature_names_out(["Ticker", "Industry", "Sub_Industry"])
    static_features_df = pd.DataFrame(encoded_static, columns=static_columns)
    static_features_df["Ticker"] = static_features["Ticker"].values

    sequences, targets, tickers = [], [], []
    scalers_dict = {}
    dynamic_features = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]

    for ticker, ticker_df in df.groupby("Ticker"):
        scaler = StandardScaler()
        train_data = ticker_df[dynamic_features].iloc[:-30]  # Avoid leakage
        scaler.fit(train_data)
        scalers_dict[ticker] = scaler

        normalized_data = scaler.transform(ticker_df[dynamic_features])
        ticker_df.loc[:, dynamic_features] = normalized_data

        for i in range(len(ticker_df) - window_size):
            seq = ticker_df.iloc[i:i + window_size][dynamic_features].values
            target = ticker_df.iloc[i + window_size]["Adj Close"]
            static_vector = static_features_df[static_features_df["Ticker"] == ticker]\
                                .drop(columns=["Ticker"]).values.flatten()
            seq_with_static = np.hstack([seq, np.tile(static_vector, (window_size, 1))])
            sequences.append(seq_with_static)
            targets.append(target)
            tickers.append(ticker)

    return np.array(sequences), np.array(targets), np.array(tickers), scalers_dict, static_features_df, df

def main():
    data_folder = "../data/input/"  # adjust path as needed
    df = load_and_concat_data(data_folder)
    X, y, tickers_array, scalers_dict, static_features_df, processed_df = prepare_data(df)
    np.savez("processed_data.npz", X=X, y=y, tickers=tickers_array)
    
    # Save scalers and static features for later use (e.g., in retraining and visualization)
    with open("models/scalers_dict.pkl", "wb") as f:
        pickle.dump(scalers_dict, f)
    with open("models/static_features_df.pkl", "wb") as f:
        pickle.dump(static_features_df, f)
    
    # Optionally, you could also save the processed_df with normalized values
    processed_df.to_csv("data/processed_data.csv", index=False)
    print("Data preparation complete.")

if __name__ == "__main__":
    main()

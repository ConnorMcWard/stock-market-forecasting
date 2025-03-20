import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from multiprocessing import Pool, cpu_count
import json


# Define paths dynamically
BASE_DIR = os.getenv("DATA_DIR", os.path.abspath("data"))
RAW_DATA_PATH = os.path.join(BASE_DIR, "input")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "processed")

# Ensure output directory exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Get all CSV files
csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DATA_PATH}")

# Load DataFrames efficiently using generators
df_list = (pd.read_csv(file) for file in csv_files)
full_df = pd.concat(df_list, ignore_index=True)

# Extract and encode static features
static_features = full_df[['Ticker', 'Industry', 'Sub_Industry']].drop_duplicates(subset=["Ticker"], keep="first")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_static = encoder.fit_transform(static_features[['Industry', 'Sub_Industry']])
static_columns = list(encoder.get_feature_names_out(['Industry', 'Sub_Industry']))
static_columns = [col.replace("Industry_", "Ind_").replace("Sub_Industry_", "SubInd_") for col in static_columns]
static_features_df = pd.DataFrame(encoded_static, columns=static_columns)
static_features_df['Ticker'] = static_features['Ticker'].values

# Function to process a single ticker
def process_ticker_and_save(ticker_data):
    ticker, ticker_df, static_vector, window_size = ticker_data
    dynamic_features = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]

    # Avoid data leakage: fit on training set only
    scaler = StandardScaler()
    train_idx = int(len(ticker_df) * 0.8)  # Use 80% for training
    train_data = ticker_df[dynamic_features].iloc[:train_idx]
    scaler.fit(train_data)
    
    # Apply transformation
    ticker_df[dynamic_features] = scaler.transform(ticker_df[dynamic_features])

    # Create sliding windows
    sequences, targets = [], []
    for i in range(len(ticker_df) - window_size):
        seq = ticker_df.iloc[i:i + window_size][dynamic_features].values
        target = ticker_df.iloc[i + window_size]["Adj Close"]
        static_repeated = np.tile(static_vector, (window_size, 1))
        full_seq = np.hstack([seq, static_repeated])
        sequences.append(full_seq)
        targets.append(target)

    # Save processed data
    np.savez(os.path.join(PROCESSED_DATA_PATH, f"{ticker}_data.npz"), X=sequences, y=targets)

# Parallelized processing
def create_sliding_windows_parallel(ticker_groups, static_features_df, window_size=10):
    ticker_data_list = []
    for ticker, ticker_df in ticker_groups:
        static_vector = static_features_df[static_features_df["Ticker"] == ticker].drop(columns=["Ticker"]).values.flatten()
        ticker_data_list.append((ticker, ticker_df, static_vector, window_size))
    
    n_jobs = max(cpu_count() - 2, 1)  # Avoid overloading the CPU
    with Pool(n_jobs) as pool:
        pool.map(process_ticker_and_save, ticker_data_list)

# Main execution
if __name__ == "__main__":
    create_sliding_windows_parallel(full_df.groupby('Ticker'), static_features_df, window_size=10)
    print(f"Preprocessing complete! Processed data saved in '{PROCESSED_DATA_PATH}'.")
    
    # Save static feature names to a JSON file
    static_feature_names_path = os.path.join(PROCESSED_DATA_PATH, "static_feature_names.json")

    with open(static_feature_names_path, "w") as f:
        json.dump(static_columns, f)

    print(f"Static feature names saved to {static_feature_names_path}")
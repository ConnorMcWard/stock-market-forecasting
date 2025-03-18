# scripts/visualization.py
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from tensorflow.keras.models import load_model

# Load scalers, static features, and model
with open("models/scalers_dict.pkl", "rb") as f:
    scalers_dict = pickle.load(f)
with open("models/static_features_df.pkl", "rb") as f:
    static_features_df = pickle.load(f)
model = load_model("models/model.h5")

# Load original processed data if needed
df = pd.read_csv("data/processed_data.csv")

# Function to make a prediction for a given ticker
def predict_tomorrow(ticker, window_size=10):
    ticker_df = df[df["Ticker"] == ticker].sort_values("Date")
    if len(ticker_df) < window_size:
        return None, None, None
    dynamic_features = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    latest_window = ticker_df.iloc[-window_size:][dynamic_features].values
    scaler = scalers_dict[ticker]
    normalized_window = scaler.transform(latest_window)
    static_vector = static_features_df[static_features_df["Ticker"] == ticker]\
                        .drop(columns=["Ticker"]).values.flatten()
    static_repeated = np.tile(static_vector, (window_size, 1))
    input_seq = np.hstack([normalized_window, static_repeated])[np.newaxis, ...]
    normalized_pred = model.predict(input_seq)[0, 0]
    predicted_price = normalized_pred * scaler.scale_[0] + scaler.mean_[0]
    current_price = ticker_df.iloc[-1]["Adj Close"]
    roi = (predicted_price - current_price) / current_price
    return predicted_price, current_price, roi

# Build dashboard with Streamlit
st.title("Top Performing Stocks Prediction Dashboard")

tickers = df["Ticker"].unique()
results = []
for ticker in tickers:
    pred, current, roi = predict_tomorrow(ticker)
    if pred is not None:
        results.append({"Ticker": ticker, "Predicted Price": pred, "Current Price": current, "Predicted ROI": roi})

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Predicted ROI", ascending=False)

st.subheader("Top Performing Stocks by Predicted ROI")
st.dataframe(results_df)

st.subheader("Historical Data & Prediction Plots")
selected_ticker = st.selectbox("Select a ticker", tickers)
ticker_df = df[df["Ticker"] == selected_ticker].sort_values("Date")
fig = px.line(ticker_df, x="Date", y="Adj Close", title=f"Historical Adjusted Close for {selected_ticker}")
st.plotly_chart(fig)

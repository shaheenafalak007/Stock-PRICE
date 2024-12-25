import yfinance as yf
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st

# Your existing functions and model code here...
# You can copy the functions `fetch_and_clean_data`, `add_technical_indicators`, `create_sequences`, etc., from your existing code.

# Add Streamlit UI components for user input
st.title("Stock Price Prediction using Transformer Model")

# User Input for Ticker, Start Date, and End Date
ticker = st.text_input("Enter Stock Ticker:", "AAPL")
start_date = st.date_input("Select Start Date:", pd.to_datetime("1990-01-01"))
end_date = st.date_input("Select End Date:", pd.to_datetime("2023-01-01"))
seq_len = st.slider("Sequence Length (timesteps per sequence):", min_value=5, max_value=30, value=10)

# Fetch and prepare data based on user input
if st.button("Fetch Data"):
    try:
        raw_data = fetch_and_clean_data(ticker, start_date, end_date)
        data_with_indicators = add_technical_indicators(raw_data)
        features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_High', 'BB_Low']
        data_for_modeling = data_with_indicators[features].values

        # Normalize data
        scaled_data, scaler = normalize_data(data_with_indicators, features)

        # Create sequences
        X, y = create_sequences(scaled_data, features, seq_len)

        # Prepare PyTorch DataLoaders
        train_loader, test_loader = prepare_pytorch_data(X, y)

        st.success("Data successfully fetched and prepared!")

        # Visualize the stock data and indicators
        st.subheader("Stock Price and Technical Indicators")
        plot_data(data_with_indicators)

        st.write(f"Shape of Data for Model: {X.shape}, {y.shape}")
        st.write(f"Training DataLoader: {len(train_loader)} batches")
        st.write(f"Testing DataLoader: {len(test_loader)} batches")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Add Model Hyperparameters input and Model Creation
embedding_dim = st.slider("Embedding Dimension:", min_value=32, max_value=128, value=64)
num_heads = st.slider("Number of Attention Heads:", min_value=2, max_value=8, value=4)
num_layers = st.slider("Number of Transformer Layers:", min_value=1, max_value=4, value=2)

# Create the model
model = StockPriceTransformer(input_dim=6, embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, seq_len=seq_len)

st.write("Transformer Model Architecture:")
st.text(model)

# Add Model Training and Evaluation Buttons
if st.button("Train Model"):
    # Train your model here, you would need to write the training loop
    # You can display training progress using st.progress or st.write
    st.write("Training your model...")

    # You can also display evaluation metrics like MSE, RMSE, R² on success
    st.write("Model trained successfully!")

# Add a Section for Evaluation
if st.button("Evaluate Model"):
    # Evaluate your model on test set here
    st.write("Evaluating the model...")
    
    # After evaluation, you can show metrics like RMSE, R², etc.
    st.write("Model evaluation completed!")

    # Display actual vs predicted plot
    st.subheader("Actual vs Predicted Stock Prices")
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, label="Actual Prices", color="blue")
    plt.plot(y_pred, label="Predicted Prices", color="red", linestyle="--")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.title("Actual vs Predicted Stock Prices")
    plt.legend()
    st.pyplot()


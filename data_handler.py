"""Data fetching and preprocessing functions"""

import numpy as np
import yfinance as yf
import torch
from sklearn.preprocessing import StandardScaler
from config import DEVICE, SEQ_LENGTH, TRAIN_SPLIT, START_DATE, HISTORICAL_START, HISTORICAL_END


def get_stock_data(ticker):
    """Ask the user for the stock ticker, and make sure the ticker exists"""
    try:
        stock_data = yf.download(ticker, start=HISTORICAL_START, end=HISTORICAL_END)
        if stock_data.empty:
            raise ValueError("No data found for the ticker.")
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def download_recent_data(ticker):
    """Download recent data for analysis"""
    return yf.download(ticker, START_DATE)


def prepare_data(df, seq_length=SEQ_LENGTH):
    """Prepare data for LSTM training"""
    scaler = StandardScaler()
    df_copy = df.copy()
    df_copy['Close'] = scaler.fit_transform(df_copy[['Close']])
    
    data = []
    for i in range(len(df_copy) - seq_length):
        data.append(df_copy.Close[i:i+seq_length])
    data = np.array(data)
    
    return data, scaler


def create_train_test_split(data, train_split=TRAIN_SPLIT):
    """Split data into train and test sets"""
    train_size = int(train_split * len(data))
    
    X_train = torch.from_numpy(data[:train_size, :-1]).unsqueeze(-1).type(torch.Tensor).to(DEVICE)
    y_train = torch.from_numpy(data[:train_size, -1]).unsqueeze(-1).type(torch.Tensor).to(DEVICE)
    X_test = torch.from_numpy(data[train_size:, :-1]).unsqueeze(-1).type(torch.Tensor).to(DEVICE)
    y_test = torch.from_numpy(data[train_size:, -1]).unsqueeze(-1).type(torch.Tensor).to(DEVICE)
    
    return X_train, y_train, X_test, y_test
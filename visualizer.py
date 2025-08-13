"""Visualization functions for stock prediction results"""

import matplotlib.pyplot as plt
from config import FIGURE_SIZE, PREDICTION_FIGURE_SIZE


def plot_initial_data(df, ticker):
    """Plot the initial stock data"""
    df.Close.plot(figsize=FIGURE_SIZE)
    plt.title(f"{ticker} Stock Price - Raw Data")
    plt.show()


def plot_predictions(df, ticker, y_test, y_test_pred, test_rmse):
    """Plot predictions vs actual values"""
    fig = plt.figure(figsize=PREDICTION_FIGURE_SIZE)
    gs = fig.add_gridspec(4, 1)
    
    # Main prediction plot
    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.plot(df.iloc[-len(y_test):].index, y_test, color='blue', label='Actual Price')
    ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, color='green', label='Predicted Price')
    ax1.legend()
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    # Error plot
    ax2 = fig.add_subplot(gs[3, 0])
    ax2.axhline(test_rmse, color='blue', linestyle='--', label='RMSE')
    ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_pred), 'r', label='Prediction Error')
    ax2.legend()
    plt.title('Prediction Error')
    plt.xlabel('Date')
    plt.ylabel('Error')
    
    plt.tight_layout()
    plt.show()
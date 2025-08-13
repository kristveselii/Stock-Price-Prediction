"""Main execution script for stock prediction"""

from data_handler import get_stock_data, download_recent_data, prepare_data, create_train_test_split
from model import PredictionModel
from trainer import train_model, evaluate_model
from visualizer import plot_initial_data, plot_predictions
from config import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DEVICE


def main():
    # Get user input
    ticker = input("Enter the stock ticker symbol (e.g., AAPL, MSFT): ").strip().upper()
    
    # Fetch data
    stock_data = get_stock_data(ticker)
    if stock_data is None:
        print("Failed to fetch stock data. Exiting.")
        return
    
    # Download and plot initial data
    df = download_recent_data(ticker)
    plot_initial_data(df, ticker)
    
    # Prepare data
    data, scaler = prepare_data(df)
    X_train, y_train, X_test, y_test = create_train_test_split(data)
    
    # Create and train model
    model = PredictionModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(DEVICE)
    model = train_model(model, X_train, y_train)
    
    # Evaluate model
    y_train_pred, y_train_actual, y_test_pred, y_test_actual, train_rmse, test_rmse = evaluate_model(
        model, X_train, y_train, X_test, y_test, scaler
    )
    
    # Print results
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Plot results
    plot_predictions(df, ticker, y_test_actual, y_test_pred, test_rmse)


if __name__ == "__main__":
    main()
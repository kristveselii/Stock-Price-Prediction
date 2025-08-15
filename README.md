# Stock Price Prediction with LSTM

A deep learning-based stock price prediction tool using Long Short-Term Memory (LSTM) neural networks. This project fetches real-time stock data and uses historical price patterns to predict future stock prices.

## Features

- **Real-time Data Fetching**: Downloads stock data from Yahoo Finance
- **LSTM Neural Network**: Uses PyTorch to build and train a 2-layer LSTM model
- **Data Preprocessing**: Automatic scaling and sequence preparation for time series data
- **Interactive Visualization**: Plots both raw data and prediction results with error analysis
- **Performance Metrics**: Calculates RMSE for both training and testing datasets
- **GPU Support**: Automatically detects and uses CUDA if available

## Requirements

```
numpy
pandas
matplotlib
yfinance
torch
scikit-learn
```

## Installation

1. Clone this repository:
```bash
git clone <https://github.com/kristveselii/Stock-Price-Prediction.git>
cd Stock-Price-Prediction
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib yfinance torch scikit-learn
```

## Usage

Run the main script:
```bash
python main.py
```

When prompted, enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA, GOOGL).

The program will:
1. Fetch historical stock data
2. Display the raw stock price chart
3. Train an LSTM model on the data
4. Show training progress with loss values
5. Display prediction results vs actual prices
6. Show prediction error analysis

## Model Architecture

- **Input Layer**: Single feature (closing price)
- **LSTM Layers**: 2 hidden layers with 32 units each
- **Output Layer**: Single dense layer for price prediction
- **Sequence Length**: 30 days of historical data to predict next day
- **Training Split**: 80% training, 20% testing

## Configuration

Key parameters can be modified in `config.py`:

```python
# Data parameters
SEQ_LENGTH = 30        # Days of history to use for prediction
TRAIN_SPLIT = 0.8      # Train/test split ratio

# Model parameters
HIDDEN_DIM = 32        # LSTM hidden units
NUM_LAYERS = 2         # Number of LSTM layers

# Training parameters
LEARNING_RATE = 0.01   # Adam optimizer learning rate
NUM_EPOCHS = 200       # Training epochs
```

## Output Interpretation

### RMSE Values
- **Lower RMSE = Better Model**: Smaller values indicate more accurate predictions
- **Units**: RMSE is in the same units as stock price (dollars)
- **Example**: RMSE of $2.50 means predictions are typically off by about $2.50

### Visualizations
1. **Raw Data Plot**: Shows historical stock price trend
2. **Prediction Plot**: Green line (predictions) vs Blue line (actual prices)
3. **Error Plot**: Shows prediction errors over time with RMSE reference line

## Performance Notes

- **Training Time**: Depends on dataset size and hardware (GPU recommended)
- **Memory Usage**: Scales with sequence length and batch size
- **Accuracy**: Results vary significantly based on stock volatility and market conditions

## Limitations

- **Past Performance**: Historical patterns may not predict future prices
- **Market Volatility**: High volatility stocks are harder to predict accurately
- **External Factors**: Model doesn't account for news, earnings, or market sentiment
- **Short-term Focus**: Optimized for next-day predictions, not long-term forecasting

## Example Output

```
Enter the stock ticker symbol (e.g., AAPL, MSFT): AAPL
Epoch 0: Loss = 0.847329
Epoch 25: Loss = 0.234567
Epoch 50: Loss = 0.156789
...
Epoch 175: Loss = 0.023456
Train RMSE: 2.1234
Test RMSE: 3.4567
```

## Customization

### Adding New Features
- Modify `data_handler.py` to include volume, moving averages, or technical indicators
- Update `INPUT_DIM` in `config.py` accordingly

### Different Models
- Replace LSTM in `model.py` with GRU, Transformer, or other architectures
- Adjust parameters in `config.py` as needed

### Extended Predictions
- Modify sequence generation to predict multiple days ahead
- Update visualization to show longer prediction horizons

## Troubleshooting

### Common Issues

**"No data found for ticker"**
- Verify ticker symbol is correct
- Check internet connection
- Some tickers may not be available on Yahoo Finance

**CUDA out of memory**
- Reduce `SEQ_LENGTH` or batch size
- Set device to CPU in `config.py`: `DEVICE = torch.device('cpu')`

**Poor prediction accuracy**
- Try different hyperparameters (learning rate, hidden dimensions)
- Increase training epochs
- Consider using more features beyond just closing price

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your change
"""Model training and evaluation functions"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
from config import NUM_EPOCHS, LEARNING_RATE, PRINT_EVERY


def train_model(model, X_train, y_train, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """Train the LSTM model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for i in range(num_epochs):
        y_train_pred = model(X_train)
        loss = criterion(y_train_pred, y_train)
        
        if i % PRINT_EVERY == 0:
            print(f"Epoch {i}: Loss = {loss.item():.6f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    """Evaluate model performance"""
    model.eval()
    
    # Get predictions
    y_train_pred = model(X_train)
    y_test_pred = model(X_test)
    
    # Inverse transform predictions and actual values
    y_train_pred_inv = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
    y_train_inv = scaler.inverse_transform(y_train.detach().cpu().numpy())
    y_test_pred_inv = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
    y_test_inv = scaler.inverse_transform(y_test.detach().cpu().numpy())
    
    # Calculate RMSE
    train_rmse = root_mean_squared_error(y_train_inv[:, 0], y_train_pred_inv[:, 0])
    test_rmse = root_mean_squared_error(y_test_inv[:, 0], y_test_pred_inv[:, 0])
    
    return y_train_pred_inv, y_train_inv, y_test_pred_inv, y_test_inv, train_rmse, test_rmse
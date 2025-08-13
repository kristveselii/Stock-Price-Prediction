"""Configuration settings for stock prediction model"""

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data parameters
SEQ_LENGTH = 30
TRAIN_SPLIT = 0.8
START_DATE = "2020-01-01"
HISTORICAL_START = "2010-01-01"
HISTORICAL_END = "2023-10-01"

# Model parameters
INPUT_DIM = 1
HIDDEN_DIM = 32
NUM_LAYERS = 2
OUTPUT_DIM = 1

# Training parameters
LEARNING_RATE = 0.01
NUM_EPOCHS = 200
PRINT_EVERY = 25

# Visualization parameters
FIGURE_SIZE = (12, 8)
PREDICTION_FIGURE_SIZE = (12, 10)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.mlp import MLPRegressor
from src.evaluation.metrics import mae, rmse


# ----------------------------
# STEP 1: Load dataset
# ----------------------------
# We read the synthetic option pricing data you generated on Day 1.
df = pd.read_csv("data/options_data.csv")


# ----------------------------
# STEP 2: Select inputs and target
# ----------------------------
# Inputs (features): the variables that describe each option
X = df[["S", "K", "T", "r", "sigma"]].values

# Target: what we want the model to learn to predict
y = df["call_price"].values.reshape(-1, 1)


# ----------------------------
# STEP 3: Train/test split
# ----------------------------
# We will use 80% of the data for training
# and 20% for testing.
split_idx = int(0.8 * len(df))

X_train = X[:split_idx]
X_test = X[split_idx:]

y_train = y[:split_idx]
y_test = y[split_idx:]


# ----------------------------
# STEP 4: Convert data to PyTorch tensors
# ----------------------------
# PyTorch models need tensors, not NumPy arrays.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# ----------------------------
# STEP 5: Create model
# ----------------------------
model = MLPRegressor(input_dim=5, hidden_dim=64)


# ----------------------------
# STEP 6: Define loss function and optimizer
# ----------------------------
# Loss function tells us how wrong the model is.
criterion = nn.MSELoss()

# Optimizer updates the model weights to reduce loss.
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ----------------------------
# STEP 7: Training loop
# ----------------------------
epochs = 100

for epoch in range(epochs):
    # Put model in training mode
    model.train()

    # Forward pass: model makes predictions
    predictions = model(X_train_tensor)

    # Compute loss between predictions and true values
    loss = criterion(predictions, y_train_tensor)

    # Zero out old gradients
    optimizer.zero_grad()

    # Backward pass: compute gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}")


# ----------------------------
# STEP 8: Evaluate on test set
# ----------------------------
model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor).numpy()

test_mae = mae(y_test, test_predictions)
test_rmse = rmse(y_test, test_predictions)

print("\nTest Results:")
print(f"MAE: {test_mae:.6f}")
print(f"RMSE: {test_rmse:.6f}")
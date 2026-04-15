import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.models.mlp import MLPRegressor
from src.evaluation.metrics import mae, rmse


# ----------------------------
# STEP 1: Load dataset
# ----------------------------
# Read the synthetic option pricing data generated from Black-Scholes.
df = pd.read_csv("data/options_data.csv")


# ----------------------------
# STEP 2: Select inputs and target
# ----------------------------
# Inputs (features): the variables that describe each option contract
X = df[["S", "K", "T", "r", "sigma"]].values

# Target: what we want the neural network to predict
y = df["call_price"].values.reshape(-1, 1)


# ----------------------------
# STEP 3: Shuffle the dataset
# ----------------------------
# This prevents any accidental ordering effects in the train/test split.
indices = np.random.permutation(len(df))
X = X[indices]
y = y[indices]


# ----------------------------
# STEP 4: Train/test split
# ----------------------------
# Use 80% of the data for training and 20% for testing.
split_idx = int(0.8 * len(df))

X_train = X[:split_idx]
X_test = X[split_idx:]

y_train = y[:split_idx]
y_test = y[split_idx:]


# ----------------------------
# STEP 5: Scale features using TRAINING data only
# ----------------------------
# This avoids data leakage from the test set.
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


# ----------------------------
# STEP 6: Scale targets using TRAINING data only
# ----------------------------
# Scaling the output usually makes neural network training easier.
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)

y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std


# ----------------------------
# STEP 7: Convert data to PyTorch tensors
# ----------------------------
# PyTorch models require tensors, not NumPy arrays.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


# ----------------------------
# STEP 8: Create model
# ----------------------------
model = MLPRegressor(input_dim=5, hidden_dim=64)


# ----------------------------
# STEP 9: Define loss function and optimizer
# ----------------------------
# MSELoss measures squared error between predictions and true targets.
criterion = nn.MSELoss()

# Adam updates the model weights to reduce the loss.
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ----------------------------
# STEP 10: Training loop
# ----------------------------
epochs = 300

for epoch in range(epochs):
    # Put model in training mode
    model.train()

    # Forward pass: predict scaled call prices
    predictions = model(X_train_tensor)

    # Compute training loss
    loss = criterion(predictions, y_train_tensor)

    # Clear old gradients
    optimizer.zero_grad()

    # Compute gradients
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Print progress every 25 epochs
    if (epoch + 1) % 25 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.6f}")


# ----------------------------
# STEP 11: Evaluate on test set
# ----------------------------
model.eval()

with torch.no_grad():
    # Predictions are still in the scaled target space
    test_predictions_scaled = model(X_test_tensor).numpy()

# Convert predictions back to original dollar scale
test_predictions = test_predictions_scaled * y_std + y_mean


# ----------------------------
# STEP 12: Compute metrics in original scale
# ----------------------------
test_mae = mae(y_test, test_predictions)
test_rmse = rmse(y_test, test_predictions)

print("\nTest Results:")
print(f"MAE: {test_mae:.6f}")
print(f"RMSE: {test_rmse:.6f}")


# ----------------------------
# STEP 13: Plot true vs predicted prices
# ----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_predictions, alpha=0.3)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("True vs. Predicted Prices")

# Plot ideal line y = x
min_val = min(y_test.min(), test_predictions.min())
max_val = max(y_test.max(), test_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red")

plt.tight_layout()
plt.savefig("results/pred_vs_true.png")
plt.show()


# ----------------------------
# STEP 14: Save results
# ----------------------------
print("\nSaving results...")
np.save("results/y_test.npy", y_test)
np.save("results/predictions.npy", test_predictions)
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.black_scholes import call_price
from src.models.mlp import MLPRegressor


# ----------------------------
# STEP 1: Recreate training-data scaling constants
# ----------------------------
# These must match the training dataset generation ranges.
# Since the dataset was generated uniformly, we will recompute
# the same training statistics by loading the dataset itself.
import pandas as pd

df = pd.read_csv("data/options_data.csv")

X = df[["S", "K", "T", "r", "sigma"]].values
y = df["call_price"].values.reshape(-1, 1)

# Match the same shuffle logic used in training
np.random.seed(42)
indices = np.random.permutation(len(df))
X = X[indices]
y = y[indices]

split_idx = int(0.8 * len(df))
X_train = X[:split_idx]
y_train = y[:split_idx]

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)


# ----------------------------
# STEP 2: Load trained model
# ----------------------------
model = MLPRegressor(input_dim=5, hidden_dim=64)
model.load_state_dict(torch.load("results/mlp_model.pth"))
model.eval()


# ----------------------------
# STEP 3: Define a fixed option contract
# ----------------------------
# We will vary sigma only.
S = 100
K = 100
T = 1.0
r = 0.03

sigma_values = np.linspace(0.1, 0.8, 100)


# ----------------------------
# STEP 4: Compute Black-Scholes prices
# ----------------------------
bs_prices = call_price(S, K, T, r, sigma_values)


# ----------------------------
# STEP 5: Compute MLP prices
# ----------------------------
mlp_prices = []

with torch.no_grad():
    for sigma in sigma_values:
        x = np.array([[S, K, T, r, sigma]])

        # Scale input using training stats
        x_scaled = (x - X_mean) / X_std
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        # Predict in scaled target space
        pred_scaled = model(x_tensor).numpy()

        # Convert back to original price scale
        pred = pred_scaled * y_std + y_mean
        mlp_prices.append(pred.item())

mlp_prices = np.array(mlp_prices)


# ----------------------------
# STEP 6: Plot comparison
# ----------------------------
plt.figure(figsize=(8, 6))
plt.plot(sigma_values, bs_prices, label="Black-Scholes")
plt.plot(sigma_values, mlp_prices, label="MLP", linestyle="--")

plt.xlabel("Volatility (sigma)")
plt.ylabel("Call Price")
plt.title("Volatility Perturbation: Black-Scholes vs MLP")
plt.legend()
plt.tight_layout()

plt.savefig("results/volatility_perturbation.png")
plt.show()
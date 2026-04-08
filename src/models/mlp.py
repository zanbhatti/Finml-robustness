import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    A simple feedforward neural network (MLP)

    It learns a function:
    (S, K, T, r, sigma) -> call_price
    """

    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()

        # nn.Sequential = stack layers in order
        self.network = nn.Sequential(

            # First layer: maps 5 inputs -> 64 hidden units
            nn.Linear(input_dim, hidden_dim),

            # ReLU activation = introduces non-linearity
            nn.ReLU(),

            # Second layer: 64 -> 64
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            # Final layer: 64 -> 1 (predict price)
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass:
        takes input x and passes it through the network
        """
        return self.network(x)
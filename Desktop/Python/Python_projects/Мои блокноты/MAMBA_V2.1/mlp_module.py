import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            dropout=0.1,
            bias=True,
            device='cuda',
            dtype=None,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        factory_kwargs = {"device": device, "dtype": dtype}

        # First linear layer
        self.fc1 = nn.Linear(
            in_features,
            2 * hidden_features,
            bias=bias,
            **factory_kwargs
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Second linear layer
        self.fc2 = nn.Linear(
            hidden_features,
            out_features,
            bias=bias,
            **factory_kwargs
        )

    def forward(self, x):
        """
        Forward pass for Gated MLP.

        Args:
            x: Input tensor [batch, ..., in_features]

        Returns:
            Output tensor [batch, ..., out_features]
        """
        # First linear layer
        x = self.fc1(x)

        # Split into two parts for gating
        x, gate = x.chunk(2, dim=-1)

        # Apply SiLU activation and gating
        x = x * F.silu(gate)

        # Apply dropout
        x = self.dropout(x)

        # Second linear layer
        x = self.fc2(x)

        return x

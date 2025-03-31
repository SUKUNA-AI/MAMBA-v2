import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyHead(nn.Module):
    def __init__(
            self,
            d_model,
            output_dim,
            dropout=0.1,
            device='cuda',
            dtype=None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        # Mean prediction head
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, **factory_kwargs),
            nn.LayerNorm(d_model // 2, eps=1e-5, **factory_kwargs),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model // 2, output_dim, **factory_kwargs)
        )

        # Variance prediction head
        self.var_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, **factory_kwargs),
            nn.LayerNorm(d_model // 2, eps=1e-5, **factory_kwargs),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model // 2, output_dim, **factory_kwargs),
            nn.Softplus()  # Ensure positive variance
        )

    def forward(self, x):
        """
        Predict mean and variance.

        Args:
            x: Input tensor [batch, d_model]

        Returns:
            Tuple of (mean, variance) tensors [batch, output_dim]
        """
        mean = self.mean_head(x)
        var = self.var_head(x)

        # Add small constant to variance for numerical stability
        var = var + 1e-6

        return mean, var

    def loss(self, mean, var, target):
        """
        Compute negative log likelihood loss with uncertainty.

        Args:
            mean: Predicted mean [batch, output_dim]
            var: Predicted variance [batch, output_dim]
            target: Target values [batch, output_dim]

        Returns:
            Loss scalar
        """
        # Negative log likelihood
        nll = 0.5 * (torch.log(var) + (target - mean)**2 / var)

        # Mean over batch and output dimensions
        return nll.mean()

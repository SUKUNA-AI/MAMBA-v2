import torch
import torch.nn as nn
import numpy as np

class TelemetryPreprocessor(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, device='gpu', dtype=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.device = device

        # Learnable normalization parameters
        self.register_buffer("mean", torch.zeros(input_dim, device=device, dtype=dtype))
        self.register_buffer("std", torch.ones(input_dim, device=device, dtype=dtype))


        enhanced_dim = input_dim + 3 + input_dim * 3  # 30 + 3 + 90 = 123
        self.feature_proj = nn.Linear(enhanced_dim, output_dim)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(seq_len, output_dim, device, dtype)

    def _create_positional_encoding(self, seq_len, d_model, device, dtype):
        pos_enc = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float().to(device) * (-np.log(10000.0) / d_model))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pos_enc.unsqueeze(0), requires_grad=False)

    # Остальной код остается без изменений

    def add_engineered_features(self, x):
        """
        Add domain-specific engineered features for grinding circuit.

        Args:
            x: Normalized input tensor [batch, seq_len, input_dim]

        Returns:
            Enhanced features tensor
        """
        batch_size, seq_len, _ = x.shape

        # Extract relevant features (example indices, adjust based on actual data)
        power_indices = [1, 2, 3, 4]  # Power consumption features
        current_indices = [5, 6, 7, 8]  # Current features
        feed_indices = [9, 10]  # Feed rate features
        water_indices = [13, 14, 15, 16, 17, 18]  # Water flow features

        # Extract feature groups
        power = x[:, :, power_indices]
        current = x[:, :, current_indices]
        feed = x[:, :, feed_indices]
        water = x[:, :, water_indices]

        # Create new features

        # 1. Power to feed ratio (energy efficiency)
        power_feed_ratio = power.mean(dim=-1, keepdim=True) / (feed.mean(dim=-1, keepdim=True) + 1e-6)

        # 2. Water to feed ratio (dilution)
        water_feed_ratio = water.mean(dim=-1, keepdim=True) / (feed.mean(dim=-1, keepdim=True) + 1e-6)

        # 3. Power to current ratio (electrical efficiency)
        power_current_ratio = power.mean(dim=-1, keepdim=True) / (current.mean(dim=-1, keepdim=True) + 1e-6)

        # 4. Rolling statistics (if sequence length allows)
        if seq_len >= 5:
            # Calculate rolling mean with kernel size 5
            rolling_mean = torch.zeros_like(x)
            for i in range(seq_len):
                start_idx = max(0, i - 2)
                end_idx = min(seq_len, i + 3)
                rolling_mean[:, i] = x[:, start_idx:end_idx].mean(dim=1)

            # Calculate rolling standard deviation
            rolling_std = torch.zeros_like(x)
            for i in range(seq_len):
                start_idx = max(0, i - 2)
                end_idx = min(seq_len, i + 3)
                rolling_std[:, i] = x[:, start_idx:end_idx].std(dim=1)
        else:
            rolling_mean = x
            rolling_std = torch.zeros_like(x)

        # 5. Temporal differences (rate of change)
        temp_diff = torch.zeros_like(x)
        temp_diff[:, 1:] = x[:, 1:] - x[:, :-1]

        # Concatenate original and new features
        enhanced = torch.cat([
            x,
            power_feed_ratio,
            water_feed_ratio,
            power_current_ratio,
            rolling_mean,
            rolling_std,
            temp_diff
        ], dim=-1)

        return enhanced

    def forward(self, x, update_stats=False):
        """
        Preprocess telemetry data.

        Args:
            x: Raw input tensor [batch, seq_len, input_dim]
            update_stats: Whether to update normalization statistics

        Returns:
            Preprocessed tensor [batch, seq_len, output_dim]
        """
        # Update statistics if in training mode
        if update_stats and self.training:
            with torch.no_grad():
                self.mean = 0.9 * self.mean + 0.1 * x.mean(dim=(0, 1))
                self.std = 0.9 * self.std + 0.1 * x.std(dim=(0, 1))

        # Normalize
        x_norm = (x - self.mean) / (self.std + 1e-6)

        # Add engineered features
        x_enhanced = self.add_engineered_features(x_norm)

        # Project to model dimension
        x_proj = self.feature_proj(x_enhanced)

        # Add positional encoding
        x_pos = x_proj + self.pos_encoding[:, :x_proj.size(1)]

        return x_pos

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_module import Mamba2
from mha_module import MHA
from mlp_module import GatedMLP
from config import HybridConfig

class HybridBlock(nn.Module):
    def __init__(self, dim, d_intermediate, dropout=0.1, norm_epsilon=1e-5, device='cuda', dtype=None, config=None):
        super().__init__()
        self.config = config or HybridConfig()  # Добавляем доступ к config

        factory_kwargs = {"device": device, "dtype": dtype}

        # Normalization layers
        self.norm1 = nn.LayerNorm(dim, eps=norm_epsilon, **factory_kwargs)
        self.norm2 = nn.LayerNorm(dim, eps=norm_epsilon, **factory_kwargs)

        # Mamba2 layer
        self.mamba = Mamba2(
            d_model=dim,
            d_state=self.config.d_state,
            d_conv=self.config.d_conv,
            expand=self.config.expand,
            headdim=self.config.headdim,  # Используем из config
            **factory_kwargs
        )

        # MHA layer
        self.mha = MHA(
            embed_dim=dim,
            num_heads=self.config.num_heads,
            num_heads_kv=self.config.num_heads_kv,
            dropout=dropout,
            **factory_kwargs
        )

        # Adaptive gating
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(dim // 2, 2, **factory_kwargs),
            nn.Softmax(dim=-1)
        )

        # MLP layer
        self.mlp = GatedMLP(
            in_features=dim,
            hidden_features=d_intermediate,
            out_features=dim,
            dropout=dropout,
            **factory_kwargs
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for Hybrid Block.

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # First normalization
        residual = x
        x = self.norm1(x)

        # Apply Mamba2 and MHA
        mamba_out = self.mamba(x)
        mha_out = self.mha(x)

        # Adaptive gating
        # Calculate sequence-wise features for gating
        mamba_feat = mamba_out.mean(dim=1)  # [batch, dim]
        mha_feat = mha_out.mean(dim=1)  # [batch, dim]
        combined = torch.cat([mamba_feat, mha_feat], dim=-1)  # [batch, 2*dim]

        # Compute gating weights
        weights = self.gate(combined)  # [batch, 2]

        # Apply weights to outputs
        mamba_weight = weights[:, 0].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
        mha_weight = weights[:, 1].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]

        x = mamba_weight * mamba_out + mha_weight * mha_out

        # Residual connection
        x = residual + self.dropout(x)

        # Second normalization and MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)

        # Residual connection
        x = residual + self.dropout(x)

        return x

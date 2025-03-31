import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            num_heads_kv=None,
            head_dim=None,
            dropout=0.1,
            causal=False,
            device='cuda',
            dtype=None,
    ):
        super().__init__()

        # Save parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        self.causal = causal
        self.dropout = dropout

        assert self.num_heads % self.num_heads_kv == 0, "num_heads must be divisible by num_heads_kv"

        # Compute head dimension
        if head_dim is None:
            assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
            self.head_dim = self.embed_dim // num_heads
        else:
            self.head_dim = head_dim

        # Compute dimensions
        self.kv_dim = self.head_dim * self.num_heads_kv
        self.q_dim = self.head_dim * self.num_heads
        self.inner_dim = self.head_dim * self.num_heads

        # QKV projection
        self.qkv_proj = nn.Linear(
            embed_dim,
            self.q_dim + 2 * self.kv_dim,
            bias=True,
            device=device,
            dtype=dtype
        )

        # Output projection
        self.out_proj = nn.Linear(
            self.inner_dim,
            embed_dim,
            bias=True,
            device=device,
            dtype=dtype
        )

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Scaling factor for attention
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        Forward pass for Multi-Head Attention.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(
            qkv,
            [self.q_dim, self.kv_dim, self.kv_dim],
            dim=-1
        )

        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads_kv, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads_kv, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat k and v for multi-query attention if needed
        if self.num_heads > self.num_heads_kv:
            repeats = self.num_heads // self.num_heads_kv
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            attn_weights.masked_fill_(mask, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and transpose back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.inner_dim)

        # Output projection and dropout
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output

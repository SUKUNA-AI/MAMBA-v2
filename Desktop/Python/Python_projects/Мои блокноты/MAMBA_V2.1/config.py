from dataclasses import dataclass

@dataclass
class HybridConfig:
    # Model architecture
    d_model: int = 256  # Hidden dimension
    n_layer: int = 6    # Number of hybrid blocks
    d_intermediate: int = 512  # Intermediate dimension for MLP
    num_heads: int = 8  # Number of attention heads (for MHA)
    num_heads_kv: int = 4  # Number of key/value heads (for MHA)

    # Input/output dimensions
    input_dim: int = 33  # Number of telemetry features
    output_dim: int = 2  # Number of sieve characteristics to predict

    # Temporal parameters
    seq_len: int = 60    # Sequence length (60 minutes of data)
    lab_feature_dim: int = 5  # Number of laboratory analysis features

    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    dropout: float = 0.1

    # Normalization parameters
    norm_epsilon: float = 1e-5
    fused_add_norm: bool = True
    residual_in_fp32: bool = False

    # Mamba parameters
    d_state: int = 128  # State dimension for Mamba
    d_conv: int = 4     # Convolution kernel size
    expand: int = 2     # Expansion factor
    headdim: int = 64   # Head dimension for Mamba (добавлено)

    # Uncertainty estimation
    estimate_uncertainty: bool = True

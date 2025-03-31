import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple, Any

from MHA import MHA
from MLP import GatedMLP
from MAMBAv2 import Mamba2

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
except ImportError:
    RMSNorm = nn.LayerNorm
    layer_norm_fn = None


class Block(pl.LightningModule):
    def __init__(
            self,
            dim: int,
            d_intermediate: int,
            layer_idx: int,
            norm_epsilon: float = 1e-5,
            learning_rate: float = 1e-3,
            fused_add_norm: bool = True,
            residual_in_fp32: bool = False,
            device=None,
            dtype=None,
            **mixer_kwargs
    ):
        """
        Гибридный блок, объединяющий Mamba2 и MHA с адаптивным гейтингом, нормализацией и MLP.

        Args:
            dim: Размерность входных данных.
            d_intermediate: Размер промежуточного слоя в MLP.
            layer_idx: Индекс слоя для передачи в миксеры.
            norm_epsilon: Эпсилон для нормализации.
            learning_rate: Скорость обучения.
            fused_add_norm: Использовать ли оптимизированную операцию Add+Norm.
            residual_in_fp32: Хранить ли остатки в float32.
            device: Устройство.
            dtype: Тип данных.
            **mixer_kwargs: Аргументы для миксеров (Mamba2 и MHA).
        """
        super().__init__()
        self.save_hyperparameters()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm and layer_norm_fn is not None

        # Два миксера: Mamba2 и MHA
        self.mamba = Mamba2(dim=dim, layer_idx=layer_idx, **mixer_kwargs, **factory_kwargs)
        self.mha = MHA(embed_dim=dim, layer_idx=layer_idx, **mixer_kwargs, **factory_kwargs)

        # Адаптивное объединение
        self.gating = nn.Sequential(
            nn.Linear(dim * 2, dim // 2, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(dim // 2, 2, **factory_kwargs),
            nn.Softmax(dim=-1)
        )

        # Нормализация
        self.norm1 = RMSNorm(dim, eps=norm_epsilon, **factory_kwargs)
        self.norm2 = RMSNorm(dim, eps=norm_epsilon, **factory_kwargs)

        # GatedMLP
        self.mlp = GatedMLP(
            in_features=dim,
            hidden_features=d_intermediate,
            out_features=dim,
            learning_rate=learning_rate,
            **factory_kwargs
        )

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(
            self,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            inference_params: Optional[Any] = None,
            **mixer_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход через гибридный блок.

        Args:
            hidden_states: Входной тензор (batch, seqlen, dim).
            residual: Остаточный тензор (batch, seqlen, dim) или None.
            inference_params: Параметры инференса для миксеров.
            **mixer_kwargs: Дополнительные аргументы для миксеров.

        Returns:
            Tuple[hidden_states, residual]: Обновленные состояния и остаток.
        """
        # Первая нормализация
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            normed_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            normed_states, residual = layer_norm_fn(
                hidden_states,
                self.norm1.weight,
                self.norm1.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm1.eps,
                is_rms_norm=True
            )

        # Применение Mamba2 и MHA
        mamba_out = self.mamba(normed_states, inference_params=inference_params, **mixer_kwargs)
        mha_out = self.mha(normed_states)

        # Адаптивное объединение
        combined = torch.cat([mamba_out.mean(dim=1), mha_out.mean(dim=1)], dim=-1)  # (batch, 2 * dim)
        weights = self.gating(combined).unsqueeze(1)  # (batch, 1, 2)
        stacked = torch.stack([mamba_out, mha_out], dim=-1)  # (batch, seq_len, dim, 2)
        merged = torch.einsum("bsdm,bsm->bsd", stacked, weights)  # (batch, seq_len, dim)

        # Вторая нормализация и MLP
        if not self.fused_add_norm:
            residual = merged + residual
            normed_merged = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            normed_merged, residual = layer_norm_fn(
                merged,
                self.norm2.weight,
                self.norm2.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm2.eps,
                is_rms_norm=True
            )

        hidden_states = self.mlp(normed_merged)
        residual = residual + self.dropout(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Выделяет кэш для инференса.

        Args:
            batch_size: Размер батча.
            max_seqlen: Максимальная длина последовательности.
            dtype: Тип данных.
            **kwargs: Дополнительные аргументы.

        Returns:
            Словарь с кэшем для Mamba2 и MHA.
        """
        return {
            "mamba": self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs),
            "mha": self.mha.allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs)
        }
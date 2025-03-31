import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Any

from Block_MAMBAv2 import Block  # Импорт переработанного Block из файла block.py

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except ImportError:
    RMSNorm = nn.LayerNorm

from dataclasses import dataclass

@dataclass
class HybridConfig:
    """
    Конфигурация для гибридной регрессионной модели.

    Attributes:
        d_model: Размерность скрытого состояния.
        n_layer: Количество слоёв (блоков).
        d_intermediate: Размер промежуточного слоя в GatedMLP.
        input_dim: Размерность входных данных (число признаков телеметрии).
        output_dim: Размерность выходных данных (число предсказываемых характеристик).
        norm_epsilon: Эпсилон для нормализации (RMSNorm).
        learning_rate: Скорость обучения.
        fused_add_norm: Использовать ли оптимизированную операцию Add+Norm.
        residual_in_fp32: Хранить ли остатки в float32.
    """
    d_model: int = 256
    n_layer: int = 6
    d_intermediate: int = 512
    input_dim: int = 10  # Пример: число признаков телеметрии
    output_dim: int = 1  # Пример: одна ситовая характеристика
    norm_epsilon: float = 1e-5
    learning_rate: float = 1e-3
    fused_add_norm: bool = True
    residual_in_fp32: bool = False


# Полная гибридная модель
class AdvancedHybridRegressionModel(pl.LightningModule):
    def __init__(
            self,
            config: 'HybridConfig',  # Принимаем объект конфигурации
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.save_hyperparameters(asdict(config))  # Сохраняем гиперпараметры из config
        factory_kwargs = {"device": device, "dtype": dtype}

        # Проекция входных данных
        self.input_proj = nn.Linear(config.input_dim, config.d_model, **factory_kwargs)

        # Стек гибридных блоков (используем переработанный Block)
        self.layers = nn.ModuleList(
            [
                Block(
                    dim=config.d_model,
                    d_intermediate=config.d_intermediate,
                    layer_idx=i,
                    norm_epsilon=config.norm_epsilon,
                    learning_rate=config.learning_rate,
                    fused_add_norm=config.fused_add_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    **factory_kwargs
                )
                for i in range(config.n_layer)
            ]
        )

        # Финальная нормализация
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_epsilon, **factory_kwargs)

        # Регрессионная голова
        self.regression_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2, **factory_kwargs),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.output_dim, **factory_kwargs)
        )

    def forward(self, inputs: torch.Tensor, inference_params: Optional[Any] = None) -> torch.Tensor:
        # inputs: (batch, seq_len, input_dim)
        hidden_states = self.input_proj(inputs)  # (batch, seq_len, d_model)
        residual = None

        # Проход через слои
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params)

        # Финальная нормализация
        hidden_states = self.norm_f(residual)

        # Взвешенное усреднение по последовательности
        attention_weights = torch.softmax(hidden_states.norm(dim=-1), dim=1)  # (batch, seq_len)
        pooled = torch.einsum("bsd,bs->bd", hidden_states, attention_weights)  # (batch, d_model)

        # Регрессионный выход
        outputs = self.regression_head(pooled)  # (batch, output_dim)
        return outputs

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Выделение кэша для инференса."""
        cache = {}
        for i, layer in enumerate(self.layers):
            cache[f"layer_{i}"] = layer.allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs)
        return cache


# Для сохранения гиперпараметров из dataclass в save_hyperparameters
from dataclasses import asdict
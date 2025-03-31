import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mlp_module import GatedMLP


class Mamba2(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2, headdim=64, bias=False, conv_bias=True,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, dt_limit=(0.0, float("inf")),
                 device='cuda', dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.d_inner = self.expand * self.d_model  # 512
        self.d_ssm = self.d_inner
        self.nheads = self.d_ssm // self.headdim  # 8

        factory_kwargs = {"device": device, "dtype": dtype}

        # Входная проекция: z (гейтинг), x (SSM), B, C, dt, mlp_in (для MLP)
        d_in_proj = self.d_inner + self.d_ssm + 2 * self.nheads * self.d_state + self.nheads + self.d_inner
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        # Свёртка только для x
        self.conv1d = nn.Conv1d(self.d_ssm, self.d_ssm, bias=conv_bias, kernel_size=d_conv, groups=self.d_ssm,
                                padding=d_conv - 1, **factory_kwargs)

        self.act = F.silu

        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nn.Parameter(dt.clamp(min=dt_init_floor))

        A = torch.empty(self.nheads, self.d_state, dtype=torch.float32, device=device).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device, dtype=dtype))

        self.mlp = GatedMLP(
            in_features=self.d_inner,
            hidden_features=self.d_inner * 2,
            out_features=self.d_inner,
            dropout=0.1,
            bias=bias,
            **factory_kwargs
        )

        self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u):
        batch, seqlen, _ = u.shape

        # Входная проекция
        z_x_b_c_dt_mlp = self.in_proj(u)
        z, x, B, C, dt, mlp_in = torch.split(
            z_x_b_c_dt_mlp,
            [self.d_inner, self.d_ssm, self.nheads * self.d_state, self.nheads * self.d_state, self.nheads,
             self.d_inner],
            dim=-1
        )

        # SSM путь
        # Обрабатываем x через свёртку
        x = x.transpose(1, 2)
        x = self.act(self.conv1d(x)[:, :, :seqlen])
        x = x.transpose(1, 2)  # [batch, seqlen, d_ssm]

        # Приводим B и C к форме [batch, seqlen, nheads, d_state]
        B = B.view(batch, seqlen, self.nheads, self.d_state)  # [64, 60, 8, 128]
        C = C.view(batch, seqlen, self.nheads, self.d_state)  # [64, 60, 8, 128]

        A = -torch.exp(self.A_log.float())  # [nheads, d_state]
        dt = F.softplus(dt + self.dt_bias)  # [batch, seqlen, nheads]
        dA = torch.exp(dt[..., None] * A[None, None, :, :])  # [batch, seqlen, nheads, d_state]

        # Параллельное сканирование
        # 1. Вычисляем кумулятивное произведение dA по оси времени (seqlen)
        dA_cumprod = torch.cumprod(dA, dim=1)  # [batch, seqlen, nheads, d_state]

        # 2. Вычисляем вклад B на каждом шаге
        # Для этого умножаем B на кумулятивное произведение dA с предыдущих шагов
        # Чтобы сделать это параллельно, создаём "сдвинутую" версию dA_cumprod
        dA_cumprod_shifted = torch.cat(
            [torch.ones(batch, 1, self.nheads, self.d_state, device=u.device, dtype=u.dtype), dA_cumprod[:, :-1]],
            dim=1
        )  # [batch, seqlen, nheads, d_state]

        # 3. Вычисляем состояния x_s
        # x_s[t] = sum_{i=0}^{t} (dA[t] * ... * dA[i+1]) * B[i]
        x_s = torch.cumsum(dA_cumprod_shifted * B, dim=1)  # [batch, seqlen, nheads, d_state]

        # 4. Вычисляем выход y
        y = torch.einsum('blnd,blnd->bln', x_s, C)  # [batch, seqlen, nheads]

        # 5. Расширяем y до d_inner и добавляем skip connection
        x = x.view(batch, seqlen, self.nheads, self.headdim)  # [batch, seqlen, nheads, headdim]
        y = y.repeat_interleave(self.headdim, dim=-1)  # [batch, seqlen, d_inner]
        y = y + x.view(batch, seqlen, -1) * self.D  # [batch, seqlen, d_inner]

        ssm_out = y * F.silu(z)  # [batch, seqlen, d_inner]

        # MLP путь
        mlp_out = self.mlp(mlp_in)  # [batch, seqlen, d_inner]

        # Комбинируем
        out = ssm_out + mlp_out  # [batch, seqlen, d_inner]
        out = self.norm(out)
        out = self.out_proj(out)  # [batch, seqlen, d_model]

        return out

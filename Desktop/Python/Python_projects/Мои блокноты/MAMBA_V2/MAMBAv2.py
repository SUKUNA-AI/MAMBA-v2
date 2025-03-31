import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

# Импорт кастомной нормализации RMSNormGated из mamba_ssm
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
# Импорты для оптимизированных операций SSM
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

class Mamba2(pl.LightningModule):
    def __init__(
            self,
            d_model,  # Размерность входных данных (например, размер скрытого состояния в последовательности)
            d_state=128,  # Размерность внутреннего состояния SSM (State Space Model)
            d_conv=4,  # Размер ядра свертки в одномерной причинной свертке
            conv_init=None,  # Начальная амплитуда для инициализации весов свертки (если None — стандартная инициализация)
            expand=2,  # Коэффициент расширения внутренней размерности (d_inner = expand * d_model)
            headdim=64,  # Размерность одной головы в SSM
            d_ssm=None,  # Размерность SSM-слоя, если None — равна d_inner
            ngroups=1,  # Количество групп для группировки состояний и параметров SSM
            A_init_range=(1, 16),  # Диапазон для инициализации матрицы A (динамика состояния в SSM)
            D_has_hdim=False,  # Флаг: имеет ли параметр D размерность по головам (headdim)
            rmsnorm=True,  # Использовать ли RMS нормализацию
            norm_before_gate=False,  # Применять нормализацию перед гейтингом (если True) или после (если False)
            dt_min=0.001,  # Минимальное значение шага дискретизации (dt) для SSM
            dt_max=0.1,  # Максимальное значение шага дискретизации (dt) для SSM
            dt_init_floor=1e-4,  # Минимальный порог для начальной инициализации dt
            dt_limit=(0.0, float("inf")),  # Ограничения на значения dt во время вычислений
            bias=False,  # Использовать ли bias в линейных слоях
            conv_bias=True,  # Использовать ли bias в сверточном слое
            chunk_size=256,  # Размер чанка для обработки последовательности в оптимизированном пути
            use_mem_eff_path=True,  # Использовать ли memory-efficient путь для инференса
            layer_idx=None,  # Индекс слоя (нужен для кэширования состояний в инференсе)
            device=None,  # Устройство (CPU/GPU), если None — выбирается автоматически
            dtype=None,  # Тип данных (например, torch.float32), если None — выбирается автоматически
            learning_rate=1e-3,  # Скорость обучения для оптимизатора в PyTorch Lightning
    ):
        # Инициализация родительского класса LightningModule
        super().__init__()

        # Сохранение параметров модели как атрибутов
        self.d_model = d_model  # Входная размерность модели
        self.d_state = d_state  # Размер внутреннего состояния SSM
        self.d_conv = d_conv  # Ширина окна причинной свертки
        self.conv_init = conv_init  # Начальная амплитуда для весов свертки (опционально)
        self.expand = expand  # Коэффициент расширения внутренней размерности
        self.headdim = headdim  # Размер одной головы в SSM
        # Размер SSM-слоя: либо d_inner (по умолчанию), либо заданное значение
        self.d_inner = self.expand * self.d_model  # Внутренняя размерность = expand * d_model (без деления на world_size)
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm  # Размер SSM
        self.ngroups = ngroups  # Количество групп (без деления на world_size)
        # Проверка: размер SSM должен делиться на размер головы без остатка
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim  # Число голов = d_ssm / headdim
        self.D_has_hdim = D_has_hdim  # Флаг зависимости D от размера головы
        self.rmsnorm = rmsnorm  # Флаг использования RMS нормализации
        self.norm_before_gate = norm_before_gate  # Порядок нормализации и гейтинга
        self.dt_limit = dt_limit  # Ограничения на шаг дискретизации
        self.activation = "silu"  # Функция активации (фиксирована как SiLU)
        self.chunk_size = chunk_size  # Размер чанка для обработки последовательностей
        self.use_mem_eff_path = use_mem_eff_path  # Флаг memory-efficient пути
        self.layer_idx = layer_idx  # Индекс слоя для кэширования
        self.learning_rate = learning_rate  # Скорость обучения

        # Словарь с аргументами для создания слоев (устройство и тип данных)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Размер входного проекционного слоя: 2*d_inner (z и x) + 2*ngroups*d_state (B и C) + nheads (dt)
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        # Стандартный линейный слой для проекции входных данных (без параллелизма)
        self.in_proj = nn.Linear(
            self.d_model,  # Входная размерность
            d_in_proj,  # Выходная размерность
            bias=bias,  # Использовать ли bias (по умолчанию False)
            **factory_kwargs  # Устройство и тип данных
        )

        # Размерность входа/выхода для свертки: d_ssm + 2*ngroups*d_state (x, B, C)
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        # Одномерная причинная свертка для обработки последовательности
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,  # Входные каналы
            out_channels=conv_dim,  # Выходные каналы (равны входным)
            bias=conv_bias,  # Использовать ли bias (по умолчанию True)
            kernel_size=d_conv,  # Размер ядра свертки
            groups=conv_dim,  # Групповая свертка: каждый канал независим
            padding=d_conv - 1,  # Паддинг для сохранения длины (причинность)
            **factory_kwargs  # Устройство и тип данных
        )
        # Если задан conv_init, инициализируем веса свертки равномерно
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        # Функция активации SiLU (Swish), применяется после свертки
        self.act = nn.SiLU()

        # Инициализация шага дискретизации (dt) для SSM
        # Генерируем случайные значения в логарифмическом диапазоне [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        # Ограничиваем dt снизу значением dt_init_floor
        dt = torch.clamp(dt, min=dt_init_floor)
        # Вычисляем обратное значение softplus для dt (для стабильности)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        # Регистрируем dt_bias как параметр модели
        self.dt_bias = nn.Parameter(inv_dt)
        # Отключаем регуляризацию весов для dt_bias
        self.dt_bias._no_weight_decay = True

        # Инициализация матрицы A (динамика состояния SSM)
        # Генерируем случайные значения в диапазоне A_init_range
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        # Берем логарифм от A для стабильности (экспоненцируется в forward)
        A_log = torch.log(A).to(dtype=dtype)
        # Регистрируем A_log как параметр модели
        self.A_log = nn.Parameter(A_log)
        # Отключаем регуляризацию весов для A_log
        self.A_log._no_weight_decay = True

        # Инициализация параметра D (skip-соединение в SSM)
        # Размер D зависит от D_has_hdim: либо d_ssm, либо nheads
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        # Отключаем регуляризацию весов для D
        self.D._no_weight_decay = True

        # Если включена RMS нормализация, создаем слой RMSNormGated
        if self.rmsnorm:
            self.norm = RMSNormGated(
                self.d_ssm,  # Размерность нормализации
                eps=1e-5,  # Малое значение для численной стабильности
                norm_before_gate=self.norm_before_gate,  # Порядок нормализации и гейтинга
                group_size=self.d_ssm // ngroups,  # Размер группы для нормализации
                **factory_kwargs  # Устройство и тип данных
            )

        # Выходной проекционный слой (стандартный nn.Linear, без параллелизма)
        self.out_proj = nn.Linear(
            self.d_inner,  # Входная размерность (внутренняя после SSM)
            self.d_model,  # Выходная размерность (возвращаем к исходной)
            bias=bias,  # Использовать ли bias
            **factory_kwargs  # Устройство и тип данных
        )

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        Прямой проход модели Mamba2.

        Аргументы:
            u: Входной тензор. Форма зависит от seqlen:
               - Если seqlen=None: (batch, seqlen, hidden_dim)
               - Если seqlen задан: (batch * seqlen, hidden_dim)
            seqlen: Длина последовательности. Если None, извлекается из формы u.
            seq_idx: Индексы последовательности (опционально, для переменной длины).
            cu_seqlens: Кумулятивные длины последовательностей (для переменной длины в инференсе).
            inference_params: Параметры инференса (например, кэш состояний). Если None, используется тренировочный режим.

        Возвращает:
            Тензор той же формы, что и входной u.
        """
        # Сохраняем исходное значение seqlen для проверки позже
        seqlen_og = seqlen

        # Определяем размеры входного тензора в зависимости от seqlen
        if seqlen is None:
            # Если seqlen не задан, предполагаем, что u имеет форму (batch, seqlen, dim)
            batch, seqlen, dim = u.shape
        else:
            # Если seqlen задан, u имеет плоскую форму (batch * seqlen, dim)
            batch_seqlen, dim = u.shape
            # Вычисляем размер батча, разделив общую длину на длину последовательности
            batch = batch_seqlen // seqlen

        # Инициализируем состояния для свертки и SSM как None (по умолчанию для тренировочного режима)
        conv_state, ssm_state = None, None

        # Проверяем, переданы ли параметры инференса (используется для генерации или пошагового инференса)
        if inference_params is not None:
        # Определяем размер батча для инференса
        # Если cu_seqlens задан (переменная длина), берем его размер минус 1 (число последовательностей)
        inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
        # Извлекаем состояния свертки и SSM из кэша (метод _get_states_from_cache)
        conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
        # Если смещение последовательности > 0, это пошаговый инференс (например, генерация)
        if inference_params.seqlen_offset > 0:
            # Вызываем метод step для обработки одного токена и обновления состояний
            out, _, _ = self.step(u, conv_state, ssm_state)
            # Возвращаем результат без дальнейшей обработки
            return out

        # Проекция входных данных через линейный слой in_proj
        # u: (batch, seqlen, d_model) или (batch * seqlen, d_model)
        # zxbcdt: (batch, seqlen, d_in_proj) или (batch * seqlen, d_in_proj)
        zxbcdt = self.in_proj(u)

        # Если seqlen был задан изначально, преобразуем zxbcdt в форму (batch, seqlen, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = zxbcdt.view(batch, seqlen, -1)  # -1 автоматически вычисляет d_in_proj

        # Вычисляем матрицу A для SSM (динамика состояния)
        # A_log — обучаемый параметр, экспоненцируем его и берем с минусом (A всегда отрицательное)
        A = -torch.exp(self.A_log.float())  # Форма: (nheads,)

        # Устанавливаем дополнительные аргументы для ограничения шага dt (если dt_limit не бесконечен)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # Проверяем, использовать ли memory-efficient путь (оптимизированный для памяти и скорости)
        # Условие: включен флаг use_mem_eff_path и нет параметров инференса (тренировочный режим)
        if self.use_mem_eff_path and inference_params is None:
            # Вызываем оптимизированную функцию mamba_split_conv1d_scan_combined
            # Эта функция объединяет свертку, SSM и выходную проекцию в один вызов
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,  # Входной тензор: (batch, seqlen, d_in_proj)
                self.conv1d.weight.squeeze(1),  # Веса свертки: (conv_dim, kernel_size)
                self.conv1d.bias,  # Смещение свертки: (conv_dim,) или None
                self.dt_bias,  # Смещение шага dt: (nheads,)
                A,  # Матрица A для SSM: (nheads,)
                D=self.D.view(self.nheads, self.headdim) if self.D_has_hdim else self.D,  # Параметр D: (nheads, headdim) или (nheads,)
                chunk_size=self.chunk_size,  # Размер чанка для обработки последовательности
                seq_idx=seq_idx,  # Индексы последовательности (None для фиксированной длины)
                activation=self.activation,  # Функция активации: "silu"
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,  # Веса нормализации или None
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,  # Эпсилон для нормализации
                outproj_weight=self.out_proj.weight,  # Веса выходного слоя: (d_model, d_inner)
                outproj_bias=self.out_proj.bias,  # Смещение выходного слоя: (d_model,) или None
                headdim=None if self.D_has_hdim else self.headdim,  # Размер головы (None, если D_has_hdim)
                ngroups=self.ngroups,  # Число групп для SSM
                norm_before_gate=self.norm_before_gate,  # Порядок нормализации и гейтинга
                **dt_limit_kwargs,  # Ограничения на dt
            )
            # Если seqlen задан изначально, преобразуем выход в плоскую форму (batch * seqlen, d_model)
            if seqlen_og is not None:
                out = out.view(batch * seqlen, -1)  # -1 автоматически вычисляет d_model
            # Убрана часть с reduce_scatter/all_reduce, так как нет параллелизма
        else:
            # Стандартный путь (более детализированный и гибкий, но менее эффективный по памяти)

            # Вычисляем размер MLP-компоненты (z0 и x0), вычитая остальные части из общей размерности
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            # Разделяем zxbcdt на компоненты: z0, x0 (MLP), z (гейтинг), xBC (свертка), dt (шаг SSM)
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1  # Разделение по последней размерности
            )

            # Обновляем состояние свертки, если оно передано (для инференса)
            if conv_state is not None:
                if cu_seqlens is None:
                    # Для фиксированной длины: транспонируем xBC в (batch, conv_dim, seqlen)
                    xBC_t = xBC.transpose(1, 2)
                    # Паддинг или обрезка до нужной длины окна свертки, обновляем состояние
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))
                else:
                    # Для переменной длины: проверяем наличие нужной функции
                    assert causal_conv1d_varlen_states is not None, "Требуется пакет causal_conv1d для переменной длины"
                    assert batch == 1, "Переменная длина поддерживает только batch=1"
                    # Вычисляем состояния для переменной длины
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0),  # Убираем batch=1: (seqlen, conv_dim)
                        cu_seqlens,  # Кумулятивные длины
                        state_len=conv_state.shape[-1]  # Длина состояния
                    )
                    conv_state.copy_(conv_varlen_states)

            # Применяем одномерную причинную свертку
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                # Если нет оптимизированной функции или активация не поддерживается
                assert seq_idx is None, "Переменная длина требует causal_conv1d_fn"
                # Транспонируем xBC в (batch, conv_dim, seqlen), применяем свертку, обрезаем паддинг
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )   # Форма: (batch, seqlen, conv_dim)
            else:
                # Используем оптимизированную функцию causal_conv1d_fn
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),  # (batch, conv_dim, seqlen)
                    self.conv1d.weight.squeeze(1),  # (conv_dim, kernel_size)
                    bias=self.conv1d.bias,  # (conv_dim,) или None
                    activation=self.activation,  # "silu"
                    seq_idx=seq_idx,  # Индексы последовательности
                ).transpose(1, 2)  # Возвращаем в (batch, seqlen, conv_dim)

            # Разделяем xBC на x (вход SSM), B и C (параметры SSM)
            x, B, C = torch.split(
                xBC,
                [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                dim=-1
            )  # Формы: (batch, seqlen, d_ssm), (batch, seqlen, ngroups * d_state), (batch, seqlen, ngroups * d_state)

            # Применяем SSM через mamba_chunk_scan_combined
            y = mamba_chunk_scan_combined(
                x.view(batch, seqlen, self.nheads, self.headdim),  # Преобразуем x в (batch, seqlen, nheads, headdim)
                dt,  # Шаг дискретизации: (batch, seqlen, nheads)
                A,  # Матрица A: (nheads,)
                B.view(batch, seqlen, self.ngroups, self.d_state),  # Преобразуем B в (batch, seqlen, ngroups, d_state)
                C.view(batch, seqlen, self.ngroups, self.d_state),  # Преобразуем C в (batch, seqlen, ngroups, d_state)
                chunk_size=self.chunk_size,  # Размер чанка
                D=self.D.view(self.nheads, self.headdim) if self.D_has_hdim else self.D,  # D: (nheads, headdim) или (nheads,)
                z=z.view(batch, seqlen, self.nheads, self.headdim) if not self.rmsnorm else None,  # z для гейтинга или None
                dt_bias=self.dt_bias,  # Смещение dt: (nheads,)
                dt_softplus=True,  # Применять softplus к dt
                seq_idx=seq_idx,  # Индексы последовательности
                cu_seqlens=cu_seqlens,  # Кумулятивные длины
                **dt_limit_kwargs,  # Ограничения dt
                return_final_states=ssm_state is not None,  # Возвращать финальные состояния, если ssm_state задан
                return_varlen_states=cu_seqlens is not None and inference_params is not None,  # Возвращать состояния переменной длины
            )

            # Если ssm_state задан, обрабатываем возвращаемые состояния
            if ssm_state is not None:
                y, last_state, *rest = y  # y — выход, last_state — последнее состояние
                if cu_seqlens is None:
                    # Для фиксированной длины: обновляем ssm_state последним состоянием
                    ssm_state.copy_(last_state)
                else:
                    # Для переменной длины: обновляем ssm_state состояниями переменной длины
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)

            # Преобразуем y обратно в (batch, seqlen, d_ssm)
            y = y.view(batch, seqlen, -1)

            # Применяем RMS нормализацию, если включена
            if self.rmsnorm:
                y = self.norm(y, z)  # y: (batch, seqlen, d_ssm), z: (batch, seqlen, d_ssm)

            # Если есть MLP-компонента (d_mlp > 0), добавляем её к выходу
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)  # Объединяем (batch, seqlen, d_mlp + d_ssm)

            # Если seqlen задан изначально, преобразуем y в плоскую форму (batch * seqlen, d_inner)
            if seqlen_og is not None:
                y = y.view(batch * seqlen, -1)

            # Применяем выходной линейный слой
            out = self.out_proj(y)  # Форма: (batch, seqlen, d_model) или (batch * seqlen, d_model)

        # Возвращаем итоговый результат
        return out


    def step(self, hidden_states, conv_state, ssm_state):
        """
        Пошаговый проход модели Mamba2 для обработки одного токена (используется в инференсе).

        Аргументы:
            hidden_states: Входной тензор формы (batch, 1, d_model) — один токен на батч.
            conv_state: Состояние свертки формы (batch, conv_dim, d_conv) для причинной свертки.
            ssm_state: Состояние SSM формы (batch, nheads, headdim, d_state) для динамики SSM.

        Возвращает:
            Кортеж (out, conv_state, ssm_state):
                - out: Выходной тензор формы (batch, 1, d_model).
                - conv_state: Обновленное состояние свертки.
                - ssm_state: Обновленное состояние SSM.
        """
        # Сохраняем тип данных входного тензора для приведения результатов
        dtype = hidden_states.dtype

        # Проверяем, что входной тензор содержит ровно один токен по оси последовательности
        assert hidden_states.shape[1] == 1, "Метод step поддерживает только один токен за раз"

        # Убираем размерность последовательности (1) и проецируем вход через линейный слой
        # hidden_states: (batch, 1, d_model) -> squeeze(1) -> (batch, d_model)
        # zxbcdt: (batch, d_in_proj), где d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads
        zxbcdt = self.in_proj(hidden_states.squeeze(1))

        # Вычисляем размер MLP-компоненты (z0 и x0), вычитая остальные части из общей размерности
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2

        # Разделяем zxbcdt на компоненты:
        # - z0, x0: MLP-компоненты (batch, d_mlp)
        # - z: Гейтинг (batch, d_ssm)
        # - xBC: Вход для свертки (batch, d_ssm + 2 * ngroups * d_state)
        # - dt: Шаг дискретизации (batch, nheads)
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1  # Разделение по последней размерности
        )

        # Обновляем состояние свертки и применяем свертку
        if causal_conv1d_update is None:
            # Если нет оптимизированной функции causal_conv1d_update, используем ручную реализацию
            # Сдвигаем состояние свертки влево (удаляем самый старый токен)
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Форма: (batch, conv_dim, d_conv)
            # Добавляем новый токен в конец состояния
            conv_state[:, :, -1] = xBC  # xBC: (batch, conv_dim)
            # Вычисляем свертку: поэлементное умножение состояния на веса и суммирование по оси окна
            xBC = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)  # Форма: (batch, conv_dim)
            # Если есть смещение свертки, добавляем его
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias  # Форма: (batch, conv_dim)
            # Применяем активацию SiLU и приводим к исходному типу данных
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            # Используем оптимизированную функцию causal_conv1d_update
            xBC = causal_conv1d_update(
                xBC,  # Входной токен: (batch, conv_dim)
                conv_state,  # Текущее состояние: (batch, conv_dim, d_conv)
                self.conv1d.weight.squeeze(1),  # Веса свертки: (conv_dim, d_conv)
                self.conv1d.bias,  # Смещение: (conv_dim,) или None
                self.activation  # "silu"
            )  # Результат: (batch, conv_dim)

        # Разделяем xBC на компоненты для SSM:
        # - x: Вход SSM (batch, d_ssm)
        # - B: Параметр B (batch, ngroups * d_state)
        # - C: Параметр C (batch, ngroups * d_state)
        x, B, C = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )

        # Вычисляем матрицу A для SSM (динамика состояния)
        # A_log — обучаемый параметр, экспоненцируем и берем с минусом
        A = -torch.exp(self.A_log.float())  # Форма: (nheads,)

        # Выполняем шаг SSM
        if selective_state_update is None:
            # Если нет оптимизированной функции selective_state_update, используем ручную реализацию
            assert self.ngroups == 1, "Ручная реализация SSM требует ngroups=1"
            # Применяем softplus к dt с учетом смещения, приводим тип данных
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # Форма: (batch, nheads)
            # Вычисляем экспоненту A с учетом dt (дискретизация динамики)
            dA = torch.exp(dt * A)  # Форма: (batch, nheads)
            # Преобразуем x в форму (batch, nheads, headdim)
            x = x.view(-1, self.nheads, self.headdim)
            # Вычисляем обновление состояния: dt * B * x (тензорное произведение)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)  # Форма: (batch, nheads, d_state, headdim)
            # Обновляем состояние SSM: затухание старого состояния + новое слагаемое
            ssm_state.copy_(ssm_state * dA.unsqueeze(-1).unsqueeze(-1) + dBx)  # Форма: (batch, nheads, headdim, d_state)
            # Вычисляем выход SSM: скалярное произведение состояния с C
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)  # Форма: (batch, nheads, headdim)
            # Добавляем skip-соединение через параметр D
            y = y + self.D.to(dtype).unsqueeze(-1) * x  # Форма: (batch, nheads, headdim)
            # Преобразуем y в плоскую форму (batch, nheads * headdim)
            y = y.view(-1, self.nheads * self.headdim)
            # Если нормализация отключена, применяем гейтинг через z
            if not self.rmsnorm:
                y = y * self.act(z)  # z: (batch, d_ssm)
        else:
            # Используем оптимизированную функцию selective_state_update
            # Расширяем A до (nheads, headdim, d_state)
            A = A.unsqueeze(1).unsqueeze(2).expand(-1, self.headdim, self.d_state)
            # Расширяем dt до (batch, nheads, headdim)
            dt = dt.unsqueeze(-1).expand(-1, -1, self.headdim)
            # Расширяем dt_bias до (nheads, headdim)
            dt_bias = self.dt_bias.unsqueeze(-1).expand(-1, self.headdim)
            # Расширяем D до (nheads, headdim)
            D = self.D.unsqueeze(-1).expand(-1, self.headdim)
            # Преобразуем B и C в (batch, ngroups, d_state)
            B = B.view(-1, self.ngroups, self.d_state)
            C = C.view(-1, self.ngroups, self.d_state)
            # Преобразуем x в (batch, nheads, headdim)
            x_reshaped = x.view(-1, self.nheads, self.headdim)
            # Если нормализация отключена, преобразуем z для гейтинга
            if not self.rmsnorm:
                z = z.view(-1, self.nheads, self.headdim)
            # Выполняем шаг SSM с обновлением состояния
            y = selective_state_update(
                ssm_state,  # Текущее состояние: (batch, nheads, headdim, d_state)
                x_reshaped,  # Вход: (batch, nheads, headdim)
                dt,  # Шаг: (batch, nheads, headdim)
                A,  # Динамика: (nheads, headdim, d_state)
                B,  # Параметр B: (batch, ngroups, d_state)
                C,  # Параметр C: (batch, ngroups, d_state)
                D,  # Skip-соединение: (nheads, headdim)
                z=z if not self.rmsnorm else None,  # Гейтинг или None
                dt_bias=dt_bias,  # Смещение dt: (nheads, headdim)
                dt_softplus=True  # Применять softplus к dt
            )  # Результат: (batch, nheads * headdim)
            # Преобразуем y в плоскую форму
            y = y.view(-1, self.nheads * self.headdim)

        # Применяем RMS нормализацию, если включена
        if self.rmsnorm:
            y = self.norm(y, z)  # y: (batch, d_ssm), z: (batch, d_ssm)

        # Если есть MLP-компонента, добавляем её к выходу
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)  # Форма: (batch, d_mlp + d_ssm)

        # Применяем выходной линейный слой
        out = self.out_proj(y)  # Форма: (batch, d_model)

        # Добавляем размерность последовательности (1) и возвращаем результат с обновленными состояниями
        return out.unsqueeze(1), conv_state, ssm_state  # Форма out: (batch, 1, d_model)


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Выделяет память для кэша состояний в инференсе.

        Аргументы:
            batch_size: Размер батча.
            max_seqlen: Максимальная длина последовательности (не используется здесь напрямую).
            dtype: Тип данных для кэша (если None, используется тип слоев).
            **kwargs: Дополнительные аргументы (игнорируются).

        Возвращает:
            Кортеж (conv_state, ssm_state):
                - conv_state: Состояние свертки формы (batch_size, conv_dim, d_conv).
                - ssm_state: Состояние SSM формы (batch_size, nheads, headdim, d_state).
        """
        # Определяем устройство на основе весов выходного слоя
        device = self.out_proj.weight.device

        # Устанавливаем тип данных для состояния свертки (по умолчанию — тип весов conv1d)
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        # Создаем тензор нулей для состояния свертки
        # Форма: (batch_size, d_conv, conv_dim), затем транспонируем в (batch_size, conv_dim, d_conv)
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0],  # conv_dim = d_ssm + 2 * ngroups * d_state
            device=device,
            dtype=conv_dtype
        ).transpose(1, 2)

        # Устанавливаем тип данных для состояния SSM (по умолчанию — тип весов in_proj)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        # Создаем тензор нулей для состояния SSM
        # Форма: (batch_size, nheads, headdim, d_state)
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state,
            device=device,
            dtype=ssm_dtype
        )

        # Возвращаем созданные состояния
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        """
        Извлекает или создает состояния из кэша для инференса.

        Аргументы:
            inference_params: Объект с параметрами инференса, содержащий словарь key_value_memory_dict.
            batch_size: Размер батча.
            initialize_states: Флаг для сброса существующих состояний в нули (по умолчанию False).

        Возвращает:
            Кортеж (conv_state, ssm_state):
                - conv_state: Состояние свертки формы (batch_size, conv_dim, d_conv).
                - ssm_state: Состояние SSM формы (batch_size, nheads, headdim, d_state).
        """
        # Проверяем, что индекс слоя задан (нужен для кэширования)
        assert self.layer_idx is not None, "layer_idx должен быть задан для кэширования"

        # Проверяем, есть ли состояния для текущего слоя в кэше
        if self.layer_idx not in inference_params.key_value_memory_dict:
            # Если нет, создаем новые состояния
            # Состояние свертки: (batch_size, conv_dim, d_conv)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],  # conv_dim = d_ssm + 2 * ngroups * d_state
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            # Состояние SSM: (batch_size, nheads, headdim, d_state)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            # Сохраняем новые состояния в кэш
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            # Если состояния есть, извлекаем их из кэша
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # Если задан флаг initialize_states, сбрасываем состояния в нули
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()

        # Возвращаем состояния
        return conv_state, ssm_state

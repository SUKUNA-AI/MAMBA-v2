import torch
from typing import Callable

def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool) -> Callable:
    """Декоратор для поддержки AMP в PyTorch."""
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = "cuda"  # Указываем устройство для AMP
        return dec(*args, **kwargs)
    return decorator

# Проверяем наличие torch.amp и выбираем соответствующие декораторы
if hasattr(torch, "amp") and hasattr(torch.amp, "custom_fwd"):
    # Современный PyTorch (torch.amp доступен)
    deprecated = True
    from custom_amp_decorator.amp import custom_fwd, custom_bwd
else:
    # Старый PyTorch (torch.cuda.amp)
    deprecated = False
    from custom_amp_decorator.cuda.amp import custom_fwd, custom_bwd

# Применяем декоратор для совместимости
custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)
import torch
from typing import Iterable

def clip(
    params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6
) -> None:
    params = list(params)
    norm = 0.0
    for p in params:
        if p.grad is None:
            continue
        norm += torch.sum(p.grad * p.grad)
    norm = norm.sqrt()
    if norm < max_norm:
        return
    scale = max_norm / (norm + eps)
    for p in params:
        if p.grad is None:
            continue
        p.grad *= scale

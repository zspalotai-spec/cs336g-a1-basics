from einops import reduce
import torch
from typing import Iterable

from cs336_basics import timer


def compute_norm(params):
    norm = None
    for p in params:
        if p.grad is None:
            continue
        n = reduce(p.grad.pow(2), "... -> 1", "sum")
        if norm is None:
            norm = n
        else:
            norm += n
    return norm.sqrt()


def compute_scale(norm, max_norm, eps):
    return torch.min(norm, max_norm) / (norm + eps)


def scale_params(params, scale):
    for p in params:
        if p.grad is None:
            continue
        p.grad *= scale


def clip(params: Iterable[torch.nn.Parameter], max_norm, eps: float = 1e-6):
    norm = timer.measure("gradient_clipping_compute_norm", lambda: compute_norm(params))
    scale = timer.measure(
        "gradient_clipping_compute_scale", lambda: compute_scale(norm, max_norm, eps)
    )
    timer.measure("gradient_clipping_scale_params", lambda: scale_params(params, scale))

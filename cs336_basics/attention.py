from einops import einsum, rearrange
import math
import torch

from cs336_basics import softmax


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    q_size = q.size()
    d_k = q_size[-1]
    n = q_size[-2]
    qkt = einsum(q, k, "... n d_k, ... m d_k -> ... n m")
    qkt_normed = qkt / math.sqrt(d_k)
    if mask is not None:
        qkt_normed[mask.logical_not()] = -math.inf
    qkt_normed = softmax.softmax(qkt_normed, -1)
    out = einsum(qkt_normed, v, "... n m, ... m d_v -> ... n d_v")
    return out

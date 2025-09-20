from einops import einsum, rearrange
import math
import torch

from cs336_basics import softmax
from cs336_basics import timer


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    timer.start("attention_size_compute")
    q_size = q.size()
    d_k = q_size[-1]
    timer.update("attention_size_compute")
    k_normed = timer.measure("k_normed", lambda: k * math.sqrt(1.0 / d_k))
    qkt_normed = timer.measure(
        "attention_qkt", lambda: einsum(q, k_normed, "... n d_k, ... m d_k -> ... n m")
    )
    #if mask is not None:
    #    mask_not = mask.logical_not().float()
    #    mask = mask.float()
    #    qkt_normed = timer.measure(
    #        "masking", lambda: qkt_normed * mask + mask_not * torch.div(-1.0, mask)
    #    )
    qkt_normed = timer.measure(
        "attention_softmax", lambda: softmax.softmax(qkt_normed, -1, mask=mask)
    )
    out = timer.measure(
        "attention_out",
        lambda: einsum(qkt_normed, v, "... n m, ... m d_v -> ... n d_v"),
    )
    return out

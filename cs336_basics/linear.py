from einops import einsum
import math
import torch
import torch.nn as nn


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_matrices: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_matrices = num_matrices
        self.device = device
        self.dtype = dtype
        std2 = 2.0 / (in_features + out_features)
        std = math.sqrt(std2)
        if num_matrices == 1:
            m_size = (out_features, in_features)
        else:
            m_size = (num_matrices, out_features, in_features)
        W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(m_size, device=device, dtype=dtype),
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )
        self.register_parameter("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       if self.num_matrices == 1:
           return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
       else:
           return einsum(x, self.W, "... d_in, n d_out d_in -> n ... d_out")

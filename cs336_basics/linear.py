from einops import einsum
import math
import torch
import torch.nn as nn


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        std2 = 2.0 / (in_features + out_features)
        std = math.sqrt(std2)
        W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((out_features, in_features), dtype=dtype),
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )
        self.register_parameter("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")

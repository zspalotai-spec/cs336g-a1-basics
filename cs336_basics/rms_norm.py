from einops import reduce
import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        g = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
        self.register_parameter("g", g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_size = x.size()
        gain_size = [1]*len(orig_size)
        gain_size[-1] = self.d_model
        mean_size = list(orig_size)
        mean_size[-1] = 1
        x = x.to(torch.float32)
        m = x.mul(x)
        m = reduce(m,'... d_in -> ...', 'mean')
        m = (m+self.eps).sqrt()
        x = x.mul(self.g.reshape(gain_size)).div(m.reshape(mean_size))
        return x.to(orig_dtype)

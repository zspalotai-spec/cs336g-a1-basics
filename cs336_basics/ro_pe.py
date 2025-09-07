from einops import einsum, rearrange
import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        i = torch.arange(0, max_seq_len)
        k = torch.arange(0, d_k // 2)
        theta_ik = torch.outer(i, torch.pow(theta, -2 * k  / d_k))
        rot_ik = torch.exp(1j * theta_ik)
        self.register_buffer("rot_ik", rot_ik, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_pos_size = token_positions.size()
        rotation_size = list(token_pos_size) + [self.d_k // 2]
        rotation_size = [1] * (len(x.size()) - len(rotation_size)) + rotation_size
        token_positions = token_positions.reshape((-1,))
        act_rot_ik = self.rot_ik.index_select(0, token_positions).reshape(rotation_size)
        complex_x = torch.view_as_complex(rearrange(x, "... (d d1) -> ... d d1", d1=2))
        rotated_as_real = torch.view_as_real(complex_x.mul(act_rot_ik))
        out = rearrange(rotated_as_real, "... d d1 -> ... (d d1)")
        return out

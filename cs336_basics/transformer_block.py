import torch
from torch import nn

from cs336_basics import causal_multi_head_self_attention
from cs336_basics import rms_norm
from cs336_basics import ro_pe
from cs336_basics import swi_glu
from cs336_basics import timer


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: ro_pe.RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.has_rope = rope is not None
        self.device = device
        self.dtype = dtype
        if self.has_rope:
            indices = torch.arange(0, rope.max_seq_len, device=device)
            self.register_buffer("indices", indices, persistent=False)
        self.add_module(
            "attn_rms_norm", rms_norm.RMSNorm(d_model, device=device, dtype=dtype)
        )
        self.add_module(
            "ff_rms_norm", rms_norm.RMSNorm(d_model, device=device, dtype=dtype)
        )
        self.add_module(
            "attn",
            causal_multi_head_self_attention.CausalMultiHeadSelfAttention(
                d_model, num_heads, rope=rope, device=device, dtype=dtype
            ),
        )
        self.add_module("ff", swi_glu.SwiGlu(d_model, d_ff, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = None
        if self.has_rope:
            x_size = x.size()
            if x_size[-2] == self.attn.rope.max_seq_len:
                indices = self.indices
            else:
                indices = timer.measure(
                    "get_rope_indices", lambda: self.indices[0 : x.size()[-2]]
                )
        x_normed = timer.measure(
            "block_input_rms", lambda: self.attn_rms_norm.forward(x)
        )
        y = timer.measure("attention", lambda: x + self.attn.forward(x_normed, indices))
        y_normed = timer.measure("block_ff_rms", lambda: self.ff_rms_norm.forward(y))
        return timer.measure("block_ff", lambda: y + self.ff.forward(y_normed))

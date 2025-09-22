from einops import rearrange
from torch import nn
import torch

from cs336_basics import attention
from cs336_basics import linear
from cs336_basics import ro_pe
from cs336_basics import timer


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: ro_pe.RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.has_rope = rope is not None
        if self.has_rope:
            self.add_module("rope", rope)
        qkv = linear.Linear(d_model, d_model, 3, device=device, dtype=dtype)
        self.add_module("qkv", qkv)
        wo = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.add_module("wo", wo)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        qkvx = timer.measure("attn_qkv", lambda: self.qkv.forward(x))
        qkvx_heads = timer.measure(
            "attn_qkv_headsreshape",
            lambda: rearrange(
                qkvx, "... seq_len (h d_h)-> ... h seq_len d_h", h=self.num_heads
            ),
        )
        if self.has_rope:
            qkvx_heads[0] = timer.measure(
                "attn_rope", lambda: self.rope(qkvx_heads[0], token_positions)
            )
            qkvx_heads[1] = timer.measure(
                "attn_rope", lambda: self.rope(qkvx_heads[1], token_positions)
            )
        timer.start("attn_size_compute")
        x_size = x.size()
        seq_len = x_size[-2]
        mask_size = list(x_size[:-2]) + [self.num_heads, seq_len, seq_len]
        timer.update("attn_size_compute")
        # mask = timer.measure(
        #    "attn_create_mask",
        #    lambda: torch.tril(
        #        torch.ones(mask_size, device=x.device)
        #    ),
        # )
        qkvx_heads_attention = timer.measure(
            "attn_attention",
            lambda: attention.causal_attention(
                qkvx_heads[0], qkvx_heads[1], qkvx_heads[2]
            ),
        )
        qkvx_attention = timer.measure(
            "attn_attention_dmodelreshape",
            lambda: rearrange(
                qkvx_heads_attention, "... h seq_len d_h -> ... seq_len (h d_h)"
            ),
        )
        return timer.measure("attn_ff_out", lambda: self.wo.forward(qkvx_attention))

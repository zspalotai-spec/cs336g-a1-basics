import torch
from torch import nn

from cs336_basics import embedding
from cs336_basics import linear
from cs336_basics import rms_norm
from cs336_basics import ro_pe
from cs336_basics import transformer_block


class TransformerLm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.add_module(
            "rope",
            ro_pe.RotaryPositionalEmbedding(
                rope_theta,
                d_model // num_heads,
                context_length,
                device=device,
                dtype=dtype,
            ),
        )
        self.add_module(
            "embed",
            embedding.Embedding(vocab_size, d_model, device=device, dtype=dtype),
        )
        for l in range(num_layers):
            self.add_module(
                f"layer{l}",
                transformer_block.TransformerBlock(
                    d_model, num_heads, d_ff, self.rope, device=device, dtype=dtype
                ),
            )
        self.add_module(
            "final_rms_norm", rms_norm.RMSNorm(d_model, device=device, dtype=dtype)
        )
        self.add_module(
            "final_linear",
            linear.Linear(d_model, vocab_size, device=device, dtype=dtype),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        x = self.embed.forward(indices)
        for l in range(self.num_layers):
            x = self.get_submodule(f"layer{l}").forward(x)
        x = self.final_rms_norm.forward(x)
        y = self.final_linear.forward(x)
        return y

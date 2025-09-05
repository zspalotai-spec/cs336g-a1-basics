import math
import torch
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), dtype=dtype),
                mean=0.0,
                std=1,
                a=-3,
                b=3,
            )
        )
        self.register_parameter("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.size()
        new_size = list(orig_size)
        new_size.append(self.embedding_dim)
        embedded = torch.index_select(self.W, 0, torch.reshape(x, (-1,)))
        return torch.reshape(embedded, new_size)

import math
import torch
import torch.nn as nn

from cs336_basics import linear

def silu(in_features: torch.Tensor) -> torch.Tensor:
    return in_features.mul(torch.sigmoid(in_features))

class SwiGlu(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        if not hidden_features:
            hidden_features = math.ceil(in_features*8/3/64)*64
        self.hidden_features = hidden_features
        self.device = device
        self.dtype = dtype
        lin1 = linear.Linear(in_features, hidden_features)
        lin2 = linear.Linear(hidden_features, in_features)
        lin3 = linear.Linear(in_features, hidden_features)
        self.add_module('lin1', lin1)
        self.add_module('lin2', lin2)
        self.add_module('lin3', lin3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w3x = self.lin3.forward(x)
        w1x = self.lin1.forward(x)
        siluw1x = silu(w1x)
        siluw1xw3x = siluw1x.mul(w3x)
        return self.lin2.forward(siluw1xw3x)

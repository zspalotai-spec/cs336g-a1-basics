from einops import reduce
import torch

def softmax(x: torch.Tensor, dim: int, mask=None) -> torch.Tensor:
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_normed = x - x_max
    x_exp = torch.exp(x_normed)
    if mask is not None:
        x_exp = x_exp * mask
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    x_out = torch.div(x_exp, x_exp_sum)
    return x_out

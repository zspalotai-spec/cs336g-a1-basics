from einops import reduce
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_normed = x - x_max
    x_exp = torch.exp(x_normed)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    x_out = x_exp / x_exp_sum
    return x_out

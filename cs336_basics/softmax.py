from einops import reduce
import torch


def causal_softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = reduce(x, '... last -> ... 1', 'max')
    x_normed = x - x_max
    x_exp = torch.exp(x_normed)
    x_exp = torch.tril(x_exp)
    x_exp_sum = reduce(x_exp, '... last -> ... 1', 'sum')
    x_out = torch.div(x_exp, x_exp_sum)
    return x_out


def softmax(x: torch.Tensor, dim: int, mask=None) -> torch.Tensor:
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_normed = x - x_max
    x_exp = torch.exp(x_normed)
    if mask is not None:
        x_exp = x_exp * mask
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    x_out = torch.div(x_exp, x_exp_sum)
    return x_out

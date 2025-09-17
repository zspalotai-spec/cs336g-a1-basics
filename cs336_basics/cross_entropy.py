import torch


def cross_entropy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    input_max, _ = torch.max(input, dim=-1, keepdim=True)
    input_normed = input - input_max
    input_exp = torch.exp(input_normed)
    input_exp_sum = torch.sum(input_exp, dim=-1, keepdim=True)
    input_exp_sum_log = torch.log(input_exp_sum) + input_max
    # writes 1s to the indices in target in an input sized mask matrix
    mask = torch.zeros_like(input).scatter_(dim=-1, index=target.reshape(list(target.size())+[1]), value=1)==1
    input_target = input[mask]
    p = input_target - input_exp_sum_log
    return torch.mean(-p)

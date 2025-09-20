from einops import reduce
import torch


def gather_values(input, target):
    return input.gather(dim=-1, index=target.view(list(target.size())+[1]))

def cross_entropy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    input_max = reduce(input, '... last -> ... 1', 'max')
    input_normed_exp = input - input_max
    input_normed_exp.exp_()
    input_exp_sum_log = reduce(input_normed_exp, '... last -> ... 1', 'sum')
    input_exp_sum_log.log_()
    input_exp_sum_log.add_(input_max)
    masked_values = gather_values(input, target)
    p =  input_exp_sum_log - masked_values
    return reduce(p, '... -> 1', 'mean')

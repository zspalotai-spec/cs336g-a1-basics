import numpy as np
import torch
from typing import Iterator


def get_batch(
    x: np.typing.NDArray, batch_size: int, context_length: int, device
) -> tuple[torch.LongTensor, torch.LongTensor]:
    start_indices = torch.randint(0, len(x) - context_length, (batch_size, 1))
    context_indices = torch.arange(0, context_length+1).reshape((1, context_length+1))
    indices = start_indices + context_indices
    input_target_x = torch.from_numpy(x[indices]).reshape((batch_size, context_length+1)).to(device)
    input = input_target_x[:,:-1]
    target = input_target_x[:,1:]
    return input, target

import numpy as np
import torch


def get_batch(
    x: np.typing.NDArray, batch_size: int, context_length: int, device
) -> tuple[torch.LongTensor, torch.LongTensor]:
    start_indices = torch.randint(0, x.shape[0] - context_length, (batch_size, 1))
    context_indices = torch.arange(0, context_length).reshape((1, context_length))
    indices = start_indices + context_indices
    input = torch.LongTensor(data=x[indices].reshape((batch_size, context_length)))
    target = torch.LongTensor(data=x[indices + 1].reshape((batch_size, context_length)))
    return input, target

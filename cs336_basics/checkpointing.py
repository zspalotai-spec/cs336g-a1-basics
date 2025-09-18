import os
import torch
import typing

_MODEL_KEY = "model"
_OPTMIZER_KEY = "optimizer"
_ITERATION_KEY = "iteration"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    d = {
        _MODEL_KEY: model.state_dict(),
        _OPTMIZER_KEY: optimizer.state_dict(),
        _ITERATION_KEY: iteration,
    }
    torch.save(d, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    d = torch.load(src)
    model.load_state_dict(d[_MODEL_KEY])
    optimizer.load_state_dict(d[_OPTMIZER_KEY])
    return d[_ITERATION_KEY]

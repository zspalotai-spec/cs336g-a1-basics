import numpy as np
import torch

from cs336_basics import adamw
from cs336_basics import cross_entropy
from cs336_basics import get_batch
from cs336_basics import transformer_lm
from cs336_basics import timer


def initialize_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    return transformer_lm.TransformerLm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
    )


def loss_fn(outputs, targets):
    return cross_entropy.cross_entropy(outputs, targets)


def initialize_optimizer(
    model,
    lr_max: float,
    lr_min: float,
    t_w: int,
    t_c: int,
    max_gradient_norm: float,
    weight_decay=1e-3,
    betas=(0.9, 0.95),
    eps=1e-8,
):
    return adamw.AdamWextra(
        model.parameters(),
        lr_max=lr_max,
        lr_min=lr_min,
        t_w=t_w,
        t_c=t_c,
        max_gradient_norm=max_gradient_norm,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        device=model.device,
    )


def train_one_step(training_inputs, training_targets, model, optimizer):
    timer.measure("zero_grad", lambda: optimizer.zero_grad(set_to_none=True))
    outputs = timer.measure("forward", lambda: model.forward(training_inputs))
    loss = timer.measure("loss", lambda: loss_fn(outputs, training_targets))
    timer.measure("backward", loss.backward)
    timer.measure("optmize", optimizer.step)
    return loss


def validate(model, validation_inputs, validation_targets):
    with torch.no_grad():
        outputs = timer.measure("validation_forward", lambda: model.forward(validation_inputs))
        loss = timer.measure("validation_loss", lambda: loss_fn(outputs, validation_targets))
    return loss


def train_n_steps_and_validate(
    x: np.typing.NDArray,
    batch_size: int,
    num_steps: int,
    x_valid: np.typing.NDArray,
    validation_batch_size: int,
    num_valid_step: int,
    model,
    optimizer,
):
    training_loss = None
    for _ in range(num_steps):
        inputs, targets = timer.measure(
            "train_get_batch",
            lambda: get_batch.get_batch(
                x, batch_size, model.context_length, model.device
            ),
        )
        loss = timer.measure(
            "train_one_step", lambda: train_one_step(inputs, targets, model, optimizer)
        )
        if training_loss is None:
            training_loss = loss
        else:
            training_loss += loss
    training_loss /= num_steps
    validation_loss = None
    for _ in range(num_valid_step):
        inputs, targets = timer.measure(
            "validation_get_batch",
            lambda: get_batch.get_batch(
                x_valid, validation_batch_size, model.context_length, model.device
            ),
        )
        act_validation_loss = timer.measure(
            "validate", lambda: validate(model, inputs, targets)
        )
        if validation_loss is None:
            validation_loss = act_validation_loss
        else:
            validation_loss += act_validation_loss
    validation_loss /= num_valid_step

    return validation_loss, training_loss

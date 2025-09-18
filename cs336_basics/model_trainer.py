import numpy as np
import torch

from cs336_basics import adamw
from cs336_basics import cross_entropy
from cs336_basics import get_batch
from cs336_basics import transformer_lm


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
    )


def train_one_step(training_inputs, training_targets, model, optimizer):
    optimizer.zero_grad()
    outputs = model.forward(training_inputs)
    loss = loss_fn(outputs, training_targets)
    loss.backward()
    optimizer.step()
    return loss


def validate(model, validation_inputs, validation_targets):
    with torch.no_grad():
        outputs = model.forward(validation_inputs)
        loss = loss_fn(outputs, validation_targets)
    perplexity = torch.exp(torch.mean(loss))
    return perplexity, loss


def train_n_steps_and_validate(
    x: np.typing.NDArray, batch_size: int, num_steps: int, model, optimizer
):
    running_loss = None
    for _ in range(num_steps):
        inputs, targets = get_batch.get_batch(x, batch_size, model.context_length, model.device)
        loss = train_one_step(inputs, targets, model, optimizer)
        if running_loss is None:
            running_loss = loss
        else:
            running_loss += loss
        print(".")
    mean_loss = running_loss / num_steps
    inputs, targets = get_batch.get_batch(x, batch_size, model.context_length, model.device)
    perplexity, validation_loss = validate(model, inputs, targets)
    return perplexity, validation_loss, mean_loss

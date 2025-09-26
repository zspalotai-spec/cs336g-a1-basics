import argparse
import os
import torch

from cs336_basics import checkpointing
from cs336_basics import model_trainer
from cs336_basics import softmax
from cs336_basics import timer


def load_model_from_checkpoint(checkpoint_dir: str, step: int):
    with open(os.path.join(checkpoint_dir, "params.txt"), "r") as f:
        model_args_str = f.read()

    model_args = argparse.Namespace()
    for parts in model_args_str[10:-1].split(","):
        name, value = parts.split("=")
        if value.startswith("'"):
            value = value.strip("'").strip()
        else:
            if "." in value or "e" in value:
                value = float(value)
            else:
                value = int(value)
        setattr(model_args, name.strip(), value)

    device = torch.device(model_args.device)
    model = model_trainer.initialize_model(
        model_args.vocab_size,
        model_args.context_length,
        model_args.d_model,
        model_args.num_layers,
        model_args.num_heads,
        model_args.d_ff,
        model_args.rope_theta,
        device=device,
        dtype=torch.float32,
    )
    optimizer = model_trainer.initialize_optimizer(
        model,
        model_args.lr_max,
        model_args.lr_min,
        model_args.t_w,
        model_args.t_c,
        model_args.max_gradient_norm,
        model_args.weight_decay,
        (model_args.beta1, model_args.beta2),
        model_args.adamw_eps,
    )

    current_step = checkpointing.load_checkpoint(
        os.path.join(checkpoint_dir, f"ckp_{step}.ckp"),
        model,
        optimizer,
    )

    model.to(model.device)
    return model, optimizer, current_step


def scaled_probs(input: torch.Tensor, temperature: float) -> torch.Tensor:
    return softmax.softmax(input / temperature, -1)


def nucleus_sampling(
    probs: torch.Tensor, cumumlative_probability_threshold: float
) -> torch.Tensor:
    sorted_probs, indices = probs.sort(-1)
    cumsum_sorted_probs = sorted_probs.cumsum(-1)
    not_needed = cumsum_sorted_probs < (1.0 - cumumlative_probability_threshold)
    probs[indices[not_needed]] = 0.0
    return torch.distributions.categorical.Categorical(probs).sample()


def inference_step(
    model,
    tokens: torch.Tensor,
    temperature: float = 1.0,
    cumumlative_probability_threshold: float = 1.0,
) -> torch.Tensor:
    result = timer.measure("forward", lambda: model.forward(tokens))
    probs = timer.measure(
        "scaled softmax", lambda: scaled_probs(result[-1, :], temperature)
    )
    if cumumlative_probability_threshold < 1.0:
        return timer.measure(
            "nucleus sampling",
            lambda: nucleus_sampling(probs, cumumlative_probability_threshold),
        )
    else:
        return timer.measure(
            "sampling",
            lambda: torch.distributions.categorical.Categorical(probs).sample(),
        )


def inference(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    cumumlative_probability_threshold: float = 1.0,
):
    input = torch.tensor(tokenizer.encode(prompt), device=model.device)
    output = None
    cnt = 0
    while cnt < max_tokens:
        new_token = timer.measure(
            "inference_step",
            lambda: inference_step(
                model, input, temperature, cumumlative_probability_threshold
            ).reshape((1,)),
        )
        if output is None:
            output = torch.clone(new_token.detach())
        else:
            output = torch.cat([output, new_token])
        input = torch.cat([input, new_token])
        cnt += 1
        if new_token.item() == 0:
            break
    ret = tokenizer.decode(output.tolist())
    return ret

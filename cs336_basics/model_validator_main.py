import argparse
import numpy as np
import os
import random
import torch

from cs336_basics import checkpointing
from cs336_basics import get_batch
from cs336_basics import model_trainer
from cs336_basics import timer


def main():
    seed = 0
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
            return
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
            return

    print(torch.mps.device_count(), torch.mps.recommended_max_memory() / pow(2, 30))

    parser = argparse.ArgumentParser(prog="model_trainer")
    parser.add_argument("--data_src")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--step_to_load")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=1000)
    args = parser.parse_args()

    print(str(args))

    with open(os.path.join(args.checkpoint_dir, "params.txt"), "r") as f:
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
        os.path.join(args.checkpoint_dir, f"ckp_{args.step_to_load}.ckp"),
        model,
        optimizer,
    )

    model.to(model.device)

    x = np.load(args.data_src, mmap_mode="r").astype(np.int64)

    total_loss = 0
    for idx in range(args.num_batches):
        inputs, targets = get_batch.get_batch(
            x, args.batch_size, model.context_length, model.device
        )
        _, loss = timer.measure(
            "train_n_step",
            lambda: model_trainer.validate(model, inputs, targets),
        )
        total_loss += loss.item()
        if idx % 10 == 0:
            print(total_loss / (idx + 1))

    print(total_loss / args.num_batches)
    print(timer.get_times_str())


if __name__ == "__main__":
    main()

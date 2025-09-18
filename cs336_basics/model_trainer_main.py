import argparse
import numpy as np
import os
import torch

from cs336_basics import checkpointing
from cs336_basics import model_trainer


def main():
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
    mps_device = torch.device("mps")

    parser = argparse.ArgumentParser(prog="model_trainer")
    parser.add_argument("--data_src")
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--checkpoint_to_load", default="")
    parser.add_argument("--report_counts", type=int, default=500)
    parser.add_argument("--reporting_step_count", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--validation_batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", default=10000.0)
    parser.add_argument("--lr_max", default=0.1)
    parser.add_argument("--lr_min", default=1e-6)
    parser.add_argument("--t_w", type=int, default=100)
    parser.add_argument("--t_c", type=int, default=5000)
    parser.add_argument("--max_gradient_norm", default=1.0)
    parser.add_argument("--weight_decay", default=1e-3)
    parser.add_argument("--beta1", default=0.9)
    parser.add_argument("--beta2", default=0.95)
    parser.add_argument("--adamw_eps", default=1e-8)
    args = parser.parse_args()

    print(str(args))

    model = model_trainer.initialize_model(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        device=mps_device,
    )
    optimizer = model_trainer.initialize_optimizer(
        model,
        args.lr_max,
        args.lr_min,
        args.t_w,
        args.t_c,
        args.max_gradient_norm,
        args.weight_decay,
        (args.beta1, args.beta2),
        args.adamw_eps,
    )
    if args.checkpoint_to_load:
        current_step = checkpointing.load_checkpoint(
            args.checkpoint_to_load, model, optimizer
        )
    else:
        current_step = 0

    #model = torch.compile(model, backend="aot_eager")

    x = np.load(args.data_src, mmap_mode="r").astype(np.int64)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    with open(os.path.join(args.checkpoint_dir, "params.txt"), "w") as f:
        f.write(str(args))

    while current_step < args.report_counts * args.reporting_step_count:
        perplexity, validation_loss, training_loss = (
            model_trainer.train_n_steps_and_validate(
                x,
                args.validation_batch_size,
                args.reporting_step_count,
                model,
                optimizer,
            )
        )
        current_step += args.reporting_step_count
        print(current_step, perplexity, validation_loss, training_loss)
        checkpointing.save_checkpoint(
            model,
            optimizer,
            current_step,
            os.path.join(args.checkpoint_dir, f"ckp_{current_step}.ckp"),
        )


if __name__ == "__main__":
    main()

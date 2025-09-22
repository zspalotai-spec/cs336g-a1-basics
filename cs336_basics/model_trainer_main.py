import argparse
from datetime import datetime
import numpy as np
import os
import random
import torch
import wandb

from cs336_basics import checkpointing
from cs336_basics import model_trainer
from cs336_basics import timer


def main():
    wandb.login()
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
    parser.add_argument("--training_data_src")
    parser.add_argument("--validation_data_src")
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--input_length", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--checkpoint_to_load", default="")
    parser.add_argument("--report_counts", type=int, default=200)
    parser.add_argument("--reporting_step_count", type=int, default=100)
    parser.add_argument("--validation_step_count", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--validation_batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", default=10000.0)
    parser.add_argument("--lr_max", default=5e-4)
    parser.add_argument("--lr_min", default=5e-7)
    parser.add_argument("--t_w", type=int, default=4000)
    parser.add_argument("--t_c", type=int, default=-1)
    parser.add_argument("--max_gradient_norm", default=1.0)
    parser.add_argument("--weight_decay", default=1e-1)
    parser.add_argument("--beta1", default=0.9)
    parser.add_argument("--beta2", default=0.95)
    parser.add_argument("--adamw_eps", default=1e-8)
    args = parser.parse_args()

    print(str(args))

    if args.t_c == -1:
        args.t_c = args.report_counts * args.reporting_step_count

    wandb_run = wandb.init(
        project=args.checkpoint_dir,
        config=vars(args),
    )

    device = torch.device(args.device)
    if args.device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.float32

    model = model_trainer.initialize_model(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        device=device,
        dtype=dtype,
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

    model.to(device)
    if args.device == "cpu":
        pass
        # model = torch.compile(model)
    else:
        pass
        # timer.DEBUG = False
        # model = torch.compile(model, backend="aot_eager")

    x = np.load(args.training_data_src, mmap_mode="r").astype(np.int64)
    if args.input_length > 0:
        x = x[: args.input_length]

    x_valid = np.load(args.validation_data_src, mmap_mode="r").astype(np.int64)
    if args.input_length > 0:
        x_valid = x_valid[: args.input_length]

    checkpoint_dir = os.path.join(
        args.checkpoint_dir, datetime.now().strftime("%Y%m%d%H%M%S")
    )

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(os.path.join(checkpoint_dir, "params.txt"), "w") as f:
        f.write(str(args))

    while current_step < args.report_counts * args.reporting_step_count:
        time1 = datetime.now()
        validation_loss, training_loss = timer.measure(
            "train_n_step",
            lambda: model_trainer.train_n_steps_and_validate(
                x,
                args.batch_size,
                args.reporting_step_count,
                x_valid,
                args.validation_batch_size,
                args.validation_step_count,
                model,
                optimizer,
            ),
        )
        wandb_run.log(
            {"training_loss": training_loss, "validation_loss": validation_loss}
        )
        current_step += args.reporting_step_count
        print(current_step, validation_loss, training_loss)
        timer.measure(
            "checkpointing",
            lambda: checkpointing.save_checkpoint(
                model,
                optimizer,
                current_step,
                os.path.join(checkpoint_dir, f"ckp_{current_step}.ckp"),
            ),
        )
        print(datetime.now() - time1)

    print(timer.get_times_str())


if __name__ == "__main__":
    main()

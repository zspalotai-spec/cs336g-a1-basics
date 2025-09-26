import argparse
import numpy as np
import os
import random
import torch

from cs336_basics import bpe_tokenizer
from cs336_basics import model_inference
from cs336_basics import timer


def main():
    seed = 1000
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

    parser = argparse.ArgumentParser(prog="model_inference")
    parser.add_argument("--tokenizer_vocab")
    parser.add_argument("--tokenizer_merges")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--step_to_load")
    parser.add_argument("--prompt")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--cumumlative_probability_threshold", type=float, default=0.1)
    args = parser.parse_args()

    print(str(args))

    tokenizer = bpe_tokenizer.BPETokenizer.from_files(
        args.tokenizer_vocab, args.tokenizer_merges
    )

    model, _, _ = model_inference.load_model_from_checkpoint(
        args.checkpoint_dir, args.step_to_load
    )

    response = timer.measure(
        "inference",
        lambda: model_inference.inference(
            model,
            tokenizer,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.cumumlative_probability_threshold,
        ),
    )

    print(timer.get_times_str())

    print(response)


if __name__ == "__main__":
    main()

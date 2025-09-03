from cs336_basics import bpe_tokenizer
from cs336_basics import token_utils

if __name__ == "__main__":
    FILE_PATH_PREFIX = "../data/owt_train."
    tokenizer, times = bpe_tokenizer.BPETokenizer.train(
        FILE_PATH_PREFIX + "txt",
        32000,
        ["<|endoftext|>"],
        num_processes=10,
        num_chunks=10,
    )
    for idx, t in enumerate(times[1:]):
        duration = t - times[idx]
        print("duration (s): ", duration.total_seconds())
    token_utils.save_vocab_and_merges(
        tokenizer.vocab,
        tokenizer.merges,
        vocab_path=FILE_PATH_PREFIX + "vocab",
        merges_path=FILE_PATH_PREFIX + "merges",
    )

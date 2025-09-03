from cs336_basics import bpe_tokenizer
from cs336_basics import token_utils

if __name__ == "__main__":
    tokenizer, times = bpe_tokenizer.BPETokenizer.train(
        "../data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
        num_processes=10,
        num_chunks=10,
    )
    for idx, t in enumerate(times[1:]):
        print(t - times[idx])
    token_utils.save_vocab_and_merges(
        tokenizer.vocab,
        tokenizer.merges,
        vocab_path="../data/TinyStoriesV2-GPT4-train.vocab",
        merges_path="../data/TinyStoriesV2-GPT4-train.merges",
    )

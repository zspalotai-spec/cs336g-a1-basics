import cProfile
import pstats

from cs336_basics import bpe_tokenizer
from cs336_basics import token_utils

if __name__ == "__main__":
    vocab, merges = token_utils.load_vocab_and_merges(
        vocab_path="../data/TinyStoriesV2-GPT4-train.vocab",
        merges_path="../data/TinyStoriesV2-GPT4-train.merges",
    )
    tokenizer = bpe_tokenizer.BPETokenizer(vocab, merges, ["<|endoftext|>"])

    with open("../data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
        text = f.read().decode("utf-8")

    with cProfile.Profile() as pr:
        encoded = tokenizer.encode(text)

    p = pstats.Stats(pr)
    p.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(10)
    p.print_callees(10)

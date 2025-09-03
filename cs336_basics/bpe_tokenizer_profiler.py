import cProfile
import pstats

from cs336_basics import bpe_tokenizer

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        tokenizer = bpe_tokenizer.BPETokenizer.train('../data/TinyStoriesV2-GPT4-train.txt',10000,['<|endoftext|>'])

    p = pstats.Stats(pr)
    p.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(10)
    p.print_callees(10)
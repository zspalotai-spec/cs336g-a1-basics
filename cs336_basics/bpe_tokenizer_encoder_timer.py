from datetime import datetime

from cs336_basics import bpe_tokenizer
from cs336_basics import token_utils

if __name__ == "__main__":
    vocab, merges = token_utils.load_vocab_and_merges(
        vocab_path="../data/TinyStoriesV2-GPT4-train.vocab",
        merges_path="../data/TinyStoriesV2-GPT4-train.merges",
    )
    tokenizer = bpe_tokenizer.BPETokenizer(vocab, merges, ["<|endoftext|>"])

    with open("../data/TinyStoriesV2-GPT4-train.txt", "rb") as f:
        text_b = f.read()
    text = text_b.decode("utf-8")

    time1 = datetime.now()
    encoded = tokenizer.encode(text)
    duration = datetime.now()-time1

    print('duration ',duration)
    print('throughput ',len(text_b)/duration.seconds/1e6)
    print('compression ',len(text_b)/len(encoded))

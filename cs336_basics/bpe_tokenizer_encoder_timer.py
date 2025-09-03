from datetime import datetime
import numpy as np
import os

from cs336_basics import bpe_tokenizer
from cs336_basics import pretokenization_example
from cs336_basics import token_utils

if __name__ == "__main__":
    VOCAB_PREFIX = "../data/owt_train."
    vocab, merges = token_utils.load_vocab_and_merges(
        vocab_path=VOCAB_PREFIX + "vocab",
        merges_path=VOCAB_PREFIX + "merges",
    )
    tokenizer = bpe_tokenizer.BPETokenizer(vocab, merges, ["<|endoftext|>"])

    FILE_PREFIX = VOCAB_PREFIX
    file_size = os.path.getsize(FILE_PREFIX + "txt")
    DESIRED_CHUNK_SIZE = pow(2, 30)
    num_chunks = int((file_size + DESIRED_CHUNK_SIZE - 1) / DESIRED_CHUNK_SIZE)
    encoded = []
    time1 = datetime.now()
    duration = time1 - time1
    with open(FILE_PREFIX + "txt", "rb") as f:
        boundaries = pretokenization_example.find_chunk_boundaries(
            f, num_chunks, b"<|endoftext|>"
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            text = f.read(end - start).decode("utf-8")
            time1 = datetime.now()
            encoded1 = tokenizer.encode(text)
            duration += datetime.now() - time1
            encoded.extend(encoded1)

    print("duration (s) ", duration.total_seconds())
    print("throughput (MB/s)", file_size / duration.total_seconds() / pow(2,20))
    print("compression ", file_size / len(encoded))

    np.save(FILE_PREFIX + "npy", np.asarray(encoded, "int16"))

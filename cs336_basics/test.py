import regex as re
import numpy as np
from einops import einsum, rearrange
import torch
import math

from cs336_basics import token_utils

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
for m in re.finditer(PAT, "teeeeest string"):
    print([bytes([b]) for b in m.group(0).encode("utf-8")])

# with open('../data/TinyStoriesV2-GPT4-valid.txt', 'rb') as f:
#    text = f.read()
# print(text[:1000])


vocab, merges = token_utils.load_vocab_and_merges(
    "/Users/zsoltpalotai/data/owt_train.vocab",
    "/Users/zsoltpalotai/data/owt_train.merges",
)

length_counts={}
max_len = 0
max_len_v = ''

for v in vocab.values():
    l = len(v)
    length_counts[l] = length_counts.get(l,0)+1
    if l > max_len:
        max_len = l
        max_len_v = v
    elif l == max_len:
        if isinstance(max_len_v,bytes):
            max_len_v = [max_len_v, v]
        else:
            max_len_v.append(v)

print(sorted(length_counts.items()))
print(max_len)
for v in max_len_v:
    print(v, v.decode('utf8'))
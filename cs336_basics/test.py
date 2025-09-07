import regex as re
import numpy as np
from einops import einsum, rearrange
import torch
import math

PAT=re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
for m in re.finditer(PAT, "teeeeest string"):
    print([bytes([b]) for b in m.group(0).encode('utf-8')])

#with open('../data/TinyStoriesV2-GPT4-valid.txt', 'rb') as f:
#    text = f.read()
#print(text[:1000])


print(math.exp(-math.inf-5.0))

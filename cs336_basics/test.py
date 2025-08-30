import common_types
import regex as re
import token_utils
from typing import Iterable, Iterator, Self
import utils

PAT=re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
for m in re.finditer(PAT, "teeeeest string"):
    print([bytes([b]) for b in m.group(0).encode('utf-8')])

#with open('../data/TinyStoriesV2-GPT4-valid.txt', 'rb') as f:
#    text = f.read()
#print(text[:1000])

l = ['a','b','c']
for a,b in zip(l[:-1],l[1:]):
    print(a,b)
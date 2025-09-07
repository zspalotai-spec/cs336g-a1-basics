import regex as re
import numpy as np
from einops import einsum, rearrange
import torch

PAT=re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
for m in re.finditer(PAT, "teeeeest string"):
    print([bytes([b]) for b in m.group(0).encode('utf-8')])

#with open('../data/TinyStoriesV2-GPT4-valid.txt', 'rb') as f:
#    text = f.read()
#print(text[:1000])


a = torch.Tensor([[range(0,6), range(6,12), range(12,18)],[range(18,24), range(24,30), range(30,36)]])
print(a)
a = torch.view_as_complex(rearrange(a, '... (d d1) -> ... d d1',d1=2))
print(a)
b = torch.view_as_complex(torch.Tensor([[0.5,0.1],[0.2,-0.1],[-0.1,-0.1]]))
print(b)
a = rearrange(torch.view_as_real(b.mul(a)),'... d d1 -> ... (d d1)')
print(a)
print((0+1j)*(0.5+0.1j))
print((2+3j)*(0.2-0.1j))
print((4+5j)*(-0.1-0.1j))
print((6+7j)*(0.5+0.1j))
print((8+9j)*(0.2-0.1j))
print((10+11j)*(-0.1-0.1j))

import cProfile
import pstats

import torch
from torch import nn
import unittest

from cs336_basics import cross_entropy


class TestCrossEntropy(unittest.TestCase):
    def test_multiple_batch_dims(self):
        input = nn.init.trunc_normal_(
                torch.empty((2,3,4, 10)),
                mean=0.0,
                std=1,
                a=-3,
                b=3,
            )
        target = torch.randint(0,10, (2,3,4),dtype=torch.long)
        single_batch = cross_entropy.cross_entropy(input.reshape((-1,10)), target.reshape((-1,)))
        multi_batch = cross_entropy.cross_entropy(input, target)
        self.assertEqual(single_batch, multi_batch)

    def test_profile(self):
        device = torch.device("mps")
        input = nn.init.trunc_normal_(
                torch.empty((16,32,64, 128), device=device),
                mean=0.0,
                std=1,
                a=-3,
                b=3,
            )
        target = torch.randint(0,128, (16,32,64),dtype=torch.long, device=device)
        with cProfile.Profile() as pr:
            for _ in range(10):
                cross_entropy.cross_entropy(input, target)
        p = pstats.Stats(pr)
        p.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(10)
        p.print_callees(10)

if __name__ == "__main__":
    unittest.main()

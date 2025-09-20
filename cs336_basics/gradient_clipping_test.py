import cProfile
import pstats

import torch
import unittest

from cs336_basics import gradient_clipping


def reset_gradients(params):
    for p in params:
        p.grad = torch.randn((1024, 1204), device=p.device)


class GradientClippingTest(unittest.TestCase):
    def test_profile(self):
        device = torch.device("mps")
        max_norm = torch.as_tensor([1e-2], device=device)
        eps = torch.as_tensor([1e-6], device=device)
        params = [torch.randn((1024, 1204), device=device) for _ in range(100)]
        for p in params:
            p.grad = torch.randn((1024, 1204), device=p.device)
        with cProfile.Profile() as pr:
            for _ in range(100):
                reset_gradients(params)
                gradient_clipping.clip(params, max_norm, eps)
        p = pstats.Stats(pr)
        p.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(10)
        p.print_callees(10, "gradient_clipping")


if __name__ == "__main__":
    unittest.main()

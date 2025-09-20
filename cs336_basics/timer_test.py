import unittest
import torch

from cs336_basics import timer

class TimerTest(unittest.TestCase):
    def test_measure_none_returned(self):
        def nonreturned(x):
            pass
        timer.measure("nonereturned", lambda: nonreturned(1))

    def test_measure_many_returned(self):
        def manyreturned(x):
            return x, 2*x, 3*x
        out = timer.measure("nonereturned", lambda: manyreturned(1))
        self.assertEqual(out, (1,2,3))

    def test_inf_devision(self):
        print(1*torch.div(-1.,torch.tensor([1.,0.])))
        print(torch.nan_to_num(0./torch.Tensor([0.]), nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf))
        


if __name__ == "__main__":
    unittest.main()
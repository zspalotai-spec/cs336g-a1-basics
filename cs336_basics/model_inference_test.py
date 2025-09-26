import unittest
import torch

from cs336_basics import model_inference

class ModelInferenceTest(unittest.TestCase):
    def test_scaled_probs(self):
        input = torch.randn((1,100))
        output1 = model_inference.scaled_probs(input, 0.5)
        output2 = model_inference.scaled_probs(input, 1.5)
        max_ids1 = output1.argmax()
        max_ids2 = output2.argmax()
        self.assertAlmostEqual(output1.sum().item(), 1.0, places=4)
        self.assertAlmostEqual(output2.sum().item(), 1.0, places=4)
        self.assertEqual(max_ids1, max_ids2)
        self.assertGreater(output1[0,max_ids1.item()],output2[0,max_ids2.item()])

    def test_nucleus_sampling(self):
        for _ in range(1000):
            input = torch.tensor([0.1]*9+[0.01]*10)
            idx = model_inference.nucleus_sampling(input, 0.85)
            self.assertEqual(input[9:].sum(), 0.)
            self.assertLess(idx.item(),9)

if __name__ == "__main__":
    unittest.main()
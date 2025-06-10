
# phase3/tests/test_binning.py
import unittest
import torch
from phase3.binning.learnable_binning import LearnableBinning

class TestLearnableBinning(unittest.TestCase):
    def test_output_shape_and_sum(self):
        B, bins = 4, 3
        init_edges = torch.linspace(0, 1, bins)
        x = torch.rand(B)
        model = LearnableBinning(num_bins=bins, init_edges=init_edges)
        out = model(x)
        self.assertEqual(out.shape, (B, bins))
        row_sums = out.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(B), atol=1e-5))
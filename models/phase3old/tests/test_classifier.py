# phase3/tests/test_classifier.py
import unittest
import torch
from phase3.classifier.moe_classifier import MoEClassifier

class TestMoEClassifier(unittest.TestCase):
    def test_output_shape_and_bounds(self):
        B, F, bins = 5, 8, 4
        features = torch.rand(B, F)
        bin_outputs = torch.rand(B, bins)
        model = MoEClassifier(feature_dim=F, num_bins=bins)
        out = model(features, bin_outputs)
        self.assertEqual(out.shape, (B,))
        min_vals, _ = bin_outputs.min(dim=1)
        max_vals, _ = bin_outputs.max(dim=1)
        self.assertTrue(torch.all(out >= min_vals))
        self.assertTrue(torch.all(out <= max_vals))

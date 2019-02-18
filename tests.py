import unittest

import torch as to
import neural_statistician as ns

# List tests here

class TestNeuralStatistician(unittest.TestCase):

    def test_normal_kl_divergence(self):
        statistician = ns.NeuralStatistician(1)

        mean_0 = to.rand(10)
        diag_cov_0 = to.rand(10)
        mean_1, diag_cov_1 = mean_0.clone(), diag_cov_0.clone()
        self.assertAlmostEqual(0.0, statistician.normal_kl_divergence(mean_0, diag_cov_0, mean_1, diag_cov_1))


if __name__ == '__main__':
    unittest.main()

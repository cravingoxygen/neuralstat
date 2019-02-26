import unittest

import torch as to
import neural_statistician as ns
import one_dimensional_test as one

# List tests here

class TestNeuralStatistician(unittest.TestCase):

    def test_normal_kl_divergence(self):
        statistician = ns.NeuralStatistician(1, 3,
                                             one.LatentDecoder, one.ObservationDecoder,
                                             one.StatisticNetwork, one.InferenceNetwork)

        mean_0 = to.rand((1, 10))
        diag_cov_0 = to.rand((1, 10))
        mean_1, diag_cov_1 = mean_0.clone(), diag_cov_0.clone()
        self.assertAlmostEqual(0.0, statistician.normal_kl_divergence(mean_0, diag_cov_0, mean_1, diag_cov_1).item())


        for _ in range(50):
            mean_0 = to.rand((20, 10))
            diag_cov_0 = to.rand((20, 10))
            mean_1 = to.rand((20, 10))
            diag_cov_1 = to.rand((20, 10))
            self.assertTrue((statistician.normal_kl_divergence(mean_0, diag_cov_0, mean_1, diag_cov_1) >= 0).byte().all())


if __name__ == '__main__':
    unittest.main()

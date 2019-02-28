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

        mean_0 = to.rand((1, 1, 10))
        diag_cov_0 = to.rand((1, 1, 10))
        mean_1, diag_cov_1 = mean_0.clone(), diag_cov_0.clone()
        self.assertAlmostEqual(0.0, statistician.normal_kl_divergence(mean_0, diag_cov_0, mean_1, diag_cov_1).item())


        for _ in range(50):
            mean_0 = to.rand((20, 1, 10))
            diag_cov_0 = to.rand((20, 1, 10))
            mean_1 = to.rand((20, 1, 10))
            diag_cov_1 = to.rand((20, 1, 10))
            self.assertTrue((statistician.normal_kl_divergence(mean_0, diag_cov_0, mean_1, diag_cov_1) >= 0).byte().all())


    def test_normal_distribution(self):
        mean = to.rand((16, 200, 1))
        stdev = to.rand((16, 200, 1))
        data = to.randn((16, 200, 1))

        distribution = to.distributions.normal.Normal(loc=mean, scale=stdev)

        log_probability = distribution.log_prob(data)
        log_probability_indiv = 0
        for batch_idx in range(len(mean)):
            for sample_idx in range(batch_idx):
                distribution_i = to.distributions.normal.Normal(loc=mean[batch_idx, sample_idx, 1], scale=stdev[batch_idx, sample_idx])
                log_probability_indiv += distribution_i.log_prob(data[batch_idx, sample_idx, 1])

        self.assertTrue(log_probability_indiv == log_probability)


if __name__ == '__main__':
    unittest.main()

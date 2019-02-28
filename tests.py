import unittest

import torch as to
import neural_statistician as ns
import one_dimensional_test as one
from math import log, pi
# List tests here

def gaussian_log_likelihood(x, mean, logvar, clip=True):
    if clip:
        logvar = to.clamp(logvar, min=-4, max=3)
    a = log(2*pi)
    b = logvar
    c = (x - mean)**2 / to.exp(logvar)
    return -0.5 * to.sum(a + b + c)

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




    def test_normal_distribution(self):
        mean = to.rand((16, 200, 1))
        stdev = to.rand((16, 200, 1))
        data = to.randn((16, 200, 1))
        
        
        log_probability_comparison = gaussian_log_likelihood(data.view(-1,1), mean.view(-1,1), to.log(stdev**2).view(-1,1), clip=False).item()
        
        distribution = to.distributions.normal.Normal(loc=mean, scale=stdev)
        log_probability = distribution.log_prob(data).sum().item()
        
        self.assertAlmostEqual(log_probability_comparison,log_probability)
        
        log_probability_indiv = 0
        for batch_idx in range(len(mean)):
            for sample_idx in range(batch_idx):
                distribution_i = to.distributions.normal.Normal(loc=mean[batch_idx, sample_idx, 0], scale=stdev[batch_idx, sample_idx,0])
                log_probability_indiv += distribution_i.log_prob(data[batch_idx, sample_idx, 0]).item()

        self.assertAlmostEqual(log_probability_indiv,log_probability)


if __name__ == '__main__':
    unittest.main()

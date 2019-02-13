import torch as to
import torch.nn.functional as F

import one_dimensional_test as ex


class NeuralStatistician(object):
    """Tying-together class to hold references for a particular experiment"""

    def __init__(self, num_stochastic_layers):
        super().__init__()

        self.latent_decoders = [ex.LatentDecoder() for _ in num_stochastic_layers]
        self.observation_decoder = ex.ObservationDecoder()
        self.statistic_network = ex.StatisticNetwork()
        self.inference_networks = [ex.InferenceNetwork() for _ in num_stochastic_layers]

        self.context_prior = to.distributions.multivariate_normal.MultivariateNormal(
            loc=to.zeros(ex.context_dimension), covariance_matrix=to.eye(ex.context_dimension))

    def compute_context_divergence(self, num_iterations=100):
        pass

    def compute_loss(self, distribution_parameters):
        """Compute the full model loss function"""
        # Context divergence
        # Expectation, under q(c | D; phi), of log (q(c|D;phi) / p(c))
        

    def predict(self):
        pass

    def train(self, data, num_iterations):
        """Train the Neural Statistician"""

        network_parameters = None
        optimiser = to.optim.Adam(network_parameters,
                                  lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

        for iteration in num_iterations:
            distribution_parameters = self.predict(data)
            loss = self.compute_loss(distribution_parameters)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()


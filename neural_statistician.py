import torch as to
import torch.nn.functional as F
import numpy as np

import one_dimensional_test as ex


class NeuralStatistician(object):
    """Tying-together class to hold references for a particular experiment"""

    def __init__(self, num_stochastic_layers):
        super().__init__()

        self.latent_decoders = [ex.LatentDecoder() for _ in range(num_stochastic_layers)]
        self.observation_decoder = ex.ObservationDecoder()
        self.statistic_network = ex.StatisticNetwork()
        self.inference_networks = [ex.InferenceNetwork() for _ in range(num_stochastic_layers)]

        self.context_prior_mean = to.zeros(ex.context_dimension)
        self.context_prior_cov = to.eye(ex.context_dimension)
        self.context_prior = to.distributions.multivariate_normal.MultivariateNormal(
            loc=self.context_prior_mean, covariance_matrix=cself.context_prior_cov)

    def normal_kl_divergence(self, mean_0, diag_cov_0, mean_1, diag_cov_1):
        """Compute the KL divergence between two diagonal Gaussians, where
        diag_cov_x is a 1D vector containing the diagonal elements of the
        xth covariance matrix."""
        return 0.5 * (
            to.dot(1 / diag_cov_1, diag_cov_0) +
            to.dot((mean_1 - mean_0) ** 2, 1 / diag_cov_1) - mean_0.size()[0] +
            to.sum(to.log(mean_1)) - to.sum(to.log(mean_0))
        )

    def compute_loss(self, context_output, inference_outputs, decoder_outputs, observation_decoder_outputs, data):
        """Compute the full model loss function"""
        # Context divergence
        context_mean, context_log_cov = context_output
        context_divergence = self.normal_kl_divergence(context_mean, to.exp(context_log_cov),
                                                       self.context_prior_mean, self.context_prior_cov)

        # Latent divergence
        # For computational efficiency, draw a single sample context from q(c, z | D, phi)
        # rather than computing the expectation properly.
        latent_divergence = to.tensor(0.0)
        for ((inference_mu, inference_log_cov), (decoder_mu, decoder_log_cov)) in zip(inference_outputs, decoder_outputs):
            latent_divergence += self.normal_kl_divergence(inference_mu, to.exp(inference_log_cov),
                                                           decoder_mu, to.exp(decoder_log_cov))

        # Reconstruction loss
        observation_decoder_mean, observation_decoder_log_cov = observation_decoder_outputs
        reconstruction_loss = to.distributions.normal.Normal(
            loc=observation_decoder_mean, scale=to.exp(0.5 * observation_decoder_log_cov)).log_prob(data)

        return context_divergence + latent_divergence + reconstruction_loss


    def predict(self, data):
        statistic_net_outputs = self.statistic_network(data)
        contexts = self.reparameterise_normal(context_means, context_log_vars)

        inference_net_outputs = [self.inference_networks[0](data, contexts)]
        latent_z = [self.reparameterise_normal(*inference_net_outputs)]
        for inference_network in self.inference_networks[1:]:
            inference_net_outputs.append(inference_network(data, contexts, inference_net_outputs[-1]))


    def reparameterise_normal(self, mean, log_var):
        """Draw samples from the given normal distribution via the
        reparameterisation trick"""
        std_errors = np.random.normal(np.zeros(len(mean)), np.ones(len(mean)))
        return mean + np.exp(0.5 * log_var) * std_errors
        

    def train(self, data, num_iterations):
        """Train the Neural Statistician"""

        network_parameters = None
        optimiser = to.optim.Adam(network_parameters,
                                  lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

        for iteration in num_iterations:
            distribution_parameters = self.predict(data)
            loss = self.compute_loss(*distribution_parameters)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()


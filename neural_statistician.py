import torch as to
import torch.nn.functional as F
import numpy as np
import pickle

import matplotlib.pyplot as plt


class NeuralStatistician(object):
    """Tying-together class to hold references for a particular experiment"""

    def __init__(self, num_stochastic_layers, context_dimension,
                 LatentDecoder, ObservationDecoder, StatisticNetwork, InferenceNetwork):
        super().__init__()

        self.latent_decoders = [LatentDecoder() for _ in range(num_stochastic_layers)]
        self.observation_decoder = ObservationDecoder()
        self.statistic_network = StatisticNetwork()
        self.inference_networks = [InferenceNetwork() for _ in range(num_stochastic_layers)]

        # for network in self.latent_decoders:
        #     network.apply(NeuralStatistician.init_weights)
        # self.observation_decoder.apply(NeuralStatistician.init_weights)
        # self.statistic_network.apply(NeuralStatistician.init_weights)
        # for network in self.inference_networks:
        #     network.apply(NeuralStatistician.init_weights)

        self.context_prior_mean = to.zeros(context_dimension)
        # CHECK: Is it OK to restrict ourselvs to a diagonal covariance matrix for the prior?
        self.context_prior_cov = to.ones(context_dimension)
        # self.context_prior = to.distributions.multivariate_normal.MultivariateNormal(
        #     loc=self.context_prior_mean, covariance_matrix=to.diag(self.context_prior_cov))

        self.context_divergence_history = []
        self.latent_divergence_history = []
        self.reconstruction_loss_history = []
        self.loss_history = []
        self.counter = 0


    def normal_kl_divergence(self, mean_0, diag_cov_0, mean_1, diag_cov_1):
        """Compute the KL divergence between two diagonal Gaussians, where
        diag_cov_x is a 1D vector containing the diagonal elements of the
        xth covariance matrix."""
        
        batch_size = mean_0.shape[0]
        return 0.5 * (
            ((1 / diag_cov_1) * diag_cov_0).sum(dim=2) + 
            (((mean_1 - mean_0) ** 2) * (1 / diag_cov_1)).sum(dim=2) - 
            mean_0.shape[-1] +
            to.sum(to.log(diag_cov_1), dim=-1) - to.sum(to.log(diag_cov_0), dim=-1)
        ).sum(dim=1)

    def compute_loss(self, context_output, inference_outputs, decoder_outputs, observation_decoder_outputs, data):
        """Compute the full model loss function"""
        sample_size = data.shape[1]

        # Context divergence
        context_mean, context_log_cov = context_output
        # Handle this case without separate data points by introducing a dummy
        # dimension, as if there were exactly 1 data point
        context_divergence = self.normal_kl_divergence(context_mean.unsqueeze(dim=1), to.exp(context_log_cov).unsqueeze(dim=1),
                                                       self.context_prior_mean.expand_as(context_mean.unsqueeze(dim=1)), self.context_prior_cov.expand_as(context_log_cov.unsqueeze(dim=1)))
        context_divergence *= sample_size

        # Latent divergence
        # For computational efficiency, draw a single sample context from q(c, z | D, phi)
        # rather than computing the expectation properly.
        latent_divergence = to.zeros(context_divergence.shape)
        for ((inference_mu, inference_log_cov), (decoder_mu, decoder_log_cov)) in zip(inference_outputs, decoder_outputs):
            #Expand the decoder's outputs to be the same shape as the inference networks
            # i.e. batch_size x dataset_size x data_dimensionality (for mean and log_var)
            latent_divergence += self.normal_kl_divergence(inference_mu, to.exp(inference_log_cov),
                                                           decoder_mu.unsqueeze(1).expand_as(inference_mu), to.exp(decoder_log_cov).unsqueeze(1).expand_as(inference_log_cov))

        # Reconstruction loss
        observation_decoder_mean, observation_decoder_log_cov = observation_decoder_outputs
        
        #CHECK: Check reconstruction loss accumulation logic
        reconstruction_loss = to.distributions.normal.Normal(
            loc=observation_decoder_mean, scale=to.exp(0.5 * observation_decoder_log_cov)).log_prob(data).sum(dim=1).squeeze(dim=1)

        self.context_divergence_history.append(context_divergence.sum().item())
        self.latent_divergence_history.append(latent_divergence.sum().item())
        self.reconstruction_loss_history.append(-reconstruction_loss.sum().item())
        self.loss_history.append(self.context_divergence_history[-1] +
                                 self.latent_divergence_history[-1] +
                                 self.reconstruction_loss_history[-1])

        self.counter += 1
        if self.counter % 625 == 0:
            plt.plot(self.context_divergence_history, 'r')
            plt.plot(self.latent_divergence_history, 'g')
            plt.plot(self.reconstruction_loss_history, 'b')
            plt.plot(self.loss_history, 'k')
            plt.show()


        #Logically, it makes sense to keep the divergences separate up until here. 
        #But we can probably optimize that
        return (context_divergence + latent_divergence - reconstruction_loss).sum(dim=0)


    def predict(self, data):
        #Here, we're recieving a tuple with one tensor in it. The tensor is what we need to 
        # split out to get to the mean and log_var
        statistic_net_outputs = self.statistic_network(data)
        contexts = self.reparameterise_normal(*statistic_net_outputs)

        inference_net_outputs = [self.inference_networks[0](data, contexts)]
        latent_dec_outputs = [self.latent_decoders[0](contexts)]
        latent_z = [self.reparameterise_normal(*inference_net_outputs[0])]
        for inference_network, latent_decoder in zip(self.inference_networks[1:], self.latent_decoders[1:]):
            inference_net_outputs.append(inference_network(data, contexts, latent_z[-1]))
            latent_dec_outputs.append(latent_decoder(contexts, latent_z[-1]))
            latent_z.append(*self.reparameterise_normal(inference_net_outputs[-1]))

        observation_dec_outputs = self.observation_decoder(to.cat(latent_z, dim=2), contexts)

        return statistic_net_outputs, inference_net_outputs, latent_dec_outputs, observation_dec_outputs


    def reparameterise_normal(self, mean, log_var):
        """Draw samples from the given normal distribution via the
        reparameterisation trick"""
        std_errors = to.randn(log_var.size())
        return mean + to.exp(0.5 * log_var) * std_errors
        

    def train(self, dataloader, num_iterations, optimiser_func):
        """Train the Neural Statistician"""

        network_parameters = []
        for decoder in self.latent_decoders:
            network_parameters.extend(decoder.parameters())
        network_parameters.extend(self.observation_decoder.parameters())
        network_parameters.extend(self.statistic_network.parameters())
        for network in self.inference_networks:
            network_parameters.extend(network.parameters())

        optimiser = optimiser_func(network_parameters)

        for iteration in range(num_iterations):
            print("Commencing iteration {}/{}...".format(iteration+1, num_iterations))
            for data_batch in dataloader:
                distribution_parameters = self.predict(data_batch['dataset'])
                loss = self.compute_loss(*distribution_parameters, data=data_batch['dataset'])
                print("        Batch loss: {}".format(loss))

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()


    def serialise(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)


    @staticmethod
    def deserialise(path):
        with open(path, 'rb') as file:
            pickle.load(file)


    @staticmethod
    def init_weights(m):
        to.nn.init.xavier_normal(m.weight.data, gain=to.nn.init.calculate_gain('relu'))
        to.nn.init.constant(m.bias.data, 0)

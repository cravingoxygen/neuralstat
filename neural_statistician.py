import torch as to
import torch.nn.functional as F
import numpy as np
import pickle


class NeuralStatistician(object):
    """Tying-together class to hold references for a particular experiment"""

    def __init__(self, num_stochastic_layers, context_dimension,
                 LatentDecoder, ObservationDecoder, StatisticNetwork, InferenceNetwork):
        super().__init__()

        self.latent_decoders = [LatentDecoder() for _ in range(num_stochastic_layers)]
        self.observation_decoder = ObservationDecoder()
        self.statistic_network = StatisticNetwork()
        self.inference_networks = [InferenceNetwork() for _ in range(num_stochastic_layers)]

        self.context_prior_mean = to.zeros(context_dimension)
        # CHECK: Is it OK to restrict ourselvs to a diagonal covariance matrix for the prior?
        self.context_prior_cov = to.ones(context_dimension)
        self.context_prior = to.distributions.multivariate_normal.MultivariateNormal(
            loc=self.context_prior_mean, covariance_matrix=to.diag(self.context_prior_cov))

    def normal_kl_divergence(self, mean_0, diag_cov_0, mean_1, diag_cov_1):
        """Compute the KL divergence between two diagonal Gaussians, where
        diag_cov_x is a 1D vector containing the diagonal elements of the
        xth covariance matrix."""
        
        #Use batch multiply which requires reshaping the data to get desired dot products
        #TODO: Sanity check these
        batch_size = mean_0.shape[0]
        return 0.5 * (
            to.bmm((1 / diag_cov_1).view(batch_size, 1, -1), diag_cov_0.view(batch_size, -1, 1)) + 
            to.bmm(((mean_1 - mean_0).view(batch_size, 1, -1) ** 2), (1 / diag_cov_1).view(batch_size, -1, 1)) - 
            mean_0.shape[-1] +
            to.sum(to.log(diag_cov_1).view(batch_size, 1, -1), dim=-1, keepdim=True) - to.sum(to.log(diag_cov_0).view(batch_size, 1, -1), dim=-1, keepdim=True)
        ).squeeze()

    def compute_loss(self, context_output, inference_outputs, decoder_outputs, observation_decoder_outputs, data):
        """Compute the full model loss function"""
        # Context divergence
        context_mean, context_log_cov = context_output
        context_divergence = self.normal_kl_divergence(context_mean, to.exp(context_log_cov),
                                                       self.context_prior_mean.expand_as(context_mean), self.context_prior_cov.expand_as(context_log_cov))

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
                distribution_parameters = self.predict(data_batch)
                loss = self.compute_loss(*distribution_parameters, data=data_batch)
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


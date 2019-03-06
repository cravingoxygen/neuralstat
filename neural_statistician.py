import torch as to
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm, trange

import matplotlib.pyplot as plt


class NeuralStatistician(to.nn.Module):
    """Tying-together class to hold references for a particular experiment"""

    def __init__(self, num_stochastic_layers, context_dimension, 
                 LatentDecoder, ObservationDecoder, StatisticNetwork, InferenceNetwork,
                 device="cpu"):
        super().__init__()

        self.device = device

        self.latent_decoders = to.nn.ModuleList([LatentDecoder().to(self.device) for _ in range(num_stochastic_layers)])
        self.observation_decoder = ObservationDecoder().to(self.device)
        self.statistic_network = StatisticNetwork().to(self.device)
        self.inference_networks = to.nn.ModuleList([InferenceNetwork().to(self.device) for _ in range(num_stochastic_layers)])

        for network in self.latent_decoders:
            network.apply(NeuralStatistician.init_weights)
        self.observation_decoder.apply(NeuralStatistician.init_weights)
        self.statistic_network.apply(NeuralStatistician.init_weights)
        for network in self.inference_networks:
            network.apply(NeuralStatistician.init_weights)

        self.context_prior_mean = to.zeros(context_dimension, device=self.device)
        self.context_prior_cov = to.ones(context_dimension, device=self.device)

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
        batch_size = data.shape[0]
        sample_size = data.shape[1]

        # Context divergence
        context_mean, context_log_cov = context_output
        # Handle this case without separate data points by introducing a dummy
        # dimension, as if there were exactly 1 data point
        context_divergence = self.normal_kl_divergence(context_mean.unsqueeze(dim=1), to.exp(context_log_cov).unsqueeze(dim=1),
                                                       self.context_prior_mean.expand_as(context_mean.unsqueeze(dim=1)), self.context_prior_cov.expand_as(context_log_cov.unsqueeze(dim=1)))
        # context_divergence *= sample_size

        # Latent divergence
        # For computational efficiency, draw a single sample context from q(c, z | D, phi)
        # rather than computing the expectation properly.
        latent_divergence = to.zeros(context_divergence.shape, device=self.device)
        for ((inference_mu, inference_log_cov), (decoder_mu, decoder_log_cov)) in zip(inference_outputs, decoder_outputs):
            #Expand the decoder's outputs to be the same shape as the inference networks
            # i.e. batch_size x dataset_size x data_dimensionality (for mean and log_var)
            latent_divergence += self.normal_kl_divergence(inference_mu, to.exp(inference_log_cov),
                                                           decoder_mu.expand_as(inference_mu), to.exp(decoder_log_cov).expand_as(inference_log_cov))

        # Reconstruction loss
        observation_decoder_mean, observation_decoder_log_cov = observation_decoder_outputs
        
        #CHECK: Check reconstruction loss accumulation logic
        #CHECK: For 2D decoder outputs, can we just sum over dimensions?
        reconstruction_loss = to.distributions.normal.Normal(
            loc=observation_decoder_mean, scale=to.exp(0.5 * observation_decoder_log_cov)).log_prob(data).sum(dim=2).sum(dim=1)

        self.context_divergence_history.append(context_divergence.sum().item())
        self.latent_divergence_history.append(latent_divergence.sum().item())
        self.reconstruction_loss_history.append(-reconstruction_loss.sum().item())
        self.loss_history.append(self.context_divergence_history[-1] +
                                 self.latent_divergence_history[-1] +
                                 self.reconstruction_loss_history[-1])

        self.counter += 1
        # if self.counter % 625 == 0:
        #     plt.figure()
        #     plt.plot(range(self.counter), self.context_divergence_history, 'r')
        #     plt.plot(range(self.counter), self.latent_divergence_history, 'g')
        #     plt.plot(range(self.counter), self.reconstruction_loss_history, 'b')
        #     plt.plot(range(self.counter), self.loss_history, 'k')
        #     plt.show()


        #Logically, it makes sense to keep the divergences separate up until here. 
        #But we can probably optimize that
        return (context_divergence + latent_divergence - reconstruction_loss).sum(dim=0) / (sample_size * batch_size)


    def predict(self, data):
        #Here, we're recieving a tuple with one tensor in it. The tensor is what we need to 
        # split out to get to the mean and log_var
        statistic_net_outputs = self.statistic_network(data)
        contexts = self.reparameterise_normal(*statistic_net_outputs)

        inference_net_outputs = [self.inference_networks[0](data, contexts, None)]
        latent_dec_outputs = [self.latent_decoders[0](contexts, None)]
        latent_z = [self.reparameterise_normal(*inference_net_outputs[0])]
        for inference_network, latent_decoder in zip(self.inference_networks[1:], self.latent_decoders[1:]):
            inference_net_outputs.append(inference_network(data, contexts, latent_z[-1]))
            latent_dec_outputs.append(latent_decoder(contexts, latent_z[-1]))
            latent_z.append(self.reparameterise_normal(*inference_net_outputs[-1]))

        observation_dec_outputs = self.observation_decoder(to.cat(latent_z, dim=2), contexts)

        return statistic_net_outputs, inference_net_outputs, latent_dec_outputs, observation_dec_outputs


    def reparameterise_normal(self, mean, log_var):
        """Draw samples from the given normal distribution via the
        reparameterisation trick"""
        std_errors = to.randn(log_var.size(), device=self.device)
        # No-variance check
        # return mean + 1e-5 * std_errors
        return mean + to.exp(0.5 * log_var) * std_errors


    def generate_like(self, data):
        pass
        #TODO: Check this works for Spatial MNIST
        #Here, we're recieving a tuple with one tensor in it. The tensor is what we need to 
        # split out to get to the mean and log_var
        statistic_net_outputs = self.statistic_network(data)
        contexts = self.reparameterise_normal(*statistic_net_outputs)
        
        inference_net_outputs = [self.inference_networks[0](data, contexts, None)]
        latent_dec_outputs = [self.latent_decoders[0](contexts, None)]
        latent_z = [self.reparameterise_normal(*inference_net_outputs[0])]
        for inference_network, latent_decoder in zip(self.inference_networks[1:], self.latent_decoders[1:]):
            inference_net_outputs.append(inference_network(data, contexts, latent_z[-1]))
            latent_dec_outputs.append(latent_decoder(contexts, latent_z[-1]))
            latent_z.append(*self.reparameterise_normal(inference_net_outputs[-1]))

        observation_dec_outputs = self.observation_decoder(to.cat(latent_z, dim=2), contexts)
        samples = self.reparameterise_normal(*observation_dec_outputs).squeeze()

        return samples


    def run_training(self, dataloader, num_iterations, optimiser_func, test_func, device="cpu"):
        """Train the Neural Statistician"""

        network_parameters = []
        for decoder in self.latent_decoders:
            network_parameters.extend(decoder.parameters())
        network_parameters.extend(self.observation_decoder.parameters())
        network_parameters.extend(self.statistic_network.parameters())
        for network in self.inference_networks:
            network_parameters.extend(network.parameters())

        optimiser = optimiser_func(network_parameters)

        for iteration in trange(num_iterations):
            with tqdm(dataloader, unit="bch") as progress:
                for data_batch in progress:
                    data = data_batch['dataset'].to(device)
                    distribution_parameters = self.predict(data)
                    loss = self.compute_loss(*distribution_parameters, data=data)
                    progress.set_postfix(loss=loss.item())
                    
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
            test_func(self, iteration)


    def serialise(self, path):
        save_dict = self.state_dict().copy()
        save_dict['context_divergence_history'] = self.context_divergence_history
        save_dict['latent_divergence_history'] = self.latent_divergence_history
        save_dict['reconstruction_loss_history'] = self.reconstruction_loss_history
        save_dict['loss_history'] = self.loss_history

        to.save(save_dict, path)


    def deserialise(self, path):
        save_dict = to.load(path)
        self.context_divergence_history = save_dict.pop('context_divergence_history')
        self.latent_divergence_history = save_dict.pop('latent_divergence_history')
        self.reconstruction_loss_history = save_dict.pop('reconstruction_loss_history')
        self.loss_history = save_dict.pop('loss_history')

        self.load_state_dict(save_dict)
        self.eval()


    @staticmethod
    def init_weights(m):
        if type(m) == to.nn.Linear:
            to.nn.init.xavier_normal_(m.weight.data, gain=to.nn.init.calculate_gain('relu'))
            to.nn.init.constant_(m.bias.data, 0)

import torch as to
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import os
import datetime

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import spatialdata as spd
import neural_statistician as ns

data_dir = "spatial_data/spatial"

device = to.device("cuda")
context_dimension = 64
dense_layer_size = 256
x_dimension = 2
z_dimension = 2
num_stochastic_layers = 3

### p(z_i | z_(i+1), c) parameterised by theta
class LatentDecoder(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(z_i) + dim(c)
        self.dense1 = to.nn.Linear(z_dimension + context_dimension, dense_layer_size)
        self.batchnorm1 = to.nn.BatchNorm1d(dense_layer_size)
        self.dense2 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm2 = to.nn.BatchNorm1d(dense_layer_size)
        self.dense3 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm3 = to.nn.BatchNorm1d(dense_layer_size)

        # Output is dim(z) for the mean and dim(z) for the variance
        self.final = to.nn.Linear(dense_layer_size, 2 * z_dimension)

    def forward(self, c, z):
        if z is None:
            z = to.zeros(c.shape[0], 1, z_dimension, device=device)
        c = c.unsqueeze(dim=1).expand(-1, z.shape[1], -1)

        #Concatenate for each batch
        w = to.cat((c, z), dim=2)
        w = self.dense1(w)
        w = apply_batch_norm(self.batchnorm1, w)
        w = F.relu(w)

        w = self.dense2(w)
        w = apply_batch_norm(self.batchnorm2, w)
        w = F.relu(w)
        
        w = self.dense3(w)
        w = apply_batch_norm(self.batchnorm3, w)
        w = F.relu(w)

        w = self.final(w)
        # We've now computed mu_z and log var_c
        return w[:, :, :z_dimension], w[:, :, z_dimension:]


### p(x | z_(1:L), c) parameterised by theta
class ObservationDecoder(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(z)* num_z_layers + dim(c)
        self.dense1 = to.nn.Linear(z_dimension*num_stochastic_layers + context_dimension, dense_layer_size)
        self.batchnorm1 = to.nn.BatchNorm1d(dense_layer_size)
        self.dense2 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm2 = to.nn.BatchNorm1d(dense_layer_size)
        self.dense3 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm3 = to.nn.BatchNorm1d(dense_layer_size)

        # Output is dim(x) for the mean and dim(x) for the variance
        self.final = to.nn.Linear(dense_layer_size, 2 * x_dimension)

    def forward(self, z, c):
        """Computes x from a concatenation (w) of latent variables z and context c."""
        # Augment every data point in x with the context vector for that dataset
        w = to.cat((c.unsqueeze(dim=1).expand(-1, z.shape[1], -1),
                    z),
                   dim=2)

        w = self.dense1(w)
        w = apply_batch_norm(self.batchnorm1, w)
        w = F.relu(w)

        w = self.dense2(w)
        w = apply_batch_norm(self.batchnorm2, w)
        w = F.relu(w)
        
        w = self.dense3(w)
        w = apply_batch_norm(self.batchnorm3, w)
        w = F.relu(w)

        w = self.final(w)
        
        # We've now computed mu_x and log var_x
        return w[:, :, x_dimension:], w[:, :, :x_dimension]


### q(c | D) parameterised by phi
class StatisticNetwork(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is whole dataset; a batch of dim(x)
        self.embed_dense1 = to.nn.Linear(x_dimension, dense_layer_size)
        self.batchnorm1 = to.nn.BatchNorm1d(dense_layer_size)
        self.embed_dense2 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm2 = to.nn.BatchNorm1d(dense_layer_size)
        self.embed_dense3 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm3 = to.nn.BatchNorm1d(dense_layer_size)

        self.post_pool_dense1 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm4 = to.nn.BatchNorm1d(dense_layer_size)
        self.post_pool_dense2 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm5 = to.nn.BatchNorm1d(dense_layer_size)
        # Output is Gaussian parameters for c
        self.post_pool_dense3 = to.nn.Linear(dense_layer_size, 2 * context_dimension)


    def forward(self, dataset):
        dataset = self.embed_dense1(dataset)
        dataset = apply_batch_norm(self.batchnorm1, dataset)
        dataset = F.relu(dataset)
        dataset = self.embed_dense2(dataset)
        dataset = apply_batch_norm(self.batchnorm2, dataset)
        dataset = F.relu(dataset)
        dataset = self.embed_dense3(dataset)
        dataset = apply_batch_norm(self.batchnorm3, dataset)
        dataset = F.relu(dataset)

        dataset = dataset.mean(dim=1)

        dataset = self.post_pool_dense1(dataset)
        dataset = apply_batch_norm(self.batchnorm4, dataset)
        dataset = F.relu(dataset)
        dataset = self.post_pool_dense2(dataset)
        dataset = apply_batch_norm(self.batchnorm5, dataset)
        dataset = F.relu(dataset)
        dataset = self.post_pool_dense3(dataset)

        # Output means and variances, in that order
        return dataset[:, :context_dimension], dataset[:, context_dimension:]


### q(z_i | z_(i+1), c, x) parameterised by phi
class InferenceNetwork(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(z_prev) + dim(c) + dim(x);
        self.dense1 = to.nn.Linear(z_dimension + context_dimension + x_dimension, dense_layer_size)
        self.batchnorm1 = to.nn.BatchNorm1d(dense_layer_size)
        self.dense2 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm2 = to.nn.BatchNorm1d(dense_layer_size)
        self.dense3 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm3 = to.nn.BatchNorm1d(dense_layer_size)

        # Outputs a mean and diagonal variance for the Gaussian
        # distribution defining z
        self.final = to.nn.Linear(dense_layer_size, 2 * z_dimension)

    def forward(self, x, c, z):
        """Computes x from a concatenation (w) of latent variables z_prev and context c."""
        if z is None:
            z = to.zeros(x.shape[0], x.shape[1], z_dimension, device=device)
        
        # Augment every data point in x with the context vector for that dataset
        w = to.cat((c.unsqueeze(dim=1).expand(-1, x.shape[1], -1),
                    x, z),
                   dim=2)

        w = self.dense1(w)
        w = apply_batch_norm(self.batchnorm1, w)
        w = F.relu(w)

        w = self.dense2(w)
        w = apply_batch_norm(self.batchnorm2, w)
        w = F.relu(w)
        
        w = self.dense3(w)
        w = apply_batch_norm(self.batchnorm3, w)
        w = F.relu(w)

        w = self.final(w)
        # We've now computed mu_x and log sigma_x
        return w[:, :, :z_dimension], w[:, :, z_dimension:]
    

def apply_batch_norm(batch_norm, data):
    original_shape = data.shape
    data_list = data.view(-1, batch_norm.num_features)
    data_list = batch_norm(data_list)
    return data_list.view(original_shape)


def generate_samples_like(network, datasets, timestamp, device, iteration=0):
    with to.no_grad():
        # Select an example of each digit from the dataset
        sample_digits = [None] * 10
        for dataset in datasets:
            if not any(x is None for x in sample_digits):
                break
            label = np.argmax(dataset['label'])
            if sample_digits[label] is None:
                sample_digits[label] = dataset['dataset']

        for digit in sample_digits:
            # Make batch_norm work properly by placing it in evaluation mode
            network.eval()
            samples = network.generate_like(to.as_tensor(digit).unsqueeze(dim=0).to(device))
            # Reset network to training mode
            network.train()
            import pdb; pdb.set_trace()


def visualize_data(network, dataset, iteration, timestamp, device):
    generate_samples_like(network, dataset, timestamp, device, iteration=iteration)


def main():
    timestamp = datetime.datetime.now()

    optimiser_func = lambda parameters: to.optim.Adam(parameters, lr=1e-3)
    
    train_dataset = spd.SpatialMNISTDataset(data_dir, split='train')
    test_dataset = spd.SpatialMNISTDataset(data_dir, split='test')
    train_dataloader = to.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    
    test_func = lambda network, iteration: visualize_data(network, test_dataset, iteration, timestamp, device)

    network = ns.NeuralStatistician(num_stochastic_layers, context_dimension, LatentDecoder, ObservationDecoder, StatisticNetwork, InferenceNetwork, device)
    
    test_func(network, 0)
    network.run_training(train_dataloader, 50, optimiser_func, test_func, device)

    network.serialise("results/{}/trained_mnist_model".format(timestamp))

    return network


if __name__ == '__main__':
    network = main()

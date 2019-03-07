import torch as to
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import os
import datetime
import pickle

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cmx

import neural_statistician as ns

context_dimension = 3

### p(z_i | z_(i+1), c) parameterised by theta
class LatentDecoder(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(c); there's no z_(i+1) in this application
        self.dense1 = to.nn.Linear(context_dimension, 128)
        self.dense2 = to.nn.Linear(128, 128)
        self.dense3 = to.nn.Linear(128, 128)

        # Output is dim(z) for the mean and dim(z) for the variance
        self.final = to.nn.Linear(128, 2 * 32)

    def forward(self, c, z):
        # Only one z is used in this test, so ignore z
        # Expand c, as in general we need 3D tensors here
        c = c.unsqueeze(dim=1)

        w = self.dense1(c)
        w = F.relu(w)

        w = self.dense2(w)
        w = F.relu(w)
        
        w = self.dense3(w)
        w = F.relu(w)

        w = self.final(w)
        # We've now computed mu_z and log var_c
        return w[:, :, :32], w[:, :, 32:]


### p(x | z_(1:L), c) parameterised by theta
class ObservationDecoder(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(c) + dim(z)
        self.dense1 = to.nn.Linear(32 + context_dimension, 128)
        self.dense2 = to.nn.Linear(128, 128)
        self.dense3 = to.nn.Linear(128, 128)

        # Output is dim(x) for the mean and dim(x) for the variance
        self.final = to.nn.Linear(128, 2 * 1)

    def forward(self, z, c):
        """Computes x from a concatenation (w) of latent variables z and context c."""
        # Augment every latent point in z with the context vector for that dataset
        # CHECK: Is this correct?
        w = to.cat((z,
                    c.unsqueeze(dim=1).expand(-1, z.shape[1], -1)),
                   dim=2)

        w = self.dense1(w)
        w = F.relu(w)

        w = self.dense2(w)
        w = F.relu(w)
        
        w = self.dense3(w)
        w = F.relu(w)

        w = self.final(w)
        
        # We've now computed mu_x and log var_x
        return w[:, :, 0].unsqueeze(2), w[:, :, 1].unsqueeze(2)


### q(c | D) parameterised by phi
class StatisticNetwork(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is whole dataset; a batch of dim(x)
        self.embed_dense1 = to.nn.Linear(1, 128)
        self.embed_dense2 = to.nn.Linear(128, 128)
        self.embed_dense3 = to.nn.Linear(128, 128)

        self.post_pool_dense1 = to.nn.Linear(128, 128)
        self.post_pool_dense2 = to.nn.Linear(128, 128)
        # Output is Gaussian parameters for c
        self.post_pool_dense3 = to.nn.Linear(128, 2 * context_dimension)

    def forward(self, dataset):
        dataset = self.embed_dense1(dataset)
        dataset = F.relu(dataset)
        dataset = self.embed_dense2(dataset)
        dataset = F.relu(dataset)
        dataset = self.embed_dense3(dataset)
        dataset = F.relu(dataset)

        dataset = dataset.mean(dim=1)

        dataset = self.post_pool_dense1(dataset)
        dataset = F.relu(dataset)
        dataset = self.post_pool_dense2(dataset)
        dataset = F.relu(dataset)
        dataset = self.post_pool_dense3(dataset)

        # Output means and variances, in that order
        return dataset[:, :context_dimension], dataset[:, context_dimension:]


### q(z_i | z_(i+1), c, x) parameterised by phi
class InferenceNetwork(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(c) + dim(x); there is no z_previous in this problem
        self.dense1 = to.nn.Linear(context_dimension + 1, 128)
        self.dense2 = to.nn.Linear(128, 128)
        self.dense3 = to.nn.Linear(128, 128)

        # z is 32-D, outputting a mean and diagonal variance for the Gaussian
        # distribution defining z
        self.final = to.nn.Linear(128, 2 * 32)

    def forward(self, x, c, z):
        """Computes x from a concatenation (w) of latent variables z and context c."""
        # Augment every data point in x with the context vector for that dataset
        # CHECK: Is this correct?
        w = to.cat((c.unsqueeze(dim=1).expand(-1, x.shape[1], -1),
                    x),
                   dim=2)

        w = self.dense1(w)
        w = F.relu(w)

        w = self.dense2(w)
        w = F.relu(w)
        
        w = self.dense3(w)
        w = F.relu(w)

        w = self.final(w)
        # We've now computed mu_x and log sigma_x
        return w[:, :, :32], w[:, :, 32:]


class OneDimDataset(to.utils.data.Dataset):
    def __init__(self, number_of_datasets=10000, number_of_samples=200):
        super(OneDimDataset, self).__init__()
        data = np.zeros((number_of_datasets, number_of_samples))
        self.means = np.random.uniform(-1, 1, number_of_datasets)
        self.variances = np.random.uniform(0.5, 2, number_of_datasets)
        
        block_size = int(number_of_datasets/4)
        
        data[0:block_size] = np.random.exponential(np.sqrt(self.variances[0:block_size]), (number_of_samples, block_size)).T
        data[block_size:block_size*2] = np.random.normal(self.means[block_size:block_size*2], np.sqrt(self.variances[block_size:block_size*2]), (number_of_samples, block_size)).T
        data[block_size*2:block_size*3] = np.random.uniform(self.means[block_size*2:block_size*3] - np.sqrt(3*self.variances[block_size*2:block_size*3]),
                                            self.means[block_size*2:block_size*3] + np.sqrt(3*self.variances[block_size*2:block_size*3]), (number_of_samples, block_size)).T
        data[block_size*3:block_size*4] = np.random.laplace(self.means[block_size*3:block_size*4], np.sqrt(self.variances[block_size*3:block_size*4]/2), (number_of_samples, block_size)).T

        data = [to.as_tensor(ds.reshape(number_of_samples,1), dtype=to.float) for ds in data]
        self.data = data
        self.block_size = block_size
        
    def __getitem__(self, index):
        return {'dataset': self.data[index],
                'label': int(index/self.block_size),
                'mean': self.means[index],
                'variance': self.variances[index]}

    def __len__(self):
        return len(self.data)


def plot_contexts_by_distribution(network, timestamp, device, dataset=OneDimDataset(4000, 200), iteration=0, save_plot=True, config=lambda: None):
    """Plot the context means in context space, coloured by the distribution of the original dataset."""
    with to.no_grad():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        dataloader = to.utils.data.DataLoader(dataset, batch_size=dataset.block_size, shuffle=False)

        colours = ['b', 'r', 'y', 'g']
        max_points = min(200, len(dataset[0]["dataset"]))
        for batch, colour in zip(dataloader, colours):
            data = batch["dataset"].to(device)
            statistic_net_outputs = network.predict(data[:max_points])[0]
            context_means = statistic_net_outputs[0].to("cpu") # Needed for numpy use below
            ax.scatter(context_means[:, 0], context_means[:, 1], context_means[:,2], c=colour)

        config()
        if save_plot:
            path = "results/{}".format(timestamp)
            try: os.mkdir(path)
            except FileExistsError: pass
            plt.savefig("{}/contexts_iteration_{}".format(path, iteration))
            pickle.dump(fig, open("{}/contexts_iteration_{}.plt".format(path, iteration), 'wb'))
            plt.close()
        else:
            plt.show()


def plot_contexts_by_value(network, device, value, dataset=OneDimDataset(4000, 200)):
    """Plot the context means in context space, coloured by the mean of the original dataset."""
    with to.no_grad():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        dataloader = to.utils.data.DataLoader(dataset, batch_size=len(dataset))

        for batch in dataloader:
            data = batch["dataset"].to(device)
            context_means = network.predict(data)[0][0].to("cpu") # Needed for numpy use below
            dataset_values = batch[value]
            colours = matplotlib.colors.Normalize(vmin=to.min(dataset_values), vmax=to.max(dataset_values))
            scalar_map = cmx.ScalarMappable(norm=colours, cmap='viridis')
            points = ax.scatter(context_means[:, 0], context_means[:, 1], context_means[:,2], c=scalar_map.to_rgba(dataset_values))
            
        scalar_map.set_array(dataset_values)
        plt.colorbar(scalar_map)
        plt.show()
        
            
def generate_samples_like(network, datasets, timestamp, device, iteration=0):
    with to.no_grad():
        fig = plt.figure()
        
        test_datasets = [{"distribution": "exponential", "data":datasets[0]["dataset"]},
            {"distribution": "normal", "data":datasets[datasets.block_size]["dataset"]}, 
            {"distribution": "uniform", "data":datasets[datasets.block_size*2]["dataset"]}, 
            {"distribution": "laplace", "data":datasets[datasets.block_size*3]["dataset"]},
            ]
            
        for single_dataset in test_datasets:
            reshaped_dataset = single_dataset["data"].to(device).view(1, -1, 1)
            samples = network.generate_like(reshaped_dataset).to("cpu") # Needed for numpy use below
            plt.hist(samples, bins=100)
            plt.savefig("results/{}/{}_samples_{}".format(timestamp, single_dataset["distribution"], iteration))
            plt.close()


def visualize_data(network, dataset, iteration, timestamp, device):
    plot_contexts_by_distribution(network, iteration=iteration, device=device, timestamp=timestamp)
    generate_samples_like(network, dataset, timestamp, device, iteration=iteration)
    if iteration in [0, 25]:
        network.serialise("results/{}/model_iteration_{}".format(timestamp, iteration))


def main():
    device = to.device("cuda")
    timestamp = datetime.datetime.now()

    optimiser_func = lambda parameters: to.optim.Adam(parameters, lr=1e-3)
    dataset = OneDimDataset()
    test_func = lambda network, iteration: visualize_data(network, dataset, iteration, timestamp, device)
    dataloader = to.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    network = ns.NeuralStatistician(1, 3, 
                                    LatentDecoder, ObservationDecoder, StatisticNetwork, InferenceNetwork,
                                    device)
    test_func(network, 0)
    network.run_training(dataloader, 50, optimiser_func, test_func, device)

    network.serialise("results/{}/trained_model".format(timestamp))

    return network


if __name__ == '__main__':
    network = main()

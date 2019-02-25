import torch as to
import torch.utils.data
import torch.nn.functional as F
import numpy as np

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

    def forward(self, c):
        w = self.dense1(c)
        w = F.relu(w)

        w = self.dense2(w)
        w = F.relu(w)
        
        w = self.dense3(w)
        w = F.relu(w)

        w = self.final(w)
        # We've now computed mu_z and log var_c
        return w


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
        w = to.stack((z, c), dim=1)

        w = self.dense1(w)
        w = F.relu(w)

        w = self.dense2(w)
        w = F.relu(w)
        
        w = self.dense3(w)
        w = F.relu(w)

        w = self.final(w)
        # We've now computed mu_x and log var_x
        return w[:, 0], w[:, 1]


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

    def forward(self, x, c):
        """Computes x from a concatenation (w) of latent variables z and context c."""
        # Augment every data point in x with the context vector for that dataset
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
        return w[:, :32], w[:, 32:]


class OneDimensionalDataset(to.utils.data.TensorDataset):
    def __init__(self):
        data = np.zeros((10000, 200))
        means = np.random.uniform(-1, 1, 10000)
        variances = np.random.uniform(0.5, 2, 10000)

        data[0:2500] = np.random.exponential(np.sqrt(variances[0:2500]), (200, 2500)).T
        data[2500:5000] = np.random.normal(means[2500:5000], np.sqrt(variances[2500:5000]), (200, 2500)).T
        data[5000:7500] = np.random.uniform(means[5000:7500] - np.sqrt(3*variances[5000:7500]),
                                            means[5000:7500] + np.sqrt(3*variances[5000:7500]), (200, 2500)).T
        data[7500:10000] = np.random.laplace(means[7500:10000], np.sqrt(variances[7500:10000]/2), (200, 2500)).T

        data = to.tensor(data.reshape((10000, 200, 1)))
        super().__init__(data)


class OneDimDataset(to.utils.data.Dataset):
    def __init__(self):
        super(OneDimDataset, self).__init__()
        data = np.zeros((10000, 200))
        means = np.random.uniform(-1, 1, 10000)
        variances = np.random.uniform(0.5, 2, 10000)

        data[0:2500] = np.random.exponential(np.sqrt(variances[0:2500]), (200, 2500)).T
        data[2500:5000] = np.random.normal(means[2500:5000], np.sqrt(variances[2500:5000]), (200, 2500)).T
        data[5000:7500] = np.random.uniform(means[5000:7500] - np.sqrt(3*variances[5000:7500]),
                                            means[5000:7500] + np.sqrt(3*variances[5000:7500]), (200, 2500)).T
        data[7500:10000] = np.random.laplace(means[7500:10000], np.sqrt(variances[7500:10000]/2), (200, 2500)).T

        data = [to.as_tensor(ds.reshape(200,1), dtype=to.float) for ds in data]
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def main():
    optimiser_func = lambda parameters: to.optim.Adam(parameters,
                                                      lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    dataset = OneDimDataset()
    
    dataloader = to.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    network = ns.NeuralStatistician(1, 3,
                                    LatentDecoder, ObservationDecoder, StatisticNetwork, InferenceNetwork)
    network.train(dataloader, 100, optimiser_func)


if __name__ == '__main__':
    main()

import torch as to
import torch.nn.functional as F

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
        # We've now computed mu_z and log sigma_c
        return w


### p(x | z_(1:L), c) parameterised by theta
class ObservationDecoder(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(z) + dim(z)
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
        # We've now computed mu_x and log sigma_x
        return w


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

        dataset = dataset.mean(dim=0)

        dataset = self.post_pool_dense1(dataset)
        dataset = F.relu(dataset)
        dataset = self.post_pool_dense2(dataset)
        dataset = F.relu(dataset)
        dataset = self.post_pool_dense3(dataset)

        return dataset


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
        w = to.stack((c, x), dim=1)

        w = self.dense1(w)
        w = F.relu(w)

        w = self.dense2(w)
        w = F.relu(w)
        
        w = self.dense3(w)
        w = F.relu(w)

        w = self.final(w)
        # We've now computed mu_x and log sigma_x
        return w

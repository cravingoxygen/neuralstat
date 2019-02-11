import torch as to
import torch.nn.functional as F

class LatentDecoder(to.nn.Module):
    pass


class ObservationDecoder(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(z) + dim(z)
        self.dense1 = to.nn.Linear(32 + 3, 128)
        self.dense2 = to.nn.Linear(128, 128)
        self.dense3 = to.nn.Linear(128, 128)

        # Output is dim(x) for the man and dim(x) for the variance
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


class StatisticNetwork(to.nn.Module):
    pass


class InferenceNetwork(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(c) + dim(x); there is no z_previous in this problem
        self.dense1 = to.nn.Linear(3 + 1, 128)
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




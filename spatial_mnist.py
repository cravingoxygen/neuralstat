import torch as to
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import os
import datetime

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import spatialdata as spd
import spatialcreate as sc
import neural_statistician as ns
import label_statistician as ls


data_dir = "spatial_data/spatial"

device = to.device("cuda")
context_dimension = 64
dense_layer_size = 256
x_dimension = 2
z_dimension = 2
num_stochastic_layers = 3
num_y_labels = 10

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
    def __init__(self, accept_labels):
        super().__init__()

        # Input is whole dataset; a batch of dim(x) and labels of dim(y)
        input_dim = x_dimension
        if accept_labels:
            input_dim += num_y_labels
        self.embed_dense1 = to.nn.Linear(input_dim, dense_layer_size)
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


    def forward(self, dataset, labels=None):
        if labels is None:
            input_data = dataset
        else:
            input_data = to.cat((dataset, labels.unsqueeze(dim=1).expand(-1, dataset.shape[1], -1)),
                                dim=2)

        result = self.embed_dense1(input_data)
        result = apply_batch_norm(self.batchnorm1, result)
        result = F.relu(result)
        result = self.embed_dense2(result)
        result = apply_batch_norm(self.batchnorm2, result)
        result = F.relu(result)
        result = self.embed_dense3(result)
        result = apply_batch_norm(self.batchnorm3, result)
        result = F.relu(result)

        result = result.mean(dim=1)

        result = self.post_pool_dense1(result)
        result = apply_batch_norm(self.batchnorm4, result)
        result = F.relu(result)
        result = self.post_pool_dense2(result)
        result = apply_batch_norm(self.batchnorm5, result)
        result = F.relu(result)
        result = self.post_pool_dense3(result)

        # Output means and variances, in that order
        return result[:, :context_dimension], result[:, context_dimension:]


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


### q(y | x) parameterised by phi
class ClassificationNetwork(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(x)
        self.dense1 = to.nn.Linear(x_dimension, dense_layer_size)
        self.batchnorm1 = to.nn.BatchNorm1d(dense_layer_size)
        self.dense2 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm2 = to.nn.BatchNorm1d(dense_layer_size)
        self.dense3 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm3 = to.nn.BatchNorm1d(dense_layer_size)

        # No batch norming here, because we might have batches of size 1
        self.post_pool_dense1 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.post_pool_dense2 = to.nn.Linear(dense_layer_size, dense_layer_size)
        # Output a softmaxed distribution over labels y
        self.final = to.nn.Linear(dense_layer_size, num_y_labels)

    def forward(self, x):
        """Computes a distribution over labels y from input data x"""
        y = self.dense1(x)
        y = apply_batch_norm(self.batchnorm1, y)
        y = F.relu(y)
        y = self.dense2(y)
        y = apply_batch_norm(self.batchnorm2, y)
        y = F.relu(y)
        y = self.dense3(y)
        y = apply_batch_norm(self.batchnorm3, y)
        y = F.relu(y)

        y = y.mean(dim=1)

        y = self.post_pool_dense1(y)
        y = F.relu(y)
        y = self.post_pool_dense2(y)
        y = F.relu(y)
        y = self.final(y)
        y = F.softmax(y)

        return y


### p(c | y) parameterised by theta
class ContextDecoder(to.nn.Module):
    def __init__(self):
        super().__init__()

        # Input is dim(y)
        self.dense1 = to.nn.Linear(num_y_labels, dense_layer_size)
        self.batchnorm1 = to.nn.BatchNorm1d(dense_layer_size)
        self.dense2 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm2 = to.nn.BatchNorm1d(dense_layer_size)
        self.dense3 = to.nn.Linear(dense_layer_size, dense_layer_size)
        self.batchnorm3 = to.nn.BatchNorm1d(dense_layer_size)
        self.final = to.nn.Linear(dense_layer_size, 2 * context_dimension)

    def forward(self, y):
        output = self.dense1(y)
        output = apply_batch_norm(self.batchnorm1, output)
        output = F.relu(output)
        output = self.dense2(output)
        output = apply_batch_norm(self.batchnorm2, output)
        output = F.relu(output)
        output = self.dense3(output)
        output = apply_batch_norm(self.batchnorm3, output)
        output = F.relu(output)
        output = self.final(output)

        return output[:, context_dimension:], output[:, :context_dimension]


def apply_batch_norm(batch_norm, data):
    original_shape = data.shape
    data_list = data.view(-1, batch_norm.num_features)
    data_list = batch_norm(data_list)
    return data_list.view(original_shape)


def generate_samples_with_background(network, images, labels, timestamp, device, iteration=0):
    #Select first 50 test images
    images = images[60000:60050]
    sample_size = 50
    spatial = np.zeros([50, sample_size, 2])
    
    #Generate samples points
    grid = np.array([[i, j] for j in range(27, -1, -1) for i in range(28)])
    for i, image in enumerate(images):
        replace = True if (sum(image > 0) < sample_size) else False
        ix = np.random.choice(range(28*28), size=sample_size,
                              p=image/sum(image), replace=replace)
        spatial[i, :, :] = grid[ix] + np.random.uniform(0, 1, (sample_size, 2))
        
    generated_samples = generate_samples_like(network, spatial, labels, timestamp, device, all_datasets=True, make_plots=False)
    fig, axs = plt.subplots(10, 10, figsize=(8, 8))
    axs = axs.flatten()
    for i in range(100):
        if i % 2 == 0:
            axs[i].imshow(np.flipud(images[int(i/2)].reshape(28,28)), cmap='gray', interpolation='none')
            axs[i].scatter(spatial[int(i/2), :, 0], spatial[int(i/2), :, 1], s=2, c='b')
        else:
            axs[i].imshow(np.flipud(images[int(i/2)].reshape(28,28)), cmap='gray', interpolation='none')
            axs[i].scatter(generated_samples[int(i/2)][:, 0].cpu(), generated_samples[int(i/2)][:, 1].cpu(), s=2, c='r')
            
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlim([0, 27])
        axs[i].set_ylim([0, 27])
        axs[i].set_aspect('equal', adjustable='box')
    #plt.show()
    
    plt.savefig("results/{}/samples_iteration_{}.png".format(timestamp, iteration))
    plt.close()

    
def generate_samples_like(network, datasets, labels, timestamp, device, iteration=0, all_datasets=False, make_plots = True):
    with to.no_grad():
        # Make batch_norm work properly by placing it in evaluation mode
        network.eval()
        if not all_datasets:
            # Select an example of each digit from the dataset
            sample_digits = [None] * 10
            for dataset in datasets:
                if not any(x is None for x in sample_digits):
                    break
                label = np.argmax(dataset['label'])
                if sample_digits[label] is None:
                    sample_digits[label] = dataset['dataset']
        else:
            sample_digits = datasets

        generated_samples = []
        for digit_idx in range(len(sample_digits)):
            digit = sample_digits[digit_idx]
            label = labels[digit_idx]
            # Hack - we're supposed to have a sample for each digit
            if digit is None:
                continue
                
            samples = network.generate_like(to.as_tensor(digit, dtype=to.float).unsqueeze(dim=0).to(device),
                                            to.as_tensor(label, dtype=to.float).unsqueeze(dim=0).to(device))
            generated_samples.append(samples)
            
            if make_plots:
                fig, axs = plt.subplots(1,2, figsize=(21,7))
                axs[0].set_xlim([-2, 30])
                axs[0].set_ylim([-2, 30])
                axs[1].set_xlim([-2, 30])
                axs[1].set_ylim([-2, 30])
                
                axs[0].scatter(sample_digits[digit_idx][:,0], sample_digits[digit_idx][:,1])
                axs[1].scatter(samples.cpu()[:,0], samples.cpu()[:,1])
                plt.savefig("results/{}/samples_{}_iteration_{}.png".format(timestamp, digit_idx, iteration))
                plt.close()
            
        # Reset network to training mode
        network.train()
        
        return generated_samples


def plot_digit_dataset(digits, labels, timestamp, iteration, make_plots=True):
    """Take a collection of generated digit point clouds, and plot them."""
    with to.no_grad():
        fig, axs = plt.subplots(10, 10)
        axs = axs.flatten()
        for index, digit in enumerate(digits[:100]):
            axs[index].axis('equal')
            axs[index].set_xlim([0, 28])
            axs[index].set_ylim([0, 28])
            axs[index].set_xticks([])
            axs[index].set_yticks([])
            axs[index].scatter(digit[:, 0].cpu(), digit[:, 1].cpu(), s=1, alpha=0.3)
            axs[index].text(0, 14, labels[index].argmax().item())

        if make_plots:
            plt.show()
        else:
            plt.savefig("results/{}/test_generations_iteration_{}.png".format(timestamp, iteration))
            plt.close()


def visualize_data(network, dataset, images, labels, iteration, timestamp, device):
    with to.no_grad():
        generate_samples_with_background(network, images, labels, timestamp, device, iteration=iteration)
        test_set_labels = to.from_numpy(dataset['label']).to(device)
        generated_digits = network.generate(test_set_labels, samples_per_dataset=250)
        plot_digit_dataset(generated_digits, test_set_labels, timestamp, iteration, make_plots=False)


def initialise(labelled, unsupervision=0):
    train_dataset = spd.SpatialMNISTDataset(data_dir, split='train', unsupervision=unsupervision)
    test_dataset = spd.SpatialMNISTDataset(data_dir, split='test')
    train_dataloader = to.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    if labelled:
        label_prior_probabilities = to.from_numpy(train_dataset[train_dataset.unsupervision_mask]['label']).to(device).sum(dim=0) / train_dataset.unsupervision_mask.sum()
        label_prior = to.distributions.categorical.Categorical(probs=label_prior_probabilities)
        network = ls.LabelStatistician(num_stochastic_layers, context_dimension, label_prior, LatentDecoder, ObservationDecoder, StatisticNetwork, InferenceNetwork, ClassificationNetwork, ContextDecoder, device)
    else:
        network = ns.NeuralStatistician(num_stochastic_layers, context_dimension, LatentDecoder, ObservationDecoder, StatisticNetwork, InferenceNetwork, device)

    return {'network': network,
            'train_dataloader': train_dataloader,
            'test_dataset': test_dataset}


def main(labelled, unsupervision=0):
    if unsupervision is None:
        unsupervision = 0
    init_objects = initialise(labelled, unsupervision)
    network, train_dataloader, test_dataset = init_objects['network'], init_objects['train_dataloader'], init_objects['test_dataset']
        
    timestamp = datetime.datetime.now()
    path = "results/{}".format(timestamp)
    try: os.mkdir(path)
    except FileExistsError: pass

    optimiser_func = lambda parameters: to.optim.Adam(parameters, lr=1e-3)
    
    #Load the actual mnist data so that we can plot the actual images in the background
    images, labels = sc.load_data()
    # images, labels = None, None
    test_func = lambda network, iteration: visualize_data(network, test_dataset[:100], images, labels, iteration, timestamp, device)

    test_func(network, 0)
    network.run_training(train_dataloader, 500, optimiser_func, test_func, device)

    network.serialise("results/{}/trained_mnist_model".format(timestamp))
    
    return network


if __name__ == '__main__':
    network = main(True, 0.25)


def test_labels():
    unsupervision=0
    train_dataset = spd.SpatialMNISTDataset(data_dir, split='train', unsupervision=unsupervision)
    train_dataloader = to.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    for batch in train_dataloader:
        for digit in range(len(batch)):
            plt.figure()
            plt.scatter(batch['dataset'][digit][:,0], batch['dataset'][digit][:,1])
            plt.text(0, 0, batch['label'][digit].argmax().item())
            plt.show()

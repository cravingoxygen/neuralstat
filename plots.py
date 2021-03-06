import neural_statistician as ns
import one_dimensional_test as one
import matplotlib.pyplot as plt
import torch as to

def plot_contexts_50_iterations_1D(network):
    network = ns.NeuralStatistician(1, 3,
                                    one.LatentDecoder, one.ObservationDecoder,
                                    one.StatisticNetwork, one.InferenceNetwork, to.device("cuda"))
    network.deserialise("./results/2019-03-07 12:06:00 50 Synthetic Iterations, Additional Data Saving/trained_model")

    def config():
        plt.title("Context Points after 50 Iterations")
        plt.legend(["Exponential", "Normal", "Uniform", "Laplace"], loc='lower right')
        plt.tight_layout()

    one.plot_contexts_by_distribution(network, None, to.device("cuda"), save_plot=False, config=config)


def plot_contexts_0_iterations_1D():
    network = ns.NeuralStatistician(1, 3,
                                    one.LatentDecoder, one.ObservationDecoder,
                                    one.StatisticNetwork, one.InferenceNetwork, to.device("cuda"))
    network.deserialise("./results/2019-03-07 12:06:00 50 Synthetic Iterations, Additional Data Saving/model_iteration_0")

    def config():
        plt.title("Context Points after 0 Iterations")
        plt.legend(["Exponential", "Normal", "Uniform", "Laplace"], loc='lower right')
        plt.tight_layout()

    one.plot_contexts_by_distribution(network, None, to.device("cuda"), save_plot=False, config=config)


def plot_contexts_25_iterations_1D():
    network = ns.NeuralStatistician(1, 3,
                                    one.LatentDecoder, one.ObservationDecoder,
                                    one.StatisticNetwork, one.InferenceNetwork, to.device("cuda"))
    network.deserialise("./results/2019-03-07 12:06:00 50 Synthetic Iterations, Additional Data Saving/model_iteration_25")

    def config():
        plt.title("Context Points after 25 Iterations")
        plt.legend(["Exponential", "Normal", "Uniform", "Laplace"], loc='lower right')
        plt.tight_layout()

    one.plot_contexts_by_distribution(network, None, to.device("cuda"), save_plot=False, config=config)


def plot_final_contexts_by_mean():
    network = ns.NeuralStatistician(1, 3,
                                    one.LatentDecoder, one.ObservationDecoder,
                                    one.StatisticNetwork, one.InferenceNetwork, to.device("cpu"))
    network.deserialise("./results/2019-03-07 12:06:00 50 Synthetic Iterations, Additional Data Saving/trained_model")

    def config(scalar_map):
        plt.ion()
        colour_bar = plt.colorbar(scalar_map)
        colour_bar.set_label("Distribution Mean", rotation=270, va='bottom')
        plt.title("Context Points by Distribution Mean after 50 Iterations")
        plt.tight_layout()

    one.plot_contexts_by_value(network, to.device("cpu"), "mean", config=config)


def plot_final_contexts_by_variance():
    network = ns.NeuralStatistician(1, 3,
                                    one.LatentDecoder, one.ObservationDecoder,
                                    one.StatisticNetwork, one.InferenceNetwork, to.device("cpu"))
    network.deserialise("./results/2019-03-07 12:06:00 50 Synthetic Iterations, Additional Data Saving/trained_model")

    def config(scalar_map):
        plt.ion()
        colour_bar = plt.colorbar(scalar_map)
        colour_bar.set_label("Distribution Variance", rotation=270, va='bottom')
        plt.title("Context Points by Distribution Variance after 50 Iterations")
        plt.tight_layout()

    one.plot_contexts_by_value(network, to.device("cpu"), "variance", config=config)

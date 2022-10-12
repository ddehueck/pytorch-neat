from neat.phenotype.feed_forward import FeedForwardNet
import numpy as np


class EnsembleFeedForwardNet():
    def __init__(self, genomes, config):
        self.phenotypes = [FeedForwardNet(genome, config)
                           for genome in genomes]
        self.config = config

    def __call__(self, input):
        results = [phenotype(input) for phenotype in self.phenotypes]
        return np.mean(results, axis=0)

import logging
import copy

import torch
import numpy as np

from neat.phenotype.feed_forward import FeedForwardNet

logger = logging.getLogger(__name__)


def rand_uni_val():
    """
    Gets a random value from a uniform distribution on the interval [0, 1]
    :return: Float
    """
    return float(torch.rand(1))


def rand_bool():
    """
    Returns a random boolean value
    :return: Boolean
    """
    return rand_uni_val() <= 0.5


def get_best_genome(population):
    """
    Gets best genome out of a population
    :param population: List of Genome instances
    :return: Genome instance
    """
    population_copy = copy.deepcopy(population)
    population_copy.sort(key=lambda g: g.fitness, reverse=True)

    return population_copy[0]


def cache_genomes_results(genomes, dataset, config):
    genomes_to_results = {}
    for genome in genomes:
        results = []
        phenotype = FeedForwardNet(genome, config)
        phenotype.to(config.DEVICE)
        for input in dataset:
            input.to(config.DEVICE)
            prediction = phenotype(input)
            results.append(prediction.numpy())
        genomes_to_results[genome] = np.array(results)
    return genomes_to_results

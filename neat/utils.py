import logging
import copy

import torch


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
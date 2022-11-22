import logging
import copy
import math
import random

import torch
import numpy as np

from neat.phenotype.feed_forward import FeedForwardNet
from neat.species import Species

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


def create_prediction_map(genomes, dataset, config):
    """
    Creates a prediction mapping of genomes to their 2D arrays of predictions

    Each array in the 2D array is the prediction (activation outputs) for the given input
    There is one array in each 2D array for each input in the dataset

    Parameters:
        genomes (iterable): the population of genomes
        dataset (list of tensors): the list of tensor inputs
        config (neat.config): the neat configuration

    Returns:
        dict of genome to the genome's predictions
    """
    genomes_to_results = {}
    for genome in genomes:
        results = []
        phenotype = FeedForwardNet(genome, config)
        phenotype.to(config.DEVICE)
        for input in dataset:
            input.to(config.DEVICE)
            prediction = phenotype(input).to('cpu')
            results.append(prediction.detach().numpy())
        genomes_to_results[genome] = np.array(results)
    return genomes_to_results


def random_ensemble_generator(genomes, k=None, limit=None):
    """
    A generator that randomly picks an ensemble from the given genomes of length k

    Parameters:
        genomes (iterable): the whole population of genomes to sample from
        k (None | int): None (for random size ensembles) or the size of ensembles to create
        limit: the number of ensembles to yield before iterable exhaustion

    Yields:
       A set of genomes to use in an ensemble
    """
    genomes = list(genomes)
    n = len(genomes)
    seen = set()

    max_limit = 2**n - 1 if k is None else math.comb(n, k)
    max_limit /= 2
    limit = max_limit if limit is None else min(limit, max_limit)

    while len(seen) < limit:
        ensemble_length = random.randint(1, n) if k is None else k
        all_indices = list(range(n))
        random.shuffle(all_indices)
        ensemble_indices = frozenset(all_indices[0:ensemble_length])
        if ensemble_indices not in seen:
            seen.add(ensemble_indices)
            ensemble = {genomes[i] for i in ensemble_indices}
            yield ensemble


def random_ensemble_generator_for_static_genome(genome, genomes, k=None, limit=None):
    """
    A generator that randomly picks the rest of an ensemble of size k including a given genome

    Parameters:
        genome: the genome to include in the resulting ensembles
        genomes (iterable): the whole population of genomes to sample from
        k (None | int): None (for random size ensembles) or the size of ensembles to create
        limit: the number of ensembles to yield before iterable exhaustion

    Yields:
        A set of genomes to use in an ensemble
    """
    genomes = [g for g in genomes if g != genome]
    k = None if k is None else k - 1
    for ensemble in random_ensemble_generator(genomes=genomes, k=k, limit=limit):
        yield {genome, *ensemble}


def speciate(genomes, speciation_threshold):
    """
    Copied and modified from population.py
    Creates a list of species, where each species is a list of genomes
    The speciation takes place based on genomes' genetic diversity
    """

    species = []

    def speciate(genome):
        for s in species:
            if Species.species_distance(genome, s.model_genome) <= speciation_threshold:
                s.members.append(genome)
                return

        # Did not match any current species. Create a new one
        new_species = Species(len(species), genome, 0)
        new_species.members.append(genome)
        species.append(new_species)

    for genome in genomes:
        speciate(genome)

    return [s.members for s in species]

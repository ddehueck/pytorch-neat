"""

This module contains crossover methods as described in Kenneth O. Stanley's NEAT
paper.

Todo:
    * Allow other types of crossover?

"""
import logging
from copy import deepcopy

import neat.utils as utils
from neat.genotype.genome import Genome


logger = logging.getLogger(__name__)


def crossover(genome_1, genome_2, config):
    """
    Crossovers two Genome instances as described in the original NEAT implementation
    :param genome_1: First Genome Instance
    :param genome_2: Second Genome Instance
    :param config: Experiment's configuration class
    :return: A child Genome Instance
    """

    child = Genome()
    best_parent, other_parent = order_parents(genome_1, genome_2)

    # Crossover connections
    # Randomly add matching genes from both parents
    for c_gene in best_parent.connection_genes:
        matching_gene = other_parent.get_connect_gene(c_gene.innov_num)

        if matching_gene is not None:
            # Randomly choose where to inherit gene from
            if utils.rand_bool():
                child_gene = deepcopy(c_gene)
            else:
                child_gene = deepcopy(matching_gene)

        # No matching gene - is disjoint or excess
        # Inherit disjoint and excess genes from best parent
        else:
            child_gene = deepcopy(c_gene)

        # Apply rate of disabled gene being re-enabled
        if not child_gene.is_enabled:
            is_reenabeled = utils.rand_uni_val() <= config.CROSSOVER_REENABLE_CONNECTION_GENE_RATE
            enabled_in_best_parent = best_parent.get_connect_gene(child_gene.innov_num).is_enabled

            if is_reenabeled or enabled_in_best_parent:
                child_gene.is_enabled = True

        child.add_connection_copy(child_gene)

    # Crossover Nodes
    # Randomly add matching genes from both parents
    for n_gene in best_parent.node_genes:
        matching_gene = other_parent.get_node_gene(n_gene.id)

        if matching_gene is not None:
            # Randomly choose where to inherit gene from
            if utils.rand_bool():
                child_gene = deepcopy(n_gene)
            else:
                child_gene = deepcopy(matching_gene)

        # No matching gene - is disjoint or excess
        # Inherit disjoint and excess genes from best parent
        else:
            child_gene = deepcopy(n_gene)

        child.add_node_copy(child_gene)

    return child


def order_parents(parent_1, parent_2):
    """
    Orders parents with respect to fitness
    :param parent_1: First Parent Genome
    :param parent_2: Secont Parent Genome
    :return: Two Genome Instances
    """

    # genotype.cpp line 2043
    # Figure out which genotype is better
    # The worse genotype should not be allowed to add excess or disjoint genes
    # If they are the same, use the smaller one's disjoint and excess genes

    best_parent = parent_1
    other_parent = parent_2

    len_parent_1 = len(parent_1.connection_genes)
    len_parent_2 = len(parent_2.connection_genes)

    if parent_1.fitness == parent_2.fitness:
        if len_parent_1 == len_parent_2:
            # Fitness and Length equal - randomly choose best parent
            if utils.rand_bool():
                best_parent = parent_2
                other_parent = parent_1
        # Choose minimal parent
        elif len_parent_2 < len_parent_1:
            best_parent = parent_2
            other_parent = parent_1

    elif parent_2.fitness > parent_1.fitness:
        best_parent = parent_2
        other_parent = parent_1

    return best_parent, other_parent

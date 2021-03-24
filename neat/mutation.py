import logging
import random

import neat.utils as utils


logger = logging.getLogger(__name__)


def mutate(genome, config):
    """
    Applies connection and structural mutations at proper rate.
    Connection Mutations: Uniform Weight Perturbation or Replace Weight Value with Random Value
    Structural Mutations: Add Connection and Add Node
    :param genome: Genome to be mutated
    :param config: Experiments' configuration file
    :return: None
    """

    if utils.rand_uni_val() < config.CONNECTION_MUTATION_RATE:
        for c_gene in genome.connection_genes:
            if utils.rand_uni_val() < config.CONNECTION_PERTURBATION_RATE:
                perturb = utils.rand_uni_val() * random.choice([1, -1])
                c_gene.weight += perturb
            else:
                c_gene.set_rand_weight()

    if utils.rand_uni_val() < config.ADD_NODE_MUTATION_RATE:
        genome.add_node_mutation()

    if utils.rand_uni_val() < config.ADD_CONNECTION_MUTATION_RATE:
        genome.add_connection_mutation()

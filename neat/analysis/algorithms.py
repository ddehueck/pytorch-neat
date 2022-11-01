import numpy as np
from utils import random_ensemble_generator

"""
A set of algoritms needed for each trial's analysis
These algorithms include:
- Random ensemble (control)
- Greedy 1
- Greedy 2
- Diversity selection (round robin by speciation)

Each algorithm here:
Consumes:
    - genomes prediction map
      - each genome key in the map *must* have the fitness member set
    - evaluate function (e.g. lambda) that returns accuracy of a given ensemble
      - given ensemble is represented as a list of the
        2d predicition arrays for the ensembled genomes
Returns:
    - list of size len(pred_map) that has the accuracy
      of the (i+1)-size ensemble created by the algorithm for any index i
"""


def random_selection_accuracies(pred_map, eval_func, ensembles_per_k=1):
    accuracies = []
    for k in range(1, len(pred_map) + 1):
        k_acc = []
        for ensemble in random_ensemble_generator(
            genomes=pred_map.keys(), k=k, limit=ensembles_per_k
        ):
            ensemble_member_results = [pred_map[x] for x in ensemble]
            k_acc.append(eval_func(ensemble_member_results))
        accuracies.append(np.mean(k_acc))
    return accuracies


def greedy_1_selection_accuracies(pred_map, eval_func):
    # Some variables needed for the greedy algorithm
    # genomes_left is the genomes left to choose from
    # genomes_picked is the current best predicted k-wise ensemble
    genomes_left = {*pred_map.keys()}
    genomes_picked = []
    accuracies = []

    # Remove the genome that improves ensemble the most after each round
    while genomes_left:

        # Initialize this round's variables
        best_accuracy = float('-inf')
        best_genome = None

        # Find the genome that best improves the current ensemble (genomes_picked)
        for genome in genomes_left:
            ensemble = [*genomes_picked, genome]
            ensemble_member_results = [pred_map[x] for x in ensemble]
            ensemble_accuracy = eval_func(ensemble_member_results)
            if ensemble_accuracy > best_accuracy:
                best_accuracy = ensemble_accuracy
                best_genome = genome

        # Some housekeeping to finish off the round
        genomes_left.remove(best_genome)
        genomes_picked.append(best_genome)
        accuracies.append(best_accuracy)
    return accuracies


def greedy_2_selection_accuracies(pred_map, eval_func):
    genomes = list(pred_map.keys())
    genomes.sort(reverse=True, key=lambda g: g.fitness)
    accuracies = []
    for k in range(1, len(pred_map) + 1):
        ensemble = genomes[0:k]
        ensemble_member_results = [pred_map[x] for x in ensemble]
        accuracies.append(eval_func(ensemble_member_results))
    return accuracies


def diversity_rr_selection_accuracies(pred_map, eval_func, speciation_threshold=0.8):
    # TODO round robin style picking from different species (based on threshold)
    # each species sorted by top accuracy down to lowest accuracy
    pass

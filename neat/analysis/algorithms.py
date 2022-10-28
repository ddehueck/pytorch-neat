import numpy as np
from utils import random_ensemble_generator

'''
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
'''


def random_selection_accuracies(pred_map, eval_func, ensembles_per_k=1):
    accuracies = []
    for k in range(1, len(pred_map) + 1):
        k_acc = []
        for ensemble in random_ensemble_generator(genomes=pred_map.keys(), k=k, limit=ensembles_per_k):
            ensemble_member_results = [pred_map[x] for x in ensemble]
            k_acc.append(eval_func(ensemble_member_results))
        accuracies.append(np.mean(k_acc))
    return accuracies

def greedy_1_selection_accuracies(pred_map, eval_func):
    # TODO pick network that best improves ensemble performance, iterate k times
    pass

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
